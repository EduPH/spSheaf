"""
Local and global intrinsic dimensionality estimation.

Provides tools to estimate the intrinsic dimension of the expression
manifold at multiple scales:

- **Per-patch**: PCA and Isomap quality curves on atlas patches
- **Local field**: Per-vertex dimension via neighborhood PCA / Isomap
- **Global**: Full-slice PCA and Isomap (optional subsampling)
- **Gene selection**: Manifold-aware gene importance ranking

Key finding: The intrinsic dimensionality is spatially non-uniform
(typically 2–5 per patch), which means a single global latent
dimension is insufficient — motivating local/adaptive approaches
such as the sheaf framework with spatially varying stalk spaces.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors


# ============================================================================
# Atlas cover
# ============================================================================

def build_atlas_cover(grid_coords, patch_size=400.0, overlap=0.5,
                      min_points=30, verbose=True):
    """
    Build overlapping spatial patches for the atlas.

    Parameters
    ----------
    grid_coords : ndarray, shape (N, 2)
        Grid point positions.
    patch_size : float
        Side length of each square patch (μm).
    overlap : float
        Fraction of overlap between adjacent patches.
    min_points : int
        Minimum number of grid points to keep a patch.
    verbose : bool
        Print patch statistics.

    Returns
    -------
    subset_assignments : list of ndarray
        Each entry is an index array of grid points in that patch.
    patch_centers : list of [float, float]
        Spatial center of each patch.
    """
    step = patch_size * (1.0 - overlap)
    half = patch_size / 2
    x_min, y_min = grid_coords.min(axis=0)
    x_max, y_max = grid_coords.max(axis=0)

    x_centers = np.arange(x_min + half, x_max + step, step)
    y_centers = np.arange(y_min + half, y_max + step, step)

    subset_assignments = []
    patch_centers = []
    for cx in x_centers:
        for cy in y_centers:
            mask = (
                (grid_coords[:, 0] >= cx - half)
                & (grid_coords[:, 0] < cx + half)
                & (grid_coords[:, 1] >= cy - half)
                & (grid_coords[:, 1] < cy + half)
            )
            idx = np.where(mask)[0]
            if len(idx) >= min_points:
                subset_assignments.append(idx)
                patch_centers.append([cx, cy])

    if verbose:
        sizes = [len(s) for s in subset_assignments]
        print(
            f"  Patches: {len(subset_assignments)}, "
            f"points/patch: min={min(sizes)}, max={max(sizes)}, "
            f"mean={np.mean(sizes):.0f}"
        )

    return subset_assignments, patch_centers


# ============================================================================
# Dimension estimation helpers
# ============================================================================

def _dim_at_threshold(quality_curve, threshold):
    """
    Find the first dimension where quality >= threshold.

    Returns None if the threshold is never reached.
    """
    above = np.where(quality_curve >= threshold)[0]
    if len(above) == 0:
        return None
    return int(above[0]) + 1  # 1-based


# ============================================================================
# Per-patch dimension estimation
# ============================================================================

def estimate_dim_per_patch(fibers_norm, subset_assignments, max_dim=50,
                           n_patches=10, n_neighbors_iso=10, verbose=True):
    """
    Estimate intrinsic dimension in each atlas patch using PCA and Isomap.

    The gap between PCA and Isomap estimates reveals manifold curvature:
    if Isomap needs fewer dimensions, the data lies on a curved manifold
    that PCA overestimates.

    Parameters
    ----------
    fibers_norm : ndarray, shape (N, G)
        Normalized fibers.
    subset_assignments : list of ndarray
        Patch index arrays from :func:`build_atlas_cover`.
    max_dim : int
        Maximum embedding dimension to test.
    n_patches : int
        Number of patches to analyze (first n).
    n_neighbors_iso : int
        Isomap neighborhood size.
    verbose : bool
        Print per-patch results.

    Returns
    -------
    results : list of dict
        Per-patch curves and estimated dimensions at 90%/95% thresholds.
    """
    n_patches = min(n_patches, len(subset_assignments))
    n_genes = fibers_norm.shape[1]
    max_dim = min(max_dim, n_genes - 1)

    results = []
    for i in range(n_patches):
        patch_data = fibers_norm[subset_assignments[i]]
        n_pts = len(patch_data)
        n_comp = min(max_dim, n_pts - 1)

        # PCA
        pca = PCA(n_components=n_comp).fit(patch_data)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        # Isomap
        max_d_iso = min(max_dim, n_pts - 2)
        residuals = []
        for d in range(1, max_d_iso + 1):
            iso = Isomap(
                n_components=d,
                n_neighbors=min(n_neighbors_iso, n_pts - 1),
            )
            iso.fit(patch_data)
            residuals.append(iso.reconstruction_error())
        residuals = np.array(residuals)
        iso_quality = 1 - residuals / (residuals[0] + 1e-10)

        dim_pca_90 = _dim_at_threshold(cumvar, 0.90)
        dim_pca_95 = _dim_at_threshold(cumvar, 0.95)
        dim_iso_90 = _dim_at_threshold(iso_quality, 0.90)
        dim_iso_95 = _dim_at_threshold(iso_quality, 0.95)

        results.append(dict(
            n_pts=n_pts, cumvar=cumvar, iso_quality=iso_quality,
            dim_pca_90=dim_pca_90, dim_pca_95=dim_pca_95,
            dim_iso_90=dim_iso_90, dim_iso_95=dim_iso_95,
        ))

        if verbose:
            def _fmt(d):
                return f"{d:2d}" if d is not None else f">{n_comp}"
            print(
                f"  Patch {i:2d} ({n_pts:4d} pts): "
                f"PCA 90%={_fmt(dim_pca_90)}, 95%={_fmt(dim_pca_95)} | "
                f"Isomap 90%={_fmt(dim_iso_90)}, 95%={_fmt(dim_iso_95)}"
            )

    return results


def estimate_dim_all_patches(fibers_norm, subset_assignments, patch_centers,
                             max_dim=50, n_neighbors_iso=10, verbose=True):
    """
    Lightweight 90%-threshold dimension estimate across ALL patches.

    Parameters
    ----------
    fibers_norm : ndarray, shape (N, G)
    subset_assignments : list of ndarray
    patch_centers : list of [float, float]
    max_dim : int
    n_neighbors_iso : int
    verbose : bool

    Returns
    -------
    pca_dims : ndarray, shape (n_patches,)
    iso_dims : ndarray, shape (n_patches,)
    """
    n_patches = len(subset_assignments)
    n_genes = fibers_norm.shape[1]
    max_dim = min(max_dim, n_genes - 1)

    pca_dims = np.zeros(n_patches)
    iso_dims = np.zeros(n_patches)

    for i in range(n_patches):
        patch_data = fibers_norm[subset_assignments[i]]
        n_pts = len(patch_data)
        n_comp = min(max_dim, n_pts - 1)

        # PCA
        pca = PCA(n_components=n_comp).fit(patch_data)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        d = _dim_at_threshold(cumvar, 0.90)
        pca_dims[i] = d if d is not None else n_comp

        # Isomap (cap at 15 dims for speed)
        max_d_iso = min(15, n_pts - 2)
        residuals = []
        for d in range(1, max_d_iso + 1):
            iso = Isomap(
                n_components=d,
                n_neighbors=min(n_neighbors_iso, n_pts - 1),
            )
            iso.fit(patch_data)
            residuals.append(iso.reconstruction_error())
        residuals = np.array(residuals)
        iso_quality = 1 - residuals / (residuals[0] + 1e-10)
        d_iso = _dim_at_threshold(iso_quality, 0.90)
        iso_dims[i] = d_iso if d_iso is not None else max_d_iso

        if verbose and ((i + 1) % 20 == 0 or i == n_patches - 1):
            print(f"    {i + 1}/{n_patches} patches done")

    if verbose:
        print(
            f"  PCA 90%:    mean={pca_dims.mean():.1f}, "
            f"median={np.median(pca_dims):.0f}, "
            f"range=[{pca_dims.min():.0f}, {pca_dims.max():.0f}]"
        )
        print(
            f"  Isomap 90%: mean={iso_dims.mean():.1f}, "
            f"median={np.median(iso_dims):.0f}, "
            f"range=[{iso_dims.min():.0f}, {iso_dims.max():.0f}]"
        )

    return pca_dims, iso_dims


# ============================================================================
# Local dimension fields
# ============================================================================

def local_dim_field_pca(fibers, grid_coords, k=50, threshold=0.90,
                        verbose=True):
    """
    Per-vertex local intrinsic dimension via PCA on k nearest neighbors.

    Parameters
    ----------
    fibers : ndarray, shape (N, G)
        Normalized fibers.
    grid_coords : ndarray, shape (N, 2)
        Spatial positions.
    k : int
        Neighborhood size.
    threshold : float
        Cumulative variance threshold.
    verbose : bool
        Print progress.

    Returns
    -------
    dim_field : ndarray, shape (N,)
        Local PCA dimension at each grid point.
    """
    if verbose:
        print("  Computing PCA local dimension field...")

    nbrs = NearestNeighbors(n_neighbors=k).fit(grid_coords)
    _, indices = nbrs.kneighbors(grid_coords)
    dim_field = np.zeros(len(grid_coords))

    for i in range(len(grid_coords)):
        local_data = fibers[indices[i]]
        pca = PCA().fit(local_data)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dim_field[i] = np.searchsorted(cumvar, threshold) + 1

    return dim_field


def local_dim_field_isomap(fibers, grid_coords, k=50, max_dim=10,
                           drop_threshold=0.1, verbose=True):
    """
    Per-vertex local dimension via Isomap reconstruction error elbow.

    Slower than PCA but captures nonlinear manifold structure.

    Parameters
    ----------
    fibers : ndarray, shape (N, G)
    grid_coords : ndarray, shape (N, 2)
    k : int
        Neighborhood size.
    max_dim : int
        Maximum dimension to test.
    drop_threshold : float
        Minimum relative drop in reconstruction error.
    verbose : bool

    Returns
    -------
    dim_field : ndarray, shape (N,)
    """
    if verbose:
        print("  Computing Isomap local dimension field...")

    nbrs = NearestNeighbors(n_neighbors=k).fit(grid_coords)
    _, indices = nbrs.kneighbors(grid_coords)
    dim_field = np.zeros(len(grid_coords))

    for i in range(len(grid_coords)):
        local_data = fibers[indices[i]]
        max_d = min(max_dim, k - 2)
        residuals = []
        for d in range(1, max_d + 1):
            iso = Isomap(n_components=d, n_neighbors=min(10, k - 1))
            iso.fit(local_data)
            residuals.append(iso.reconstruction_error())
        residuals = np.array(residuals)
        drops = residuals[:-1] - residuals[1:]
        rel_drops = drops / (residuals[:-1] + 1e-10)
        dim_est = 1
        for d_idx in range(len(rel_drops)):
            if rel_drops[d_idx] > drop_threshold:
                dim_est = d_idx + 2
        dim_field[i] = dim_est

        if verbose and i % 500 == 0:
            print(f"    {i}/{len(grid_coords)}, local dim={dim_est}")

    return dim_field


# ============================================================================
# Manifold gene selection
# ============================================================================

def select_manifold_genes(fibers, grid_coords, gene_names, latent_dim=3,
                          k=50, top_n=30, verbose=True):
    """
    Select genes contributing most to local manifold structure.

    At each neighborhood, PCA loadings (weighted by explained variance)
    identify which genes span the manifold. Aggregating over all
    neighborhoods gives globally important manifold genes.

    Parameters
    ----------
    fibers : ndarray, shape (N, G)
        Normalized fibers.
    grid_coords : ndarray, shape (N, 2)
        Spatial positions.
    gene_names : list of str
        Gene names.
    latent_dim : int
        Number of PCA components for loading analysis.
    k : int
        Neighborhood size.
    top_n : int
        Number of top genes to report.
    verbose : bool

    Returns
    -------
    gene_sel : dict
        Keys: ranking, importance, cum_importance, n_90, n_95,
        sorted_importance, selected_names.
    """
    if verbose:
        print("  Selecting manifold genes...")

    nbrs = NearestNeighbors(n_neighbors=k).fit(grid_coords)
    _, indices = nbrs.kneighbors(grid_coords)

    gene_importance = np.zeros(fibers.shape[1])
    for i in range(len(grid_coords)):
        local_data = fibers[indices[i]]
        pca = PCA(n_components=latent_dim).fit(local_data)
        loadings_sq = pca.components_ ** 2
        weighted = loadings_sq * pca.explained_variance_ratio_[:, None]
        gene_importance += weighted.sum(axis=0)
    gene_importance /= len(grid_coords)

    ranking = np.argsort(gene_importance)[::-1]
    sorted_imp = gene_importance[ranking]
    cum_imp = np.cumsum(sorted_imp) / np.sum(sorted_imp)

    n_90 = int(np.searchsorted(cum_imp, 0.90)) + 1
    n_95 = int(np.searchsorted(cum_imp, 0.95)) + 1

    if verbose:
        print(f"  Genes for 90% manifold variance: {n_90} / {len(gene_names)}")
        print(f"  Genes for 95% manifold variance: {n_95} / {len(gene_names)}")

    return dict(
        ranking=ranking,
        importance=gene_importance,
        cum_importance=cum_imp,
        n_90=n_90,
        n_95=n_95,
        sorted_importance=sorted_imp,
        selected_names=[
            gene_names[ranking[j]]
            for j in range(min(top_n, len(gene_names)))
        ],
    )
