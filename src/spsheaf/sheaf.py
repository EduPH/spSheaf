"""
Cellular sheaf construction and boundary detection.

This module implements the core sheaf-theoretic machinery:

1. **Stalks** — At each vertex v, the stalk F(v) ≅ ℝ^d is the column
   space of the local PCA basis computed from the k nearest fibers.
   A uniform stalk dimension d is used to avoid block-structure
   artifacts in the sheaf Laplacian.

2. **Restriction maps** — For each edge e = (u, v), restriction maps
   F_{u◁e} and F_{v◁e} are obtained from the SVD of the stalk overlap
   matrix M_uv = U_u^T U_v. The singular values are the cosines of
   the principal angles between the two stalk subspaces.

3. **Boundary detection** — The sheaf disagreement on edge e measures
   how much the local PCA bases at u and v *rotate* relative to each
   other, which is the key geometric signal distinguishing sheaf-based
   boundary detection from standard distance metrics. Per-vertex
   boundary scores are obtained by averaging incident edge disagreements.

Key insight: Boundaries are where sheaf restriction maps show high
disagreement. This directly targets boundaries rather than trying to
extract cluster information from sheaf Laplacian eigenvectors (which
encode smooth within-domain gradients, not between-domain
discontinuities).
"""

import os

import numpy as np
import time
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def knn_indices(grid_coords, k_nbrs):
    """The k nearest grid points of every vertex, with deterministic ties.

    On a regular grid many neighbours sit at exactly the k-th distance, and
    which of them a k-d tree returns depends on the library build; the
    locally centered coordinates then differ between machines by up to 0.2
    (observed between macOS and Linux on the same cache). Ties are broken by
    vertex index, so the neighbourhood is a function of the grid alone.
    """
    k_query = min(len(grid_coords), 2 * k_nbrs)
    nbrs = NearestNeighbors(n_neighbors=k_query).fit(grid_coords)
    dist, idx = nbrs.kneighbors(grid_coords)
    dist = np.round(dist, 6)
    out = np.empty((len(grid_coords), k_nbrs), dtype=int)
    for i in range(len(grid_coords)):
        order = np.lexsort((idx[i], dist[i]))          # by distance, then index
        out[i] = idx[i][order][:k_nbrs]
    return out


def compute_stalks(fibers_norm, grid_coords, uniform_dim=6, k_nbrs=50,
                   verbose=True):
    """
    Compute stalks: local PCA basis at each vertex.

    Each stalk U_v ∈ ℝ^{G×d} contains the top-d principal components
    of the globally-centered fiber data in v's spatial neighborhood.

    Parameters
    ----------
    fibers_norm : ndarray, shape (N, G)
        Normalized expression fibers.
    grid_coords : ndarray, shape (N, 2)
        Spatial positions.
    uniform_dim : int
        Stalk dimension d. Empirically insensitive across [4, 50].
    k_nbrs : int
        Number of spatial neighbors for local PCA.
    verbose : bool
        Print summary.

    Returns
    -------
    stalks : list of ndarray, each shape (G, d)
        Orthonormal stalk bases.
    """
    N, G = fibers_norm.shape
    global_mean = fibers_norm.mean(axis=0)
    fibers_centered = fibers_norm - global_mean

    indices = knn_indices(grid_coords, k_nbrs)

    stalks = []
    for i in range(N):
        local_data = fibers_centered[indices[i]]
        pca = PCA(n_components=uniform_dim)
        pca.fit(local_data)
        U_v = pca.components_.T  # (G, d)
        stalks.append(U_v)

    if verbose:
        print(f"  Stalks: {N} vertices, uniform dim = {uniform_dim}")

    return stalks


def compute_restriction_maps(stalks, edges, stalk_dim, verbose=True):
    """
    Compute restriction maps via SVD alignment of adjacent stalks.

    For edge e = (u, v), the overlap matrix M_uv = U_u^T U_v ∈ ℝ^{d×d}
    is decomposed as M_uv = L Σ R^T. The restriction maps are:

        F_{u◁e} = L^T    (projects u's stalk into shared edge space)
        F_{v◁e} = R^T    (projects v's stalk into shared edge space)

    The singular values σ_1, ..., σ_d are the cosines of the principal
    angles between the two stalk subspaces.

    Parameters
    ----------
    stalks : list of ndarray
        Stalk bases from :func:`compute_stalks`.
    edges : list of (int, int)
        Spatial graph edges.
    stalk_dim : int
        Stalk dimension d (for fallback identity matrices).
    verbose : bool
        Print summary.

    Returns
    -------
    restrictions : dict
        Maps ``(vertex_idx, edge_idx)`` → ndarray of shape ``(d, d)``.
    """
    restrictions = {}
    for ei, (u, v) in enumerate(edges):
        U_u = stalks[u]
        U_v = stalks[v]

        M = U_u.T @ U_v
        try:
            L, S, Rt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            L = np.eye(stalk_dim)
            Rt = np.eye(stalk_dim)

        restrictions[(u, ei)] = L.T
        restrictions[(v, ei)] = Rt

    if verbose:
        print(f"  Restriction maps: {len(edges)} edges")

    return restrictions


#: Reference point of the local coordinates xi_v = U_v^T (f_v - c_v).
#:
#: ``tissue``  c_v = fbar, the mean fiber of the section. The disagreement
#:             then also carries the term (L^T U_u^T - R^T U_v^T)(mu_u - mu_v)
#:             at aligned stalks, i.e. the offset of the two neighbourhood
#:             means seen through the alignment. The default since 28 Sep
#:             2026: better on osmFISH and both SpaceTx sections.
#: ``local``   c_v = mu_v, the mean fiber of v's k_nbrs grid neighbours -- the
#:             neighbourhood the stalk is fitted on. xi_v is then v's own
#:             local-PCA score, a local section. Selected by the tuning grid
#:             on the MERFISH benchmark (0.571 against 0.548), and the
#:             construction the MERFISH ablation is reported at.
#: Set with SPSHEAF_REFERENCE=local|tissue; the tuning grids of
#: experiments/ include both values.
def _default_reference():
    r = os.environ.get("SPSHEAF_REFERENCE") or {"local": "local", "global": "tissue"}.get(
        os.environ.get("SPSHEAF_CENTERING", ""), "tissue")
    if r not in ("local", "tissue"):
        raise ValueError(f"SPSHEAF_REFERENCE must be 'local' or 'tissue', got {r!r}")
    return r


REFERENCE = _default_reference()


def local_means(fibers_norm, grid_coords, k_nbrs=50):
    """Mean fiber over each vertex's k_nbrs nearest grid neighbours (self included)."""
    nbr = knn_indices(grid_coords, k_nbrs)
    return fibers_norm[nbr].mean(axis=1)


def compute_boundary_scores(fibers_norm, grid_coords, stalks, edges,
                            restrictions, smooth_radius=60.0, verbose=True,
                            reference=None, k_nbrs=50):
    """
    Compute per-vertex boundary scores from sheaf disagreement.

    For each edge e = (u, v), the sheaf disagreement measures the
    squared difference between the fiber projections into the shared
    edge space:

        D(e) = ||F_{u◁e} ξ_u  -  F_{v◁e} ξ_v||²

    where ξ_v = U_v^T (f̃_v - f̄) are the local stalk coordinates.

    Per-vertex scores use mean aggregation (more robust to outlier
    edges than max). The score field is then spatially smoothed.

    Parameters
    ----------
    fibers_norm : ndarray, shape (N, G)
        Normalized fibers.
    grid_coords : ndarray, shape (N, 2)
        Spatial positions.
    stalks : list of ndarray
        Stalk bases.
    edges : list of (int, int)
        Spatial graph edges.
    restrictions : dict
        Restriction maps from :func:`compute_restriction_maps`.
    smooth_radius : float
        Gaussian smoothing σ for the boundary field (μm).
    verbose : bool
        Print statistics.
    reference : {"local", "tissue"} or None
        Reference point of the local coordinates; default :data:`REFERENCE`.
    k_nbrs : int
        Neighbourhood of the local reference (the stalk neighbourhood).

    Returns
    -------
    boundary_raw : ndarray, shape (N,)
        Raw (unsmoothed) per-vertex boundary scores.
    boundary_smooth : ndarray, shape (N,)
        Spatially smoothed boundary scores.
    """
    N = len(grid_coords)
    reference = reference or REFERENCE
    if reference == "local":
        fibers_c = fibers_norm - local_means(fibers_norm, grid_coords, k_nbrs)
    elif reference == "tissue":
        fibers_c = fibers_norm - fibers_norm.mean(axis=0)
    else:
        raise ValueError(f"reference must be 'local' or 'tissue', got {reference!r}")

    t0 = time.time()

    # Per-edge sheaf disagreement
    edge_disagree = np.zeros(len(edges))
    for ei, (u, v) in enumerate(edges):
        F_ue = restrictions[(u, ei)]
        F_ve = restrictions[(v, ei)]

        x_u = stalks[u].T @ fibers_c[u]  # stalk coordinates
        x_v = stalks[v].T @ fibers_c[v]

        proj_u = F_ue @ x_u  # project into shared edge space
        proj_v = F_ve @ x_v
        edge_disagree[ei] = np.sum((proj_u - proj_v) ** 2)

    # Per-vertex boundary score (mean over incident edges)
    boundary_raw = np.zeros(N)
    vertex_degree = np.zeros(N)
    for ei, (u, v) in enumerate(edges):
        boundary_raw[u] += edge_disagree[ei]
        boundary_raw[v] += edge_disagree[ei]
        vertex_degree[u] += 1
        vertex_degree[v] += 1
    boundary_raw /= np.maximum(vertex_degree, 1)

    # Spatial smoothing on regular grid
    boundary_smooth = _smooth_boundary_field(
        boundary_raw, grid_coords, smooth_radius
    )

    if verbose:
        elapsed = time.time() - t0
        print(
            f"  Boundary scores ({reference} reference): mean={boundary_smooth.mean():.4f}, "
            f"max={boundary_smooth.max():.4f} ({elapsed:.1f}s)"
        )

    return boundary_raw, boundary_smooth


def _smooth_boundary_field(boundary_raw, grid_coords, smooth_radius):
    """Smooth boundary scores spatially.

    2D grids use the fast raster + gaussian_filter path (unchanged, preserves
    validated benchmark behaviour). Higher-dimensional grids (e.g. 3D) use a
    k-d-tree Gaussian: each vertex is the Gaussian-weighted mean of the raw
    scores of neighbours within 3·smooth_radius. This is dimension-agnostic.
    """
    if grid_coords.shape[1] != 2:
        return _smooth_boundary_field_nd(boundary_raw, grid_coords,
                                         smooth_radius)
    N = len(grid_coords)
    x_min, y_min = grid_coords.min(axis=0)
    x_max, y_max = grid_coords.max(axis=0)
    spacing = np.median(np.diff(np.sort(np.unique(grid_coords[:, 0]))))

    nx = int((x_max - x_min) / spacing) + 1
    ny = int((y_max - y_min) / spacing) + 1

    gi = np.clip(
        np.round((grid_coords[:, 0] - x_min) / spacing).astype(int),
        0, nx - 1,
    )
    gj = np.clip(
        np.round((grid_coords[:, 1] - y_min) / spacing).astype(int),
        0, ny - 1,
    )

    grid_image = np.zeros((nx, ny))
    grid_mask = np.zeros((nx, ny), dtype=bool)
    for idx in range(N):
        grid_image[gi[idx], gj[idx]] = boundary_raw[idx]
        grid_mask[gi[idx], gj[idx]] = True

    sigma_px = smooth_radius / spacing
    smoothed = gaussian_filter(grid_image, sigma=sigma_px)
    smoothed[~grid_mask] = 0

    boundary_smooth = np.zeros(N)
    for idx in range(N):
        boundary_smooth[idx] = smoothed[gi[idx], gj[idx]]

    return boundary_smooth


def _smooth_boundary_field_nd(boundary_raw, grid_coords, smooth_radius,
                              block_size=4000):
    """Dimension-agnostic Gaussian smoothing via a k-d tree.

    Each vertex value becomes the Gaussian-weighted mean of the raw scores of
    all vertices within 3·smooth_radius (Euclidean, in coordinate units).
    """
    N = len(grid_coords)
    tree = cKDTree(grid_coords)
    radius = 3.0 * smooth_radius
    out = np.zeros(N)
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        nbrs = tree.query_ball_point(grid_coords[start:end], r=radius)
        for li, nb in enumerate(nbrs):
            gi = start + li
            nb = np.asarray(nb)
            d2 = ((grid_coords[nb] - grid_coords[gi]) ** 2).sum(axis=1)
            w = np.exp(-0.5 * d2 / (smooth_radius ** 2))
            out[gi] = (w * boundary_raw[nb]).sum() / max(w.sum(), 1e-12)
    return out
