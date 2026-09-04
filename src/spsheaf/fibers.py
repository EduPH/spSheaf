"""
Gaussian kernel-smoothed expression fibers on a spatial grid.

At each grid point v, the fiber f_v ∈ ℝ^G is a weighted count of
nearby molecules, with weights given by a Gaussian kernel:

    f_v^(g) = Σ_{i: gene_i = g} exp(-||v - x_i||² / (2h²))

where h is the bandwidth parameter. The search radius is truncated
at 3h for computational efficiency.

The fibers are then normalized to proportions and log-transformed:

    f̃_v = log(1 + 1000 · f_v / ||f_v||₁)

This removes density variation while preserving gene composition,
analogous to counts-per-million normalization in scRNA-seq.
"""

import numpy as np
from scipy.spatial import cKDTree


def build_kernel_fibers(mol, grid_spacing=15.0, bandwidth=25.0,
                        min_molecules=15, x_col="x_local", y_col="y_local",
                        z_col=None, z_spacing=None, z_bandwidth=None,
                        block_size=None, verbose=True):
    """
    Construct Gaussian kernel-smoothed expression fibers on a regular grid.

    Works in 2D (default) or 3D. Passing ``z_col`` switches on the N-D path:
    the grid, k-d tree, and Gaussian kernel all become 3D, with an
    (optionally) anisotropic bandwidth so a thin z axis (e.g. MERSCOPE focal
    planes) can be resolved differently from x/y. The 2D path is unchanged.

    Parameters
    ----------
    mol : DataFrame
        Molecule table with columns ``gene``, ``x_col``, ``y_col`` (and
        ``z_col`` for 3D).
    grid_spacing : float
        Distance between grid points in x/y (μm). Default 15 μm.
    bandwidth : float
        Gaussian kernel bandwidth h in x/y (μm). Default 25 μm.
        Domain-level structure emerges at coarse bandwidths (~150 μm);
        finer scales (~25 μm) resolve individual cells.
    min_molecules : float
        Minimum total weighted count to retain a grid point.
    x_col, y_col : str
        Column names for spatial coordinates.
    z_col : str or None
        If given, build a 3D fiber field over (x, y, z).
    z_spacing : float or None
        Grid spacing along z (μm). Defaults to ``grid_spacing``. For a thin
        z axis set this small (e.g. the physical focal-plane spacing) so the
        volume is not collapsed to a single z level.
    z_bandwidth : float or None
        Gaussian bandwidth along z (μm). Defaults to ``bandwidth``.
    block_size : int or None
        Grid points processed per block in the 3D path (memory control).
        Defaults to 2000. Ignored in 2D.
    verbose : bool
        Print grid statistics.

    Returns
    -------
    grid_coords : ndarray, shape (N, D)
        Spatial positions of valid grid points (D = 2 or 3).
    fibers : ndarray, shape (N, G)
        Raw (unnormalized) expression vectors.
    gene_names : list of str
        Ordered gene names corresponding to fiber columns.
    """
    if z_col is not None:
        return _build_kernel_fibers_nd(
            mol, grid_spacing=grid_spacing, bandwidth=bandwidth,
            min_molecules=min_molecules, x_col=x_col, y_col=y_col,
            z_col=z_col, z_spacing=z_spacing, z_bandwidth=z_bandwidth,
            block_size=block_size or 2000, verbose=verbose,
        )

    gene_names = sorted(mol["gene"].unique())
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    mol_coords = mol[[x_col, y_col]].values.astype(np.float64)
    mol_gene_idx = mol["gene"].map(gene_to_idx).values

    # Regular grid covering the tissue extent
    x_min, x_max = mol_coords[:, 0].min(), mol_coords[:, 0].max()
    y_min, y_max = mol_coords[:, 1].min(), mol_coords[:, 1].max()
    x_grid = np.arange(x_min + grid_spacing / 2, x_max, grid_spacing)
    y_grid = np.arange(y_min + grid_spacing / 2, y_max, grid_spacing)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Gaussian kernel smoothing with 3σ truncation
    tree = cKDTree(mol_coords)
    radius = 3.0 * bandwidth
    neighbors_list = tree.query_ball_point(grid_coords, r=radius)

    fibers = np.zeros((len(grid_coords), n_genes), dtype=np.float32)
    total_weight = np.zeros(len(grid_coords), dtype=np.float32)
    keep_mask = np.zeros(len(grid_coords), dtype=bool)

    for qi in range(len(grid_coords)):
        nb = neighbors_list[qi]
        if len(nb) == 0:
            continue
        nb = np.array(nb)
        dists = np.linalg.norm(mol_coords[nb] - grid_coords[qi], axis=1)
        weights = np.exp(-0.5 * (dists / bandwidth) ** 2)
        np.add.at(fibers[qi], mol_gene_idx[nb], weights)
        total_weight[qi] = weights.sum()
        if total_weight[qi] >= min_molecules:
            keep_mask[qi] = True

    grid_coords = grid_coords[keep_mask]
    fibers = fibers[keep_mask]

    if verbose:
        print(
            f"  Grid: {len(grid_coords):,} valid points, {n_genes} genes"
        )

    return grid_coords, fibers, gene_names


def _build_kernel_fibers_nd(mol, grid_spacing, bandwidth, min_molecules,
                            x_col, y_col, z_col, z_spacing, z_bandwidth,
                            block_size, verbose=True):
    """N-D (currently 3D) kernel-fiber builder.

    Anisotropy is handled by working in a *scaled* coordinate space where each
    axis is divided by its own bandwidth, so a single isotropic Gaussian with
    3-sigma truncation (radius = 3 in scaled units) applies. This keeps the
    kernel math identical in form to the 2D path while allowing z to have its
    own spacing and bandwidth. Grid points are processed in blocks so the
    radius-query neighbour lists never materialise for the whole grid at once.
    """
    z_spacing = grid_spacing if z_spacing is None else z_spacing
    z_bandwidth = bandwidth if z_bandwidth is None else z_bandwidth

    gene_names = sorted(mol["gene"].unique())
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes = len(gene_names)

    coords = mol[[x_col, y_col, z_col]].values.astype(np.float64)
    gene_idx = mol["gene"].map(gene_to_idx).values

    spacing = np.array([grid_spacing, grid_spacing, z_spacing], dtype=float)
    bw = np.array([bandwidth, bandwidth, z_bandwidth], dtype=float)

    # Regular 3D grid over the tissue extent
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    axes = [np.arange(mins[d] + spacing[d] / 2, maxs[d], spacing[d])
            for d in range(3)]
    # guard against a degenerate (single-value) axis
    axes = [ax if len(ax) > 0 else np.array([(mins[d] + maxs[d]) / 2])
            for d, ax in enumerate(axes)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid_coords = np.column_stack([m.ravel() for m in mesh])

    # Scaled space: divide each axis by its bandwidth -> isotropic Gaussian
    coords_s = coords / bw
    grid_s = grid_coords / bw
    tree = cKDTree(coords_s)
    radius = 3.0  # 3 sigma in scaled units

    n_grid = len(grid_coords)
    fibers = np.zeros((n_grid, n_genes), dtype=np.float32)
    total_weight = np.zeros(n_grid, dtype=np.float32)

    for start in range(0, n_grid, block_size):
        end = min(start + block_size, n_grid)
        nbr_lists = tree.query_ball_point(grid_s[start:end], r=radius)
        for li, nb in enumerate(nbr_lists):
            if not nb:
                continue
            gi = start + li
            nb = np.asarray(nb)
            d2 = ((coords_s[nb] - grid_s[gi]) ** 2).sum(axis=1)
            w = np.exp(-0.5 * d2)
            np.add.at(fibers[gi], gene_idx[nb], w)
            total_weight[gi] = w.sum()
        del nbr_lists

    keep = total_weight >= min_molecules
    grid_coords = grid_coords[keep]
    fibers = fibers[keep]

    if verbose:
        nz = len(np.unique(grid_coords[:, 2]))
        print(f"  Grid (3D): {len(grid_coords):,} valid points "
              f"({nz} z levels), {n_genes} genes")

    return grid_coords, fibers, gene_names


def build_feature_fibers(coords, features, grid_spacing, bandwidth,
                         z_spacing=None, z_bandwidth=None, min_weight=1e-3,
                         block_size=2000, standardize=True, verbose=True):
    """Kernel-smoothed fiber field from *continuous per-point features*.

    The counterpart of :func:`build_kernel_fibers` for the case where each
    point already carries a feature vector (e.g. per-locus chromosome
    intensities, or per-voxel chromatin-mark IF intensities) instead of a
    single categorical identity. At each grid point the fiber is the
    Gaussian-kernel-weighted **mean** of the feature vectors of nearby points.

    Works in 2D or 3D (D = coords.shape[1]); anisotropy is handled by a
    per-axis bandwidth (``z_bandwidth``) via a bandwidth-scaled isotropic
    kernel, exactly as in the N-D count builder.

    Parameters
    ----------
    coords : ndarray (M, D)       point coordinates (physical units)
    features : ndarray (M, F)     continuous feature per point
    grid_spacing, bandwidth : float   x/y grid + kernel scale
    z_spacing, z_bandwidth : float or None   z grid + kernel scale (3D)
    min_weight : float            drop grid points with total kernel weight below this
    standardize : bool            z-score each fiber column (recommended;
                                  intensities are not compositional so the
                                  proportion+log normalize_fibers is not used)

    Returns
    -------
    grid_coords : ndarray (N, D)
    fibers : ndarray (N, F)       (standardized) feature fibers
    """
    coords = np.asarray(coords, float)
    features = np.asarray(features, float)
    D = coords.shape[1]
    if D == 2:
        bw = np.array([bandwidth, bandwidth])
        sp = np.array([grid_spacing, grid_spacing])
    else:
        bw = np.array([bandwidth, bandwidth,
                       bandwidth if z_bandwidth is None else z_bandwidth])
        sp = np.array([grid_spacing, grid_spacing,
                       grid_spacing if z_spacing is None else z_spacing])

    mins, maxs = coords.min(0), coords.max(0)
    axes = [np.arange(mins[d] + sp[d] / 2, maxs[d], sp[d]) for d in range(D)]
    axes = [ax if len(ax) else np.array([(mins[d] + maxs[d]) / 2])
            for d, ax in enumerate(axes)]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack([m.ravel() for m in mesh])

    coords_s = coords / bw
    grid_s = grid / bw
    tree = cKDTree(coords_s)
    F = features.shape[1]
    fib = np.zeros((len(grid), F), dtype=np.float64)
    tw = np.zeros(len(grid), dtype=np.float64)
    for start in range(0, len(grid), block_size):
        end = min(start + block_size, len(grid))
        nbrs = tree.query_ball_point(grid_s[start:end], r=3.0)
        for li, nb in enumerate(nbrs):
            if not nb:
                continue
            gi = start + li
            nb = np.asarray(nb)
            w = np.exp(-0.5 * ((coords_s[nb] - grid_s[gi]) ** 2).sum(1))
            sw = w.sum()
            if sw <= 0:
                continue
            fib[gi] = (w[:, None] * features[nb]).sum(0) / sw
            tw[gi] = sw
        del nbrs

    keep = tw >= min_weight
    grid, fib = grid[keep], fib[keep]
    if standardize:
        mu = fib.mean(0)
        sd = fib.std(0)
        sd[sd < 1e-9] = 1.0
        fib = ((fib - mu) / sd).astype(np.float32)
    if verbose:
        nz = len(np.unique(grid[:, 2])) if D == 3 else 1
        print(f"  Feature fibers: {len(grid):,} grid pts"
              + (f" ({nz} z levels)" if D == 3 else "")
              + f", {F} features")
    return grid, fib.astype(np.float32)


def normalize_fibers(fibers):
    """
    Normalize fibers: proportions → log(1 + 1000·p).

    Converts raw weighted counts to proportions (removing density
    variation) and applies log-transformation for variance stabilization.

    Parameters
    ----------
    fibers : ndarray, shape (N, G)
        Raw expression vectors.

    Returns
    -------
    fibers_norm : ndarray, shape (N, G)
        Normalized expression vectors.
    """
    row_sums = np.maximum(fibers.sum(axis=1, keepdims=True), 1e-8)
    proportions = fibers / row_sums
    fibers_norm = np.log1p(proportions * 1000.0).astype(np.float32)
    return fibers_norm
