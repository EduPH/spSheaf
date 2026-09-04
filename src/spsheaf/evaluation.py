"""
Evaluation metrics for spatial domain inference.

Implements the SDMBench metric suite (Yuan et al. 2024, *Nat. Methods*)
adapted to a grid-based, segmentation-free pipeline. Three families:

External (vs ground truth)
    ARI : Adjusted Rand Index           [-0.5, 1]   higher is better
    NMI : Normalized Mutual Information [ 0,  1]    higher is better
    HOM : Homogeneity                   [ 0,  1]    higher is better
    COM : Completeness                  [ 0,  1]    higher is better

Internal (spatial coherence, ground-truth-free)
    CHAOS : Spatial chaos — mean intra-cluster NN distance on
            standardized coordinates                lower is better
    PAS   : Percentage of Abnormal Spots — fraction of points whose
            majority kNN labels disagree            lower is better
    ASW   : Average Silhouette Width on spatial
            coordinates with predicted labels      higher is better
    LISI  : Local Inverse Simpson Index (optional, requires
            rpy2 + R `lisi`)                       interpret per-paper

Marker-based (expression coherence)
    markerFC      : marker fold-change vs other domains
    marker_score  : Moran's I / Geary's C on top markers
                    (optional, requires squidpy + scanpy + anndata)

The pasted SDMBench reference code had several name-resolution bugs
(class methods calling each other without ``self`` or class prefix);
the implementations below are independent rewrites with the same
mathematical definitions.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# Ground truth mapping & label matching (unchanged from prior version)
# ---------------------------------------------------------------------------

def map_ground_truth(mol_single, grid_coords):
    """
    Map ground-truth domain labels to grid points via nearest molecule.

    Parameters
    ----------
    mol_single : DataFrame
        Slice molecule table with ``x_local``, ``y_local``, ``domain``.
    grid_coords : ndarray, shape (N, 2)

    Returns
    -------
    gt_int : ndarray, shape (N,) or None
        Integer domain labels (0-indexed). -1 for missing labels.
    gt_valid : ndarray of bool, shape (N,) or None
    gt_names : list of str or None
    """
    if "domain" not in mol_single.columns:
        return None, None, None

    mol_xy = mol_single[["x_local", "y_local"]].values
    mol_dom = mol_single["domain"].values
    tree = cKDTree(mol_xy)
    _, nn_idx = tree.query(grid_coords, k=1)
    gt_labels = mol_dom[nn_idx]

    gt_valid = pd.notna(gt_labels)
    gt_names = sorted(set(gt_labels[gt_valid]))
    dom_map = {d: i for i, d in enumerate(gt_names)}
    gt_int = np.array([
        dom_map.get(g, -1) if pd.notna(g) else -1 for g in gt_labels
    ])
    return gt_int, gt_valid, gt_names


def hungarian_match_labels(labels_pred, labels_true, valid_mask):
    """Remap predicted labels to best-match ground truth via Hungarian
    algorithm. Used for visual comparison."""
    pred = labels_pred[valid_mask]
    true = labels_true[valid_mask]
    n_pred = len(set(pred))
    n_true = len(set(true))
    n = max(n_pred, n_true)

    cost = np.zeros((n, n))
    for i in range(n_pred):
        for j in range(n_true):
            cost[i, j] = -np.sum((pred == i) & (true == j))

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {r: c for r, c in zip(row_ind, col_ind)}

    remapped = np.full_like(labels_pred, -1)
    for i in range(len(labels_pred)):
        remapped[i] = mapping.get(labels_pred[i], labels_pred[i])
    return remapped


# ---------------------------------------------------------------------------
# External metrics (require ground truth)
# ---------------------------------------------------------------------------

def compute_ari(labels, gt_int, mask=None):
    """Adjusted Rand Index, optionally restricted to ``mask``."""
    if mask is not None:
        return adjusted_rand_score(gt_int[mask], labels[mask])
    return adjusted_rand_score(gt_int, labels)


def compute_nmi(labels, gt_int, mask=None):
    """Normalized Mutual Information."""
    if mask is not None:
        return normalized_mutual_info_score(gt_int[mask], labels[mask])
    return normalized_mutual_info_score(gt_int, labels)


def compute_homogeneity(labels, gt_int, mask=None):
    """Homogeneity: each predicted cluster contains only one true class."""
    if mask is not None:
        return homogeneity_score(gt_int[mask], labels[mask])
    return homogeneity_score(gt_int, labels)


def compute_completeness(labels, gt_int, mask=None):
    """Completeness: all members of a true class fall in one cluster."""
    if mask is not None:
        return completeness_score(gt_int[mask], labels[mask])
    return completeness_score(gt_int, labels)


# ---------------------------------------------------------------------------
# Internal spatial metrics (no ground truth needed)
# ---------------------------------------------------------------------------

def _nn_distance(coords):
    """Vector of nearest-neighbor distances for every point in ``coords``.

    Uses a KD-tree (k=2: self + nearest). Returns ``np.inf`` for clusters
    of size 1.
    """
    if len(coords) <= 1:
        return np.full(len(coords), np.inf)
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=2)
    return d[:, 1]


def compute_chaos(labels, coords, min_cluster_size=3):
    """
    Spatial Chaos (CHAOS) — SDMBench definition.

    For each cluster with at least ``min_cluster_size`` points, sum the
    intra-cluster nearest-neighbor distances on standardized coordinates,
    then divide by the total number of points. Lower is better; large
    values indicate spatially scattered clusters.

    Parameters
    ----------
    labels : ndarray, shape (N,)
    coords : ndarray, shape (N, 2)
    min_cluster_size : int
        Clusters with fewer points than this contribute zero.

    Returns
    -------
    chaos : float
    """
    labels = np.asarray(labels)
    coords = np.asarray(coords)
    matched = StandardScaler().fit_transform(coords)

    total = 0.0
    for k in np.unique(labels):
        idx = labels == k
        if idx.sum() < min_cluster_size:
            continue
        nn = _nn_distance(matched[idx])
        total += float(np.sum(nn))
    return total / len(labels)


def compute_pas(labels, coords, k=10):
    """
    Percentage of Abnormal Spots (PAS).

    For each point, look up its ``k`` nearest spatial neighbors. The point
    is "abnormal" if a majority (> k/2) of these neighbors have a label
    different from its own. Lower is better.

    Parameters
    ----------
    labels : ndarray, shape (N,)
    coords : ndarray, shape (N, 2)
    k : int
        Neighborhood size (default 10, matching SDMBench).

    Returns
    -------
    pas : float
        Fraction of abnormal points in [0, 1].
    """
    labels = np.asarray(labels)
    coords = np.asarray(coords)
    tree = cKDTree(coords)
    # k+1 because the first neighbor is the point itself
    _, nn_idx = tree.query(coords, k=k + 1)
    nn_idx = nn_idx[:, 1:]                         # drop self
    nn_labels = labels[nn_idx]                     # (N, k)
    disagree = (nn_labels != labels[:, None]).sum(axis=1)
    return float(np.mean(disagree > (k / 2)))


def compute_asw(labels, coords, sample_size=None, random_state=0):
    """
    Average Silhouette Width on spatial coordinates.

    Higher is better; positive values indicate spatially compact
    clusters. The SDMBench reference precomputes the full pairwise
    distance matrix; we let sklearn compute distances on the fly to
    avoid the O(N²) memory cost (for N≈14k grid points the full matrix
    is ~1.5 GB; for ~50k it is infeasible).

    Parameters
    ----------
    labels : ndarray, shape (N,)
    coords : ndarray, shape (N, 2)
    sample_size : int or None
        If set, silhouette is computed on a random subsample. Useful
        for very large grids; ``None`` uses all points.
    random_state : int
        Seed for the subsample.

    Returns
    -------
    asw : float
    """
    labels = np.asarray(labels)
    coords = np.asarray(coords)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(
        coords, labels,
        metric="euclidean",
        sample_size=sample_size,
        random_state=random_state,
    ))


def compute_lisi(labels, coords, perplexity=30, nn_eps=0):
    """
    Local Inverse Simpson Index (LISI) — optional, requires rpy2 and the
    R package `lisi` (Korsunsky et al. 2019).

    Returns
    -------
    lisi : ndarray
        Per-point LISI values. Higher = more locally mixed (i.e. fewer,
        less separated domains around each point).
    """
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
    except ImportError as e:
        raise ImportError(
            "compute_lisi requires rpy2 and the R `lisi` package. "
            "Install with: pip install rpy2 && R -e 'devtools::"
            "install_github(\"immunogenomics/lisi\")'"
        ) from e

    pandas2ri.activate()
    importr("lisi")

    coords_df = pd.DataFrame(coords, columns=["x", "y"])
    meta = pd.DataFrame({"label": np.asarray(labels).astype(str)})

    r_coords = robjects.conversion.py2rpy(coords_df)
    r_meta = robjects.conversion.py2rpy(meta)
    as_matrix = robjects.r["as.matrix"]
    out = robjects.r["compute_lisi"](
        as_matrix(r_coords), r_meta, "label", perplexity, nn_eps,
    )
    if isinstance(out, pd.DataFrame):
        return out.values.ravel()
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Marker-based metrics
# ---------------------------------------------------------------------------

def compute_marker_fc(features, labels, gene_names=None,
                      marker_list=None, eps=1e-12):
    """
    Marker fold-change (markerFC).

    For each marker gene, find the cluster with the highest mean
    expression of that gene, then compute the ratio of that mean to the
    mean across all other clusters. Higher = more cluster-specific.

    Works on the grid-level expression matrix ``features`` (e.g.
    ``fibers_norm`` from the pipeline) without needing AnnData.

    Parameters
    ----------
    features : ndarray, shape (N, G)
        Expression matrix (points × genes).
    labels : ndarray, shape (N,)
        Cluster labels.
    gene_names : list of str or None
        Required if ``marker_list`` is a list of gene names; otherwise
        ``marker_list`` is interpreted as integer indices.
    marker_list : list of str or list of int or None
        Markers to evaluate. If None, every column of ``features`` is
        used.
    eps : float
        Numerical floor for the denominator.

    Returns
    -------
    fc : dict
        ``{marker: fold_change}``.
    """
    features = np.asarray(features)
    labels = np.asarray(labels)
    n, g = features.shape

    if marker_list is None:
        idx_list = list(range(g))
        names = gene_names if gene_names is not None else list(range(g))
    elif all(isinstance(m, (int, np.integer)) for m in marker_list):
        idx_list = list(marker_list)
        names = (
            [gene_names[i] for i in idx_list]
            if gene_names is not None else idx_list
        )
    else:
        if gene_names is None:
            raise ValueError(
                "gene_names is required when marker_list contains names."
            )
        name_to_idx = {n_: i for i, n_ in enumerate(gene_names)}
        idx_list = [name_to_idx[m] for m in marker_list]
        names = list(marker_list)

    unique = np.unique(labels)
    out = {}
    for name, gi in zip(names, idx_list):
        means = np.array([features[labels == c, gi].mean() for c in unique])
        top_c = unique[np.argmax(means)]
        top_val = float(means.max())
        other = float(features[labels != top_c, gi].mean())
        out[name] = top_val / (other + eps)
    return out


def compute_marker_score(adata, domain_key, top_n=5, n_perms=100, n_jobs=1):
    """
    Moran's I and Geary's C on top markers — optional, requires
    ``scanpy``, ``squidpy``, and an AnnData object.

    Returns
    -------
    moran_I : float
        Median Moran's I across selected genes (higher = more spatially
        structured).
    geary_C : float
        Median Geary's C (lower = more spatially structured).
    """
    try:
        import scanpy as sc
        import squidpy as sq
    except ImportError as e:
        raise ImportError(
            "compute_marker_score requires scanpy and squidpy "
            "(install with `pip install scanpy squidpy`)."
        ) from e

    adata = adata.copy()
    counts = adata.obs[domain_key].value_counts()
    keep = counts.index[counts > 3]
    adata = adata[adata.obs[domain_key].isin(keep)].copy()

    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(adata, groupby=domain_key)

    selected = []
    for i in range(top_n):
        selected.extend(list(adata.uns["rank_genes_groups"]["names"][i]))
    selected = np.unique(selected)

    sq.gr.spatial_neighbors(adata)
    sq.gr.spatial_autocorr(
        adata, mode="moran", genes=selected,
        n_perms=n_perms, n_jobs=n_jobs,
    )
    sq.gr.spatial_autocorr(
        adata, mode="geary", genes=selected,
        n_perms=n_perms, n_jobs=n_jobs,
    )
    return (
        float(np.median(adata.uns["moranI"]["I"])),
        float(np.median(adata.uns["gearyC"]["C"])),
    )


def compute_marker_moran_geary(expr, coords, labels, k=6, top_n=3):
    """Marker-gene spatial autocorrelation (SDMBench 'marker score'),
    squidpy-free.

    For each predicted domain, take its top-``top_n`` marker genes (highest
    mean-vs-global enrichment on the RAW expression), then compute Moran's I
    and Geary's C for each selected gene using a binary symmetric k-NN spatial
    weight matrix. Returns the medians across selected genes.

    IMPORTANT: pass *raw* per-cell expression (not smoothed fibers) so the
    statistic is comparable to how the baseline methods are scored — smoothed
    fibers are spatially autocorrelated by construction and would inflate I.

    Parameters
    ----------
    expr : ndarray (n_cells, n_genes)   raw (log-)expression per cell
    coords : ndarray (n_cells, 2)       cell coordinates
    labels : ndarray (n_cells,)         predicted domain per cell
    k : int                             spatial neighbours for the weights
    top_n : int                         markers per domain

    Returns
    -------
    (moran_I_median, geary_C_median) : floats (nan if not computable)
    """
    from sklearn.neighbors import NearestNeighbors
    expr = np.asarray(expr, float)
    labels = np.asarray(labels)
    N = len(coords)
    if N <= k + 1 or expr.shape[1] == 0:
        return float("nan"), float("nan")

    gmean = expr.mean(0)
    sel = set()
    for c in np.unique(labels):
        m = labels == c
        if m.sum() == 0:
            continue
        enr = expr[m].mean(0) - gmean
        sel.update(np.argsort(enr)[::-1][:top_n].tolist())
    sel = sorted(sel)
    if not sel:
        return float("nan"), float("nan")

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    rows = np.repeat(np.arange(N), k)
    cols = idx[:, 1:].ravel()
    W = len(rows)  # sum of binary weights

    mIs, gCs = [], []
    for g in sel:
        x = expr[:, g]
        z = x - x.mean()
        denom = (z ** 2).sum()
        if denom <= 0:
            continue
        moran = (N / W) * ((z[rows] * z[cols]).sum() / denom)
        geary = ((N - 1) / (2 * W)) * (((x[rows] - x[cols]) ** 2).sum() / denom)
        mIs.append(moran)
        gCs.append(geary)
    if not mIs:
        return float("nan"), float("nan")
    return float(np.median(mIs)), float(np.median(gCs))


# ---------------------------------------------------------------------------
# Unified evaluation entry points
# ---------------------------------------------------------------------------

def evaluate_clustering(labels, gt_int, gt_valid, name="Method",
                        verbose=True):
    """
    Backward-compatible ARI/NMI evaluation.

    Kept identical to the previous API so existing callers in
    ``pipeline.py`` continue to work unchanged. For the full SDMBench
    suite, use :func:`evaluate_clustering_full`.

    Returns
    -------
    ari, nmi : float, float
    """
    mask = gt_valid
    ari = adjusted_rand_score(gt_int[mask], labels[mask])
    nmi = normalized_mutual_info_score(gt_int[mask], labels[mask])
    if verbose:
        print(f"  {name:<25s}: ARI={ari:.3f}, NMI={nmi:.3f}")
    return ari, nmi


def evaluate_clustering_full(labels, coords,
                             gt_int=None, gt_valid=None,
                             pas_k=10, chaos_min_size=3,
                             asw_sample_size=None,
                             compute_lisi_flag=False, lisi_perplexity=30,
                             name="Method", verbose=True):
    """
    Full SDMBench metric suite.

    Always computes the internal spatial metrics (CHAOS, PAS, ASW) and,
    if ground truth is provided, the external metrics (ARI, NMI, HOM,
    COM). LISI is opt-in because it requires rpy2 and R.

    Parameters
    ----------
    labels : ndarray, shape (N,)
        Predicted cluster labels.
    coords : ndarray, shape (N, 2)
        Spatial positions.
    gt_int : ndarray or None
        Ground truth integer labels. If None, external metrics are
        skipped.
    gt_valid : ndarray of bool or None
        Mask of points with valid ground truth. If None and ``gt_int``
        is provided, all points are treated as valid.
    pas_k : int
        Neighborhood size for PAS (default 10).
    chaos_min_size : int
        Minimum cluster size for CHAOS contribution (default 3).
    asw_sample_size : int or None
        Subsample size for ASW; None uses all points.
    compute_lisi_flag : bool
        If True, also compute median LISI (requires rpy2 + R `lisi`).
    lisi_perplexity : int
        LISI perplexity parameter.
    name : str
        Method name for the printed summary.
    verbose : bool

    Returns
    -------
    metrics : dict
        Available keys: ``ari``, ``nmi``, ``hom``, ``com``, ``chaos``,
        ``pas``, ``asw``, ``lisi_median``. Missing entries are absent
        from the dict.
    """
    labels = np.asarray(labels)
    coords = np.asarray(coords)
    metrics = {}

    # External (vs ground truth)
    if gt_int is not None:
        mask = gt_valid if gt_valid is not None else np.ones(len(labels), bool)
        metrics["ari"] = compute_ari(labels, gt_int, mask)
        metrics["nmi"] = compute_nmi(labels, gt_int, mask)
        metrics["hom"] = compute_homogeneity(labels, gt_int, mask)
        metrics["com"] = compute_completeness(labels, gt_int, mask)

    # Internal (no ground truth)
    metrics["chaos"] = compute_chaos(
        labels, coords, min_cluster_size=chaos_min_size,
    )
    metrics["pas"] = compute_pas(labels, coords, k=pas_k)
    metrics["asw"] = compute_asw(
        labels, coords, sample_size=asw_sample_size,
    )

    if compute_lisi_flag:
        try:
            lisi_vals = compute_lisi(
                labels, coords, perplexity=lisi_perplexity,
            )
            metrics["lisi_median"] = float(np.median(lisi_vals))
        except ImportError as e:
            if verbose:
                print(f"  [LISI skipped: {e}]")

    if verbose:
        _print_metrics(metrics, name)

    return metrics


def _print_metrics(metrics, name):
    """Pretty-print a metrics dict in two lines (external + spatial)."""
    print(f"  {name}")
    ext_keys = ["ari", "nmi", "hom", "com"]
    spa_keys = ["chaos", "pas", "asw", "lisi_median"]

    ext = "  ".join(
        f"{k.upper()}={metrics[k]:.3f}" for k in ext_keys if k in metrics
    )
    spa = "  ".join(
        f"{k.upper()}={metrics[k]:.3f}" for k in spa_keys if k in metrics
    )
    if ext:
        print(f"    External : {ext}")
    if spa:
        print(f"    Spatial  : {spa}")