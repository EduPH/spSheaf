"""
High-level pipeline orchestration.

Provides entry points for:
- Single-slice sheaf boundary detection and domain inference
- Multi-slice benchmark across all slices

Evaluation reports the full SDMBench metric suite (Yuan et al. 2024)
through :mod:`evaluation`. Two levels are computed whenever the
molecule table carries per-cell metadata (``cell_id``, ``cell_x``,
``cell_y`` — as in Imoto's MERFISH export):

  - **Grid level**  : metrics on the dense grid (~14k points).
                       Diagnostic only; not directly comparable to
                       cell-level leaderboards.
  - **Cell level**  : grid predictions mapped to cell centroids via
                       nearest neighbour. **SDMBench-comparable** —
                       this is the headline metric and what
                       ``result["ari"]``/``result["nmi"]`` report when
                       cells are available.

If the molecule table has no cell metadata, cell-level eval is skipped
and grid-level becomes the headline (legacy behaviour, used e.g. for
Stereo-seq).
"""

import numpy as np
import pandas as pd
import time

from scipy.spatial import cKDTree

from . import io, fibers, graph, sheaf, clustering, evaluation


# Metric columns surfaced in the per-slice summary table (order matters).
_EXTERNAL_METRICS = ["ari", "nmi", "hom", "com"]
_INTERNAL_METRICS = ["chaos", "pas", "asw"]
_ALL_METRICS = _EXTERNAL_METRICS + _INTERNAL_METRICS


# ---------------------------------------------------------------------------
# Cell extraction (Imoto MERFISH schema)
# ---------------------------------------------------------------------------

def _extract_cells(mol_single,
                   cell_id_col="cell_id",
                   cell_x_col="cell_x",
                   cell_y_col="cell_y",
                   domain_col="domain"):
    """
    Per-cell centroids + ground truth from the molecule table.

    Assumes ``mol_single`` has been processed by :func:`io.select_slice`
    (i.e. has ``x_local``, ``y_local`` columns). Cell coordinates are
    shifted to the same local frame.

    Returns
    -------
    cell_coords : ndarray, shape (n_cells, 2) or None
    cell_gt_str : ndarray of object, shape (n_cells,) or None
        Domain labels (may contain NaN). ``None`` if no ``domain``
        column.
    cell_ids : ndarray, shape (n_cells,) or None

    Returns ``(None, None, None)`` if the cell columns are missing or
    no molecules carry valid cell metadata.
    """
    needed = [cell_id_col, cell_x_col, cell_y_col]
    if not all(c in mol_single.columns for c in needed):
        return None, None, None

    valid = mol_single.dropna(subset=needed)
    if len(valid) == 0:
        return None, None, None

    agg = {cell_x_col: "first", cell_y_col: "first"}
    if domain_col in valid.columns:
        agg[domain_col] = "first"
    cells = valid.groupby(cell_id_col, sort=False).agg(agg)

    # Local-frame shift (mol_single carries x_local = x - x.min())
    dx = mol_single["x_local"].iloc[0] - mol_single["x"].iloc[0]
    dy = mol_single["y_local"].iloc[0] - mol_single["y"].iloc[0]
    cell_coords = (
        cells[[cell_x_col, cell_y_col]].to_numpy()
        + np.array([dx, dy])
    )
    cell_gt_str = (
        cells[domain_col].to_numpy() if domain_col in cells.columns else None
    )
    cell_ids = cells.index.to_numpy()
    return cell_coords, cell_gt_str, cell_ids


def _marker_autocorr(mol_single, cell_ids, cell_coords, cell_predicted, verbose):
    """Moran's I / Geary's C on domain marker genes, over RAW per-cell counts.

    Primary path uses squidpy (``evaluation.compute_marker_score``) so the
    statistic matches SDMBench exactly (Wilcoxon-ranked markers + squidpy
    spatial graph). If scanpy/squidpy/anndata are not installed, falls back to
    the dependency-free ``compute_marker_moran_geary`` (top mean-enrichment
    markers + binary 6-NN weights) and says so.
    """
    ct = (mol_single.dropna(subset=["cell_id"])
          .groupby(["cell_id", "gene"]).size().unstack(fill_value=0)
          .reindex(cell_ids).fillna(0))
    counts = ct.to_numpy(dtype=float)
    try:
        import anndata as ad
        adata = ad.AnnData(
            X=counts,
            obs=pd.DataFrame(
                {"domain": pd.Categorical(np.asarray(cell_predicted).astype(str))},
                index=ct.index.astype(str)),
            var=pd.DataFrame(index=ct.columns.astype(str)))
        adata.obsm["spatial"] = np.asarray(cell_coords, dtype=float)
        mI, gC = evaluation.compute_marker_score(adata, "domain")
        if verbose:
            print(f"  Marker autocorr (squidpy): Moran's I={mI:.3f}, "
                  f"Geary's C={gC:.3f}")
        return mI, gC
    except ImportError:
        mI, gC = evaluation.compute_marker_moran_geary(
            np.log1p(counts), cell_coords, cell_predicted)
        if verbose:
            print(f"  Marker autocorr (manual fallback — `pip install squidpy "
                  f"scanpy` for SDMBench-exact): Moran's I={mI:.3f}, "
                  f"Geary's C={gC:.3f}")
        return mI, gC


def _cell_level_evaluation(grid_coords, grid_labels, mol_single,
                           pas_k, chaos_min_size, asw_sample_size,
                           compute_lisi, lisi_perplexity, verbose,
                           compute_autocorr=False):
    """
    Map grid predictions to cell centroids and compute the full
    SDMBench metric suite at cell level.

    Returns a dict with keys: ``metrics`` (possibly empty if cells not
    available), ``cell_coords``, ``cell_predicted``, ``cell_gt_int``,
    ``cell_gt_names``. All values are ``None``/empty if cell metadata
    is missing.
    """
    cell_coords, cell_gt_str, cell_ids = _extract_cells(mol_single)
    if cell_coords is None:
        if verbose:
            print("  (cell metadata not available — skipping cell-level eval)")
        return dict(
            metrics={}, cell_coords=None, cell_predicted=None,
            cell_gt_int=None, cell_gt_names=[],
        )

    # Each cell → nearest grid point → predicted label
    tree = cKDTree(grid_coords)
    dists, idx = tree.query(cell_coords, k=1)
    cell_predicted = grid_labels[idx]

    if verbose:
        print(f"\n  CELL-LEVEL EVALUATION ({len(cell_coords):,} cells)")
        print("  " + "-" * 40)
        print(
            f"  Nearest grid distance: "
            f"median={np.median(dists):.1f} μm, "
            f"max={dists.max():.1f} μm"
        )

    # Encode string GT as integers; mask missing entries
    cell_gt_int = None
    gt_valid_cell = None
    cell_gt_names = []
    if cell_gt_str is not None:
        gt_valid_cell = pd.notna(cell_gt_str)
        cell_gt_names = sorted(set(cell_gt_str[gt_valid_cell]))
        name_to_idx = {n: i for i, n in enumerate(cell_gt_names)}
        cell_gt_int = np.array([
            name_to_idx.get(g, -1) if pd.notna(g) else -1
            for g in cell_gt_str
        ])

    metrics = evaluation.evaluate_clustering_full(
        cell_predicted, cell_coords,
        gt_int=cell_gt_int, gt_valid=gt_valid_cell,
        pas_k=pas_k, chaos_min_size=chaos_min_size,
        asw_sample_size=asw_sample_size,
        compute_lisi_flag=compute_lisi,
        lisi_perplexity=lisi_perplexity,
        name="Sheaf boundary (cell-level)", verbose=verbose,
    )

    # Marker-gene spatial autocorrelation (Moran's I / Geary's C) on RAW
    # per-cell expression + our labels (SDMBench-comparable marker score).
    if compute_autocorr:
        try:
            mI, gC = _marker_autocorr(mol_single, cell_ids, cell_coords,
                                      cell_predicted, verbose)
            metrics["morani"] = mI
            metrics["gearyc"] = gC
        except Exception as e:
            if verbose:
                print(f"  [marker autocorr skipped: {e}]")

    return dict(
        metrics=metrics,
        cell_coords=cell_coords,
        cell_predicted=cell_predicted,
        cell_gt_int=cell_gt_int,
        cell_gt_names=cell_gt_names,
    )


# ---------------------------------------------------------------------------
# Single-slice driver
# ---------------------------------------------------------------------------

def run_single_slice(mol, slice_name, n_clusters=8,
                     grid_spacing=15, bandwidth=25,
                     k_graph=6, k_nbrs=50, uniform_dim=6,
                     smooth_radius=60.0, spatial_weight=0.5,
                     boundary_alpha=0.5,
                     overcluster_factor=1.0, grid_offset=None,
                     cleanup=True, min_fragment_frac=0.01,
                     eval_at_cell_level=True,
                     pas_k=10, chaos_min_size=3,
                     asw_sample_size=None,
                     compute_lisi=False, lisi_perplexity=30,
                     random_state=42, compute_autocorr=False,
                     verbose=True):
    """
    Run the full sheaf boundary detection pipeline on a single slice.

    Parameters
    ----------
    mol : DataFrame
        Full molecule table (all slices).
    slice_name : str
        Slice identifier (``matched_h5ad_file`` value).
    n_clusters, grid_spacing, bandwidth, k_graph, k_nbrs, uniform_dim,
    smooth_radius, spatial_weight, boundary_alpha, overcluster_factor,
    cleanup, min_fragment_frac
        Pipeline hyperparameters; see Methodology report.
    eval_at_cell_level : bool
        If True (default), also map grid predictions to cell centroids
        via the ``cell_id``/``cell_x``/``cell_y`` columns and compute
        the SDMBench-comparable cell-level suite. Falls back silently
        to grid-only when cell metadata is unavailable.
    pas_k, chaos_min_size, asw_sample_size, compute_lisi, lisi_perplexity
        Evaluation parameters; forwarded to
        :func:`evaluation.evaluate_clustering_full`.
    verbose : bool

    Returns
    -------
    result : dict
        Keys::

          slice_name, labels, grid_coords, fibers_norm,
          boundary_raw, boundary_smooth,
          gt_int, gt_names, gt_valid,           # grid-level GT
          n_points, n_reassigned,

          # Metric dicts
          metrics       — headline (cell-level if available, else grid)
          metrics_grid  — always present
          metrics_cell  — empty dict if cells not available

          # Cell-level outputs (None if cells not available)
          cell_coords, cell_predicted, cell_gt_int, cell_gt_names,
          n_cells,

          # Backward-compatible flat keys (= cell-level if available,
          # else grid-level)
          ari, nmi
    """
    if verbose:
        print(f"\n{'╔' + '═' * 58 + '╗'}")
        print(f"{'║'} SHEAF BOUNDARY — {slice_name:<40s}{'║'}")
        print(f"{'╚' + '═' * 58 + '╝'}")

    # 1. Select slice
    mol_single = io.select_slice(mol, slice_name, verbose=verbose)

    # 2. Build fibers
    grid_coords, raw_fibers, gene_names = fibers.build_kernel_fibers(
        mol_single, grid_spacing=grid_spacing, bandwidth=bandwidth,
        grid_offset=grid_offset, verbose=verbose,
    )
    fibers_norm = fibers.normalize_fibers(raw_fibers)

    # 3. Spatial graph
    edges = graph.build_spatial_graph(grid_coords, k=k_graph, verbose=verbose)

    # 4. Stalks
    stalks = sheaf.compute_stalks(
        fibers_norm, grid_coords,
        uniform_dim=uniform_dim, k_nbrs=k_nbrs, verbose=verbose,
    )

    # 5. Restriction maps
    restrictions = sheaf.compute_restriction_maps(
        stalks, edges, uniform_dim, verbose=verbose,
    )

    # 6. Boundary detection
    boundary_raw, boundary_smooth = sheaf.compute_boundary_scores(
        fibers_norm, grid_coords, stalks, edges, restrictions,
        smooth_radius=smooth_radius, verbose=verbose, k_nbrs=k_nbrs,
    )

    # 7. Boundary-aware clustering
    labels = clustering.boundary_aware_clustering(
        fibers_norm, grid_coords, boundary_smooth, edges,
        n_clusters=n_clusters, spatial_weight=spatial_weight,
        boundary_alpha=boundary_alpha,
        overcluster_factor=overcluster_factor,
        random_state=random_state, verbose=verbose,
    )

    # 8. Spatial connectivity cleanup
    n_reassigned = 0
    if cleanup:
        labels, n_reassigned = clustering.spatial_cleanup(
            labels, edges, grid_coords=grid_coords,
            min_fragment_frac=min_fragment_frac, verbose=verbose,
        )

    # 9. Grid-level evaluation (always)
    gt_int, gt_valid, gt_names = evaluation.map_ground_truth(
        mol_single, grid_coords,
    )
    if verbose:
        print("\n  GRID-LEVEL EVALUATION (diagnostic)")
        print("  " + "-" * 40)
    metrics_grid = evaluation.evaluate_clustering_full(
        labels, grid_coords,
        gt_int=gt_int, gt_valid=gt_valid,
        pas_k=pas_k, chaos_min_size=chaos_min_size,
        asw_sample_size=asw_sample_size,
        compute_lisi_flag=compute_lisi,
        lisi_perplexity=lisi_perplexity,
        name="Sheaf boundary (grid-level)", verbose=verbose,
    )

    # 10. Cell-level evaluation (when available — SDMBench-comparable)
    cell_pkg = dict(metrics={}, cell_coords=None, cell_predicted=None,
                    cell_gt_int=None, cell_gt_names=[])
    if eval_at_cell_level:
        cell_pkg = _cell_level_evaluation(
            grid_coords, labels, mol_single,
            pas_k=pas_k, chaos_min_size=chaos_min_size,
            asw_sample_size=asw_sample_size,
            compute_lisi=compute_lisi, lisi_perplexity=lisi_perplexity,
            verbose=verbose, compute_autocorr=compute_autocorr,
        )

    metrics_cell = cell_pkg["metrics"]
    # Headline: cell-level if available, else grid-level
    metrics_headline = metrics_cell if metrics_cell else metrics_grid

    return dict(
        slice_name=slice_name,
        labels=labels,
        grid_coords=grid_coords,
        fibers_norm=fibers_norm,
        boundary_raw=boundary_raw,
        boundary_smooth=boundary_smooth,
        gt_int=gt_int, gt_names=gt_names, gt_valid=gt_valid,
        n_points=len(grid_coords),
        n_reassigned=n_reassigned,
        # Metric dicts
        metrics=metrics_headline,
        metrics_grid=metrics_grid,
        metrics_cell=metrics_cell,
        # Cell-level outputs
        cell_coords=cell_pkg["cell_coords"],
        cell_predicted=cell_pkg["cell_predicted"],
        cell_gt_int=cell_pkg["cell_gt_int"],
        cell_gt_names=cell_pkg["cell_gt_names"],
        n_cells=(0 if cell_pkg["cell_coords"] is None
                 else len(cell_pkg["cell_coords"])),
        # Backward-compatible flat keys (now cell-level when available)
        ari=metrics_headline.get("ari", np.nan),
        nmi=metrics_headline.get("nmi", np.nan),
    )


# ---------------------------------------------------------------------------
# Multi-slice driver
# ---------------------------------------------------------------------------

def run_all_slices(mol_path, n_clusters=8,
                   grid_spacing=15, bandwidth=25,
                   k_graph=6, k_nbrs=50, uniform_dim=6,
                   smooth_radius=60.0, spatial_weight=0.5,
                   boundary_alpha=0.5,
                   overcluster_factor=1.0,
                   cleanup=True, min_fragment_frac=0.01,
                   eval_at_cell_level=True,
                   pas_k=10, chaos_min_size=3,
                   asw_sample_size=None,
                   compute_lisi=False, lisi_perplexity=30,
                   random_state=42, compute_autocorr=False,
                   verbose=True):
    """
    Run sheaf boundary detection on every slice in the dataset.

    Parameters
    ----------
    mol_path : str
        Path to molecule CSV.
    eval_at_cell_level : bool
        See :func:`run_single_slice`.
    Other parameters
        See :func:`run_single_slice`.
    verbose : bool

    Returns
    -------
    summary : DataFrame
        Per-slice summary with columns::

          slice, n_points, n_cells,
          ari, nmi, hom, com, chaos, pas, asw,         # headline
          ari_grid, nmi_grid, hom_grid, com_grid,      # grid diagnostic
          chaos_grid, pas_grid, asw_grid,
          n_reassigned, time_s, status

        ``ari``/``nmi`` are **cell-level** when available, grid-level
        otherwise. ``*_grid`` columns are always grid-level.
    all_results : list of dict or None
    """
    if verbose:
        print("╔" + "═" * 62 + "╗")
        print("║  SHEAF BOUNDARY — ALL SLICES BENCHMARK                    ║")
        print("╚" + "═" * 62 + "╝")

    t_total = time.time()

    mol = io.load_data(mol_path, verbose=verbose)
    slice_info = io.identify_slices(mol, verbose=verbose)
    slice_names = sorted(slice_info["slice_name"].tolist())

    all_results = []
    rows = []

    for si, slice_name in enumerate(slice_names):
        if verbose:
            print(f"\n{'#' * 60}")
            print(f"  SLICE {si + 1}/{len(slice_names)}: {slice_name}")
            print(f"{'#' * 60}")

        t_slice = time.time()
        try:
            result = run_single_slice(
                mol, slice_name,
                n_clusters=n_clusters,
                grid_spacing=grid_spacing, bandwidth=bandwidth,
                k_graph=k_graph, k_nbrs=k_nbrs, uniform_dim=uniform_dim,
                smooth_radius=smooth_radius, spatial_weight=spatial_weight,
                boundary_alpha=boundary_alpha,
                overcluster_factor=overcluster_factor,
                cleanup=cleanup, min_fragment_frac=min_fragment_frac,
                eval_at_cell_level=eval_at_cell_level,
                pas_k=pas_k, chaos_min_size=chaos_min_size,
                asw_sample_size=asw_sample_size,
                compute_lisi=compute_lisi,
                lisi_perplexity=lisi_perplexity,
                random_state=random_state,
                compute_autocorr=compute_autocorr,
                verbose=verbose,
            )
            elapsed = time.time() - t_slice

            row = {
                "slice": slice_name,
                "n_points": result["n_points"],
                "n_cells": result["n_cells"],
                "n_reassigned": result["n_reassigned"],
                "time_s": elapsed,
                "status": "ok",
            }
            for k in _ALL_METRICS:
                row[k] = result["metrics"].get(k, np.nan)
                row[f"{k}_grid"] = result["metrics_grid"].get(k, np.nan)
            if compute_lisi:
                row["lisi_median"] = result["metrics"].get(
                    "lisi_median", np.nan,
                )
                row["lisi_median_grid"] = result["metrics_grid"].get(
                    "lisi_median", np.nan,
                )
            rows.append(row)
            all_results.append(result)

        except Exception as e:
            elapsed = time.time() - t_slice
            if verbose:
                print(f"  *** ERROR on {slice_name}: {e}")
            err_row = {
                "slice": slice_name, "n_points": 0, "n_cells": 0,
                "n_reassigned": 0,
                "time_s": elapsed, "status": f"error: {e}",
            }
            for k in _ALL_METRICS:
                err_row[k] = np.nan
                err_row[f"{k}_grid"] = np.nan
            if compute_lisi:
                err_row["lisi_median"] = np.nan
                err_row["lisi_median_grid"] = np.nan
            rows.append(err_row)
            all_results.append(None)

    col_order = (
        ["slice", "n_points", "n_cells"]
        + _ALL_METRICS
        + (["lisi_median"] if compute_lisi else [])
        + [f"{k}_grid" for k in _ALL_METRICS]
        + (["lisi_median_grid"] if compute_lisi else [])
        + ["n_reassigned", "time_s", "status"]
    )
    summary = pd.DataFrame(rows)[col_order]
    total_time = time.time() - t_total

    if verbose:
        _print_summary_table(summary, total_time, with_lisi=compute_lisi)

    return summary, all_results


def _print_summary_table(summary, total_time, with_lisi=False):
    """
    Print headline (cell-level when available) per-slice and mean rows,
    plus a grid-level diagnostic row at the bottom.
    """
    valid_any = summary[summary["status"] == "ok"]
    n_valid = len(valid_any)

    # Detect whether any slice actually has cell-level metrics
    has_cells = (
        valid_any.get("n_cells", pd.Series([0])).fillna(0) > 0
    ).any()
    level_label = "cell-level" if has_cells else "grid-level"

    print("\n\n" + "=" * 105)
    print(f"RESULTS SUMMARY — ALL SLICES  (headline: {level_label})")
    print("=" * 105)
    print(f"\n  Slices with results: {n_valid}")
    print(f"  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")

    headers = (
        ["Slice", "N pts", "N cells"]
        + [m.upper() for m in _ALL_METRICS]
    )
    if with_lisi:
        headers.append("LISI")
    headers.append("Clnd")

    name_w = max(8, summary["slice"].str.len().max() + 1)
    fmt_row = (
        f"  {{:<{name_w}s}} {{:>7s}} {{:>7s}}  "
        + "  ".join(f"{{:>6s}}" for _ in _ALL_METRICS)
        + ("  {:>6s}" if with_lisi else "")
        + "  {:>6s}"
    )
    sep_width = (name_w + 2 * 8 + 2 + 8 * len(_ALL_METRICS)
                 + (8 if with_lisi else 0) + 8)

    print()
    print(fmt_row.format(*headers))
    print("  " + "-" * sep_width)

    for _, r in summary.iterrows():
        if r["status"] != "ok":
            continue
        row = [
            r["slice"],
            f"{int(r['n_points']):d}",
            f"{int(r['n_cells']):d}" if pd.notna(r.get("n_cells")) else "—",
        ]
        row += [f"{r[m]:.3f}" if pd.notna(r[m]) else "  n/a"
                for m in _ALL_METRICS]
        if with_lisi:
            row.append(f"{r['lisi_median']:.3f}"
                       if pd.notna(r.get("lisi_median", np.nan)) else "  n/a")
        row.append(f"{int(r['n_reassigned']):d}")
        print(fmt_row.format(*row))

    if n_valid > 0:
        print("  " + "-" * sep_width)
        mean_row = [
            "MEAN",
            f"{int(valid_any['n_points'].mean()):d}",
            f"{int(valid_any['n_cells'].mean()):d}"
            if valid_any["n_cells"].notna().any() else "—",
        ]
        for m in _ALL_METRICS:
            vals = valid_any[m].dropna()
            mean_row.append(f"{vals.mean():.3f}" if len(vals) else "  n/a")
        if with_lisi:
            vals = valid_any["lisi_median"].dropna()
            mean_row.append(f"{vals.mean():.3f}" if len(vals) else "  n/a")
        mean_row.append("")
        print(fmt_row.format(*mean_row))

        # Grid-level diagnostic line
        if has_cells:
            print()
            print("  Grid-level diagnostic (mean across slices):")
            grid_parts = []
            for m in _ALL_METRICS:
                col = f"{m}_grid"
                if col in valid_any.columns:
                    vals = valid_any[col].dropna()
                    if len(vals):
                        grid_parts.append(f"{m.upper()}={vals.mean():.3f}")
            if grid_parts:
                print("    " + "  ".join(grid_parts))

        print("\n  Higher is better: ARI, NMI, HOM, COM, ASW"
              + (", LISI" if with_lisi else ""))
        print("  Lower  is better: CHAOS, PAS")