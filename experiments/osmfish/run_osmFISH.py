#!/usr/bin/env python
"""
Run sheaf boundary detection on osmFISH somatosensory cortex data.

The pipeline runs segmentation-free on raw molecule coordinates.
Evaluation is done at two levels:

  - Grid level    : computed automatically inside pipeline.run_single_slice
                    on grid points (informative, not SDMBench-comparable).
  - Cell level    : each cell centroid is assigned the predicted label of
                    its nearest grid point and compared against the cell's
                    ground-truth annotation. This is the SDMBench-comparable
                    evaluation.

Both levels report the full SDMBench metric suite (Yuan et al. 2024,
*Nat. Methods*): ARI, NMI, HOM, COM, CHAOS, PAS, ASW.

Data files required:
    1. mRNA_coords_raw_counting.hdf5   — raw molecule (x, y) per gene
    2. osmfish_20260512084934.h5ad      — cell-level AnnData with ground truth

Usage:
    python run_osmfish.py
    python run_osmfish.py --grid-spacing 150 --bandwidth 500
    python run_osmfish.py --compute-lisi          # adds LISI to the suite
    python run_osmfish.py --sweep
"""

import argparse
import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


from spsheaf import pipeline, evaluation

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
PX_TO_UM = 0.065          # 1 pixel = 0.065 μm
SLICE_NAME = "osmFISH_single"
_METRIC_KEYS = ["ari", "nmi", "hom", "com", "chaos", "pas", "asw"]
# ---------------------------------------------------------------------------


# ===================================================================
# DATA LOADING
# ===================================================================

def load_osmfish_molecules(hdf5_path):
    """
    Load raw molecule coordinates from the osmFISH HDF5 file.

    Each dataset is named ``<Gene>_Hybridization<N>`` and contains
    an (M, 2) array of (x, y) in pixels.  Both axes are negated to
    match the h5ad spatial orientation.
    """
    rows = []
    with h5py.File(hdf5_path, "r") as f:
        for name in f.keys():
            gene = name.split("_Hybridization")[0]
            coords = np.array(f[name])
            rows.append(pd.DataFrame({
                "x": -coords[:, 0],
                "y": -coords[:, 1],
                "gene": gene,
            }))
    mol = pd.concat(rows, ignore_index=True)
    mol["matched_h5ad_file"] = SLICE_NAME
    return mol


REGISTRATION_FILE = "registration.json"


def registration_translation():
    """Translation (px) taking AnnData cell coordinates into the molecule grid
    frame, as fitted label-free by ``experiments.osmfish.register``; ``None``
    if it has not been fitted yet."""
    import json
    from spsheaf import paths
    f = paths.results("osmfish") / REGISTRATION_FILE
    if not f.exists():
        return None
    return np.asarray(json.loads(f.read_text())["translation"], dtype=float)


def load_cell_ground_truth(h5ad_path):
    """Load cell centroids and ground-truth labels from AnnData."""
    adata = ad.read_h5ad(h5ad_path)
    return adata.obsm["spatial"].copy(), adata.obs["ground_truth"].values


# ===================================================================
# CELL-LEVEL EVALUATION
# ===================================================================

def evaluate_at_cell_level(grid_coords, grid_labels,
                           cell_coords, cell_gt_str,
                           pas_k=10, chaos_min_size=3,
                           asw_sample_size=None,
                           compute_lisi=False, lisi_perplexity=30,
                           verbose=True):
    """
    Map grid predictions to cells and compute the full SDMBench suite.

    The grid is much denser than the cells (~66k grid points vs
    ~5k cells), so the nearest grid point to any cell centroid is
    always close. Grid and h5ad may use different origins so we align
    coordinate frames by matching centers.

    Parameters
    ----------
    grid_coords : (n_grid, 2)  — from pipeline result
    grid_labels : (n_grid,)    — predicted cluster indices
    cell_coords : (n_cells, 2) — from h5ad obsm['spatial']
    cell_gt_str : (n_cells,)   — ground-truth region strings
    pas_k, chaos_min_size, asw_sample_size, compute_lisi, lisi_perplexity
        Evaluation parameters, forwarded to evaluation.evaluate_clustering_full.

    Returns
    -------
    metrics : dict
        Full SDMBench metrics at cell level.
    cell_predicted : (n_cells,) ndarray
        Integer predicted cluster labels (SDMBench-export ready).
    cell_coords_aligned : (n_cells, 2) ndarray
        Cell coordinates after frame alignment with the grid.
    cell_gt_int : (n_cells,) ndarray
        Integer-encoded ground-truth labels (-1 where missing).
    gt_names : list of str
        Domain name in label-index order.
    """
    # Align cell centroids to the grid coordinate frame. The annotation and
    # the molecule table are in different frames; the translation between
    # them is recovered WITHOUT the labels by experiments/osmfish/register.py
    # (composition correlation, as benchmarks/segfree/osm_register.py does for
    # the competitors) and stored under data/results/osmfish/. Matching the two
    # centroids -- the fallback -- is biased by ~130 um, because the molecule
    # cloud extends beyond the annotated cells asymmetrically; it depressed the
    # published osmFISH score from 0.619 to 0.538 ARI.
    translation = registration_translation()
    if translation is None:
        warnings.warn("osmFISH registration not found; falling back to centroid "
                      "alignment, which is biased by ~130 um. Run "
                      "python -m experiments.osmfish.register", stacklevel=2)
        translation = grid_coords.mean(axis=0) - cell_coords.mean(axis=0)
    cell_coords_aligned = cell_coords + translation

    # Each cell → nearest grid point → predicted label
    tree = cKDTree(grid_coords)
    dists, idx = tree.query(cell_coords_aligned, k=1)
    cell_predicted = grid_labels[idx]

    if verbose:
        print(
            f"  Nearest grid point distance: "
            f"median={np.median(dists):.1f} px "
            f"({np.median(dists)*PX_TO_UM:.1f} μm), "
            f"max={dists.max():.1f} px "
            f"({dists.max()*PX_TO_UM:.1f} μm)"
        )

    # Encode string GT as integers, masking missing/NaN entries
    gt_valid = pd.notna(cell_gt_str)
    gt_names = sorted(set(cell_gt_str[gt_valid]))
    name_to_idx = {n: i for i, n in enumerate(gt_names)}
    cell_gt_int = np.array([
        name_to_idx.get(g, -1) if pd.notna(g) else -1 for g in cell_gt_str
    ])

    # Full SDMBench suite at cell level
    metrics = evaluation.evaluate_clustering_full(
        cell_predicted, cell_coords_aligned,
        gt_int=cell_gt_int, gt_valid=gt_valid,
        pas_k=pas_k, chaos_min_size=chaos_min_size,
        asw_sample_size=asw_sample_size,
        compute_lisi_flag=compute_lisi,
        lisi_perplexity=lisi_perplexity,
        name="Sheaf boundary (cell-level)", verbose=verbose,
    )
    return metrics, cell_predicted, cell_coords_aligned, cell_gt_int, gt_names


# ===================================================================
# SDMBENCH EXPORT
# ===================================================================

def export_sdmbench(cell_predicted, output_dir,
                    method_name="sheaf_boundary",
                    filename="sheaf_boundary_1.txt"):
    """
    Write predictions in SDMBench submission format.

    Format:
        Line 1:  method name
        Lines 2…N+1:  one predicted cluster label per cell

    The cell ordering must match the h5ad cell ordering, which is
    guaranteed because load_cell_ground_truth reads adata.obs in its
    native order, and evaluate_at_cell_level preserves that order.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write(f"{method_name}\n")
        for label in cell_predicted:
            f.write(f"{label}\n")
    print(f"  SDMBench export: {path}  ({len(cell_predicted)} cells)")
    return path


def export_metrics_csv(metrics_grid, metrics_cell, output_dir,
                       filename="osmfish_metrics.csv"):
    """
    Write a two-row CSV with grid-level and cell-level metrics
    side by side, suitable for paper tables.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for level, m in [("grid", metrics_grid), ("cell", metrics_cell)]:
        row = {"level": level}
        row.update({k: m.get(k, float("nan")) for k in _METRIC_KEYS})
        if "lisi_median" in m:
            row["lisi_median"] = m["lisi_median"]
        rows.append(row)
    path = os.path.join(output_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Metrics CSV: {path}")
    return path


# ===================================================================
# PLOTTING
# ===================================================================

def hungarian_map(cell_predicted, cell_gt_int, n_domains):
    """Match predicted clusters to ground-truth domains, maximizing overlap.

    Returns ``{predicted_cluster: gt_domain}``. Lifted out of ``plot_results``
    so that the panel export and the plot use one matching, not two.
    """
    n_pred = len(set(cell_predicted))
    n_match = max(n_pred, n_domains)
    cost = np.zeros((n_match, n_match))
    for i in range(n_pred):
        for j in range(n_domains):
            cost[i, j] = -np.sum((cell_predicted == i) & (cell_gt_int == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    return dict(zip(row_ind, col_ind))


def plot_results(grid_coords, grid_labels,
                 cell_coords_aligned, cell_gt_int, gt_names, cell_predicted,
                 boundary_raw, boundary_smooth,
                 metrics_cell, output_dir):
    """Six-panel figure: GT, predicted, errors, grid, boundary raw/smooth."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    n_domains = len(gt_names)
    colors = plt.cm.tab20(np.linspace(0, 1, n_domains))

    ari = metrics_cell.get("ari", float("nan"))
    nmi = metrics_cell.get("nmi", float("nan"))

    label_map = hungarian_map(cell_predicted, cell_gt_int, n_domains)
    cell_pred_matched = np.array([label_map.get(p, -1) for p in cell_predicted])

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    subtitle = "  ".join(
        f"{k.upper()}={metrics_cell[k]:.3f}"
        for k in _METRIC_KEYS if k in metrics_cell
    )
    fig.suptitle(
        f"Sheaf Boundary Detection — osmFISH (cell-level eval)\n{subtitle}",
        fontsize=13, fontweight="bold",
    )

    # (0,0) Ground truth
    ax = axes[0, 0]
    for i, name in enumerate(gt_names):
        mask = cell_gt_int == i
        ax.scatter(cell_coords_aligned[mask, 0],
                   cell_coords_aligned[mask, 1],
                   s=4, c=[colors[i]], label=name, alpha=0.7)
    ax.set_title(f"Ground truth ({n_domains} domains, "
                 f"{(cell_gt_int >= 0).sum()} valid cells)")
    ax.legend(fontsize=5, markerscale=3, loc="upper left")
    ax.set_aspect("equal")

    # (0,1) Predicted (Hungarian-matched)
    ax = axes[0, 1]
    for i, name in enumerate(gt_names):
        pred_cl = [p for p, g in label_map.items() if g == i]
        mask = (
            cell_predicted == pred_cl[0] if pred_cl
            else np.zeros(len(cell_predicted), dtype=bool)
        )
        ax.scatter(cell_coords_aligned[mask, 0],
                   cell_coords_aligned[mask, 1],
                   s=4, c=[colors[i]], alpha=0.7)
    ax.set_title(f"Predicted (ARI={ari:.3f}, NMI={nmi:.3f})")
    ax.set_aspect("equal")

    # (0,2) Misclassified
    ax = axes[0, 2]
    valid = cell_gt_int >= 0
    correct = (cell_pred_matched == cell_gt_int) & valid
    wrong = (~correct) & valid
    ax.scatter(cell_coords_aligned[correct, 0],
               cell_coords_aligned[correct, 1],
               s=4, c="lightgrey", alpha=0.3)
    ax.scatter(cell_coords_aligned[wrong, 0],
               cell_coords_aligned[wrong, 1],
               s=4, c="red", alpha=0.7)
    n_wrong = wrong.sum()
    ax.set_title(f"Misclassified: {n_wrong}/{valid.sum()} "
                 f"({n_wrong/valid.sum()*100:.1f}%)")
    ax.set_aspect("equal")

    # (1,0) Grid predictions
    ax = axes[1, 0]
    ax.scatter(grid_coords[:, 0], grid_coords[:, 1],
               s=1, c=grid_labels, cmap="tab20", alpha=0.5)
    ax.set_title(f"Grid clusters ({len(set(grid_labels))} clusters, "
                 f"{len(grid_coords)} pts)")
    ax.set_aspect("equal")

    # (1,1) Raw boundary
    ax = axes[1, 1]
    sc = ax.scatter(grid_coords[:, 0], grid_coords[:, 1],
                    s=1, c=boundary_raw, cmap="hot", alpha=0.8)
    plt.colorbar(sc, ax=ax)
    ax.set_title("Raw sheaf boundary score")
    ax.set_aspect("equal")

    # (1,2) Smoothed boundary
    ax = axes[1, 2]
    sc = ax.scatter(grid_coords[:, 0], grid_coords[:, 1],
                    s=1, c=boundary_smooth, cmap="hot", alpha=0.8)
    plt.colorbar(sc, ax=ax)
    ax.set_title("Smoothed sheaf boundary score")
    ax.set_aspect("equal")

    for ax in axes.flat:
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

    fig.tight_layout()
    path = os.path.join(output_dir, "sheaf_boundary_osmFISH_single.png")
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Per-domain accuracy
    fig2, ax = plt.subplots(figsize=(8, 6))
    accs = []
    for i, name in enumerate(gt_names):
        mask_gt = cell_gt_int == i
        pred_cl = [p for p, g in label_map.items() if g == i]
        acc = (cell_predicted[mask_gt] == pred_cl[0]).mean() if pred_cl else 0
        accs.append(acc)
    ax.barh(range(n_domains), accs,
            color=[colors[i] for i in range(n_domains)])
    ax.set_yticks(range(n_domains))
    ax.set_yticklabels(gt_names, fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_title(f"Per-domain accuracy (ARI={ari:.3f}, NMI={nmi:.3f})")
    for i, v in enumerate(accs):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7)
    fig2.tight_layout()
    path2 = os.path.join(output_dir, "osmFISH_per_domain_accuracy.png")
    fig2.savefig(path2, dpi=400, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {path2}")


# ===================================================================
# MAIN
# ===================================================================

def _format_metrics_line(metrics, label):
    """One-line pretty-print of a metrics dict."""
    parts = [f"{k.upper()}={metrics[k]:.3f}"
             for k in _METRIC_KEYS if k in metrics]
    if "lisi_median" in metrics:
        parts.append(f"LISI={metrics['lisi_median']:.3f}")
    return f"  {label:<14s}: " + "  ".join(parts)


def run(args):
    print("=" * 70)
    print("Sheaf Boundary Detection — osmFISH (segmentation-free)")
    print("=" * 70)

    # --- Load molecules (no ground truth injected) ---
    print(f"\n[1/4] Loading molecules from {args.mol_path} ...")
    mol = load_osmfish_molecules(args.mol_path)
    print(f"       {len(mol):,} molecules, {mol['gene'].nunique()} genes")

    # --- Print parameters ---
    print(f"\n       Parameters:")
    for name, val in [("grid_spacing", args.grid_spacing),
                      ("bandwidth", args.bandwidth),
                      ("smooth_radius", args.smooth_radius)]:
        print(f"         {name:18s} = {val} px  ({val*PX_TO_UM:.1f} μm)")
    print(f"         {'uniform_dim':18s} = {args.uniform_dim}")
    print(f"         {'n_clusters':18s} = {args.n_clusters}")
    print(f"         {'spatial_weight':18s} = {args.spatial_weight}")
    print(f"         {'boundary_alpha':18s} = {args.boundary_alpha}")
    print(f"         {'pas_k':18s} = {args.pas_k}")
    print(f"         {'chaos_min_size':18s} = {args.chaos_min_size}")
    if args.asw_sample_size:
        print(f"         {'asw_sample_size':18s} = {args.asw_sample_size}")
    if args.compute_lisi:
        print(f"         {'compute_lisi':18s} = True "
              f"(perplexity={args.lisi_perplexity})")

    # --- Run pipeline blind (now returns full grid-level metric suite) ---
    print(f"\n[2/4] Running sheaf pipeline ...")
    result = pipeline.run_single_slice(
        mol=mol,
        slice_name=SLICE_NAME,
        n_clusters=args.n_clusters,
        grid_spacing=args.grid_spacing,
        bandwidth=args.bandwidth,
        smooth_radius=args.smooth_radius,
        spatial_weight=args.spatial_weight,
        k_graph=args.k_graph,
        k_nbrs=args.k_nbrs,
        uniform_dim=args.uniform_dim,
        boundary_alpha=args.boundary_alpha,
        overcluster_factor=args.overcluster_factor,
        cleanup=not args.no_cleanup,
        pas_k=args.pas_k,
        chaos_min_size=args.chaos_min_size,
        asw_sample_size=args.asw_sample_size,
        compute_lisi=args.compute_lisi,
        lisi_perplexity=args.lisi_perplexity,
        verbose=args.verbose,
    )
    grid_coords = result["grid_coords"]
    grid_labels = result["labels"]
    metrics_grid = result["metrics"]
    print(f"       {len(grid_coords)} grid points, "
          f"{len(set(grid_labels))} clusters")

    # --- Cell-level evaluation ---
    metrics_cell = {}
    cell_predicted = cell_coords_aligned = cell_gt_int = None
    gt_names = []

    if args.h5ad_path and os.path.exists(args.h5ad_path):
        print(f"\n[3/4] Cell-level evaluation ...")
        cell_coords, cell_gt_str = load_cell_ground_truth(args.h5ad_path)
        (metrics_cell, cell_predicted, cell_coords_aligned,
         cell_gt_int, gt_names) = evaluate_at_cell_level(
            grid_coords, grid_labels, cell_coords, cell_gt_str,
            pas_k=args.pas_k, chaos_min_size=args.chaos_min_size,
            asw_sample_size=args.asw_sample_size,
            compute_lisi=args.compute_lisi,
            lisi_perplexity=args.lisi_perplexity,
            verbose=args.verbose,
        )
    else:
        print(f"\n[3/4] No h5ad — skipping cell-level evaluation")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("RESULT (full SDMBench suite)")
    print(_format_metrics_line(metrics_grid, "Grid level"))
    if metrics_cell:
        print(_format_metrics_line(metrics_cell, "Cell level"))
    if result.get("n_reassigned", 0) > 0:
        print(f"  Cleanup        : {result['n_reassigned']} "
              f"fragments reassigned")
    print(f"{'=' * 70}")

    # --- SDMBench export ---
    if cell_predicted is not None:
        export_sdmbench(
            cell_predicted,
            output_dir=args.output_dir,
            method_name=args.method_name,
            filename=f"{args.method_name}_1.txt",
        )
        export_metrics_csv(
            metrics_grid, metrics_cell,
            output_dir=args.output_dir,
        )

    # --- Panel data for the manuscript figures ---
    # Figure 3 is drawn in figures/, not here, so what it needs is saved
    # rather than plotted. The grid-level fields (e, f, g) exist only in
    # memory during this run; without this they cannot be redrawn at all.
    if cell_predicted is not None and not args.no_export:
        from experiments.common.panel_export import export_panels
        label_map = hungarian_map(cell_predicted, cell_gt_int, len(gt_names))
        export_panels(
            args.output_dir, "osmfish",
            arrays=dict(
                grid_coords=grid_coords,
                grid_labels=grid_labels,
                boundary_raw=result["boundary_raw"],
                boundary_smooth=result["boundary_smooth"],
                cell_coords=cell_coords_aligned,
                cell_gt_int=cell_gt_int,
                cell_predicted=cell_predicted,
                cell_pred_matched=np.array(
                    [label_map.get(p, -1) for p in cell_predicted]),
            ),
            meta=dict(
                gt_names=list(gt_names),
                label_map={str(k): int(v) for k, v in label_map.items()},
                ari=metrics_cell.get("ari"), nmi=metrics_cell.get("nmi"),
                grid_spacing=args.grid_spacing, bandwidth=args.bandwidth,
                n_clusters=args.n_clusters, dataset="osmFISH",
            ),
        )

    # --- Plot ---
    if not args.no_plot and cell_predicted is not None:
        print(f"\n[4/4] Plotting ...")
        plot_results(
            grid_coords, grid_labels,
            cell_coords_aligned, cell_gt_int, gt_names, cell_predicted,
            result["boundary_raw"], result["boundary_smooth"],
            metrics_cell,
            output_dir=args.output_dir,
        )

    return result, metrics_grid, metrics_cell


def sweep(args):
    print("=" * 70)
    print("Parameter sweep — osmFISH (cell-level evaluation)")
    print("=" * 70)

    mol = load_osmfish_molecules(args.mol_path)
    cell_coords, cell_gt_str = None, None
    if args.h5ad_path and os.path.exists(args.h5ad_path):
        cell_coords, cell_gt_str = load_cell_ground_truth(args.h5ad_path)

    configs = []
    for gs in [100, 150, 200]:
        for bw in [300, 500, 700]:
            for sr in [300, 500, 900]:
                for ud in [6, 9]:
                    for sw in [0.3, 0.5]:
                        for ba in [0.5, 1.0]:
                            configs.append(dict(
                                grid_spacing=gs, bandwidth=bw,
                                smooth_radius=sr, uniform_dim=ud,
                                spatial_weight=sw, boundary_alpha=ba,
                            ))

    print(f"\n{len(configs)} configurations\n")
    hdr = (
        f"{'grid':>5} {'bw':>5} {'σ_s':>5} {'d':>3} "
        f"{'α_s':>5} {'α_b':>5} | "
        f"{'ARI':>7} {'NMI':>7} {'HOM':>7} {'CHAOS':>7} {'PAS':>7} {'ASW':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    best_ari, best_cfg, best_metrics = -np.inf, None, {}
    rows = []

    for cfg in configs:
        try:
            res = pipeline.run_single_slice(
                mol=mol, slice_name=SLICE_NAME,
                n_clusters=args.n_clusters,
                k_graph=args.k_graph, k_nbrs=args.k_nbrs,
                overcluster_factor=1.0, cleanup=True,
                pas_k=args.pas_k, chaos_min_size=args.chaos_min_size,
                asw_sample_size=args.asw_sample_size,
                verbose=False,
                **cfg,
            )
            if cell_coords is not None:
                m_cell, *_ = evaluate_at_cell_level(
                    res["grid_coords"], res["labels"],
                    cell_coords, cell_gt_str,
                    pas_k=args.pas_k, chaos_min_size=args.chaos_min_size,
                    asw_sample_size=args.asw_sample_size,
                    verbose=False,
                )
            else:
                m_cell = {}
        except Exception as e:
            print(f"  ERROR: {e}")
            m_cell = {}

        def g(k):
            return m_cell.get(k, float("nan"))

        print(
            f"{cfg['grid_spacing']:5d} {cfg['bandwidth']:5d} "
            f"{cfg['smooth_radius']:5d} {cfg['uniform_dim']:3d} "
            f"{cfg['spatial_weight']:5.1f} {cfg['boundary_alpha']:5.1f}"
            f" | {g('ari'):7.4f} {g('nmi'):7.4f} {g('hom'):7.4f} "
            f"{g('chaos'):7.4f} {g('pas'):7.4f} {g('asw'):7.4f}"
        )

        rows.append({**cfg, **{k: g(k) for k in _METRIC_KEYS}})
        if g("ari") > best_ari:
            best_ari = g("ari")
            best_cfg = cfg.copy()
            best_metrics = dict(m_cell)

    print(f"\n{'=' * 70}")
    print(f"BEST (by ARI):")
    if best_metrics:
        print(_format_metrics_line(best_metrics, "Cell level"))
    if best_cfg:
        for k, v in best_cfg.items():
            suffix = f"  ({v*PX_TO_UM:.1f} μm)" if k in (
                "grid_spacing", "bandwidth", "smooth_radius") else ""
            print(f"  {k:18s} = {v}{suffix}")

    os.makedirs(args.output_dir, exist_ok=True)
    csv = os.path.join(args.output_dir, "osmfish_sweep.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\nSaved {csv}")


# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Sheaf boundary detection on osmFISH (SDMBench suite).")
    from spsheaf import paths
    osm = paths.raw("osmfish")
    p.add_argument("--mol-path",
                   default=str(osm / "mRNA_coords_raw_counting.hdf5"))
    p.add_argument("--h5ad-path",
                   default=str(osm / "osmfish_20260512084934.h5ad"))
    # Spatial (pixels, 1 px = 0.065 μm)
    p.add_argument("--grid-spacing", type=float, default=150)
    p.add_argument("--bandwidth", type=float, default=500)
    p.add_argument("--smooth-radius", type=float, default=300)
    # Sheaf
    p.add_argument("--k-graph", type=int, default=6)
    p.add_argument("--k-nbrs", type=int, default=50)
    p.add_argument("--uniform-dim", type=int, default=9)
    p.add_argument("--boundary-alpha", type=float, default=0.5)
    p.add_argument("--spatial-weight", type=float, default=0.5)
    p.add_argument("--n-clusters", type=int, default=11)
    p.add_argument("--overcluster-factor", type=float, default=1.0)
    # SDMBench eval params
    p.add_argument("--pas-k", type=int, default=10,
                   help="PAS neighborhood size (SDMBench default 10).")
    p.add_argument("--chaos-min-size", type=int, default=3,
                   help="Min cluster size to contribute to CHAOS.")
    p.add_argument("--asw-sample-size", type=int, default=None,
                   help="Subsample size for ASW; None uses all points. "
                        "Set to ~20000 for very large grids.")
    p.add_argument("--compute-lisi", action="store_true",
                   help="Also compute median LISI "
                        "(requires rpy2 + R `lisi`).")
    p.add_argument("--lisi-perplexity", type=int, default=30)
    # Flags
    p.add_argument("--no-cleanup", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-export", action="store_true",
                   help="Skip writing the panel data the manuscript "
                        "figures are drawn from.")
    p.add_argument("--method-name", default="sheaf_boundary",
                   help="Method name for SDMBench submission "
                        "(first line of .txt).")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--output-dir",
                   default=str(paths.results("osmfish")
                               / "output_osmFISH_boundary"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        sweep(args)
    else:
        run(args)   