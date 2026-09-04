#!/usr/bin/env python
"""
Run sheaf boundary detection on Stereo-seq E16.5 mouse embryo data.

Supports two strategies:
  - COARSE (default): whole embryo at ~200 μm resolution, ~3k grid points.
    Detects organ/tissue-level boundaries.  Runs in seconds once fibers
    are built.
  - ROI: crop a region of interest and run at MERFISH-like resolution
    (~15 μm grid).  Detects cell-type-level boundaries within a tissue.

The pipeline runs segmentation-free on raw molecule coordinates.
Gene filtering is applied to reduce the ~20k whole-transcriptome
genes to the top N by total count (default 300).

Evaluation uses FICTURE's pixel-level predictions as reference:
each FICTURE pixel is assigned the predicted label of its nearest
grid point, then ARI/NMI are computed.

Coordinate system:
    Stereo-seq bin1 units: 1 unit ≈ 0.5 μm.
    All spatial parameters are in these native units.
    Approximate μm equivalents are printed at runtime.

Data files required:
    1. E16.5_E1S3_GEM_bin1.tsv.gz     — raw molecule (geneID, x, y, MIDCount)
    2. FICTURE pixel-level output      — pixel-level domain predictions

Usage:
    # Coarse whole embryo (Strategy 2)
    python run_seqscope.py

    # ROI mode (Strategy 1): crop 2×2 mm around center (x0, y0)
    python run_seqscope.py --roi 15000 20000 4000 4000 \\
        --grid-spacing 30 --bandwidth 50 --smooth-radius 120 \\
        --n-clusters 12 --k-nbrs 50 --top-genes 500

    # Parameter sweep
    python run_seqscope.py --sweep
"""

import argparse
import os
import sys
import warnings
import gzip

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


from spsheaf import pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
UNIT_TO_UM = 0.5          # 1 Stereo-seq bin1 unit ≈ 0.5 μm
SLICE_NAME = "StereoSeq_E16.5_E1S3"
# ---------------------------------------------------------------------------


def _um(val):
    """Convert native units to μm for display."""
    return val * UNIT_TO_UM


# ===================================================================
# DATA LOADING
# ===================================================================

def _count_header_lines(path):
    """Count '#'-prefixed header lines in a GEM file."""
    open_func = gzip.open if path.endswith('.gz') else open
    skip = 0
    with open_func(path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                skip += 1
            else:
                break
    return skip


def load_stereoseq_filtered(path, top_n, roi=None, subsample=None):
    """
    Load Stereo-seq GEM file with gene filtering, ROI cropping, and
    optional subsampling — all in a memory-efficient two-pass approach.

    Pass 1: scan the gene column only to identify the top_n genes.
    Pass 2: load only rows matching those genes (and ROI if given).

    This avoids ever holding the full 500M+ row DataFrame in memory.

    Parameters
    ----------
    path : str
        Path to E16.5_E1S3_GEM_bin1.tsv.gz
    top_n : int
        Number of top genes by total count to retain.
    roi : tuple (x0, y0, width, height) or None
        Crop to this bounding box (native units).
    subsample : float or None
        If set, randomly subsample this fraction of molecules after
        filtering.  E.g. 0.1 keeps 10%.  At coarse grid spacing
        (≥200 μm) this has negligible effect on KDE accuracy.

    Returns
    -------
    mol : DataFrame with columns gene, x, y, MIDCount, matched_h5ad_file
    """
    skip = _count_header_lines(path)
    chunk_size = 5_000_000

    # --- Pass 1: count genes ---
    print("  Pass 1: scanning gene frequencies ...")
    gene_counts = {}
    n_total = 0
    for chunk in pd.read_csv(path, sep='\t', skiprows=skip,
                             chunksize=chunk_size, usecols=['geneID']):
        for g, cnt in chunk['geneID'].value_counts().items():
            gene_counts[g] = gene_counts.get(g, 0) + cnt
        n_total += len(chunk)

    # Sort and pick top_n
    sorted_genes = sorted(gene_counts, key=gene_counts.get, reverse=True)
    keep_genes = set(sorted_genes[:top_n])
    n_all_genes = len(gene_counts)
    kept_mol_est = sum(gene_counts[g] for g in keep_genes)
    print(f"  {n_total:,} molecules, {n_all_genes:,} genes → "
          f"top {top_n} ({kept_mol_est/n_total*100:.1f}% of molecules)")

    # --- Pass 2: load filtered rows ---
    print("  Pass 2: loading filtered molecules ...")
    chunks = []
    for chunk in pd.read_csv(path, sep='\t', skiprows=skip,
                             chunksize=chunk_size):
        chunk.rename(columns={"geneID": "gene"}, inplace=True)
        # Gene filter
        mask = chunk['gene'].isin(keep_genes)
        # ROI filter
        if roi is not None:
            x0, y0, w, h = roi
            mask &= ((chunk['x'] >= x0) & (chunk['x'] < x0 + w) &
                      (chunk['y'] >= y0) & (chunk['y'] < y0 + h))
        filtered = chunk[mask]
        if len(filtered) > 0:
            chunks.append(filtered)

    mol = pd.concat(chunks, ignore_index=True)
    mol["matched_h5ad_file"] = SLICE_NAME

    if roi is not None:
        print(f"  ROI crop: ({roi[0]}, {roi[1]}) + "
              f"({roi[2]} × {roi[3]}) units "
              f"= ({_um(roi[2]):.0f} × {_um(roi[3]):.0f}) μm")

    print(f"  Loaded {len(mol):,} molecules, {mol['gene'].nunique()} genes")

    # --- Optional subsampling ---
    if subsample is not None and subsample < 1.0:
        n_before = len(mol)
        mol = mol.sample(frac=subsample, random_state=42)
        mol = mol.reset_index(drop=True)
        print(f"  Subsampled: {n_before:,} → {len(mol):,} "
              f"({subsample*100:.0f}%)")

    return mol


def load_ficture_ground_truth(path):
    """
    Load FICTURE's pixel-level cluster predictions as reference.
    Parses the header to reverse coordinate scaling.
    """
    scale = 100.0
    offset_x = 0.0
    offset_y = 0.0

    open_func = gzip.open if path.endswith('.gz') else open
    with open_func(path, 'rt') as f:
        for line in f:
            if line.startswith('##OFFSET'):
                parts = line.strip().strip('#').split(';')
                for p in parts:
                    if '=' in p:
                        k, v = p.split('=')
                        if k == 'SCALE': scale = float(v)
                        elif k == 'OFFSET_X': offset_x = float(v)
                        elif k == 'OFFSET_Y': offset_y = float(v)
            elif not line.startswith('#'):
                break

    df = pd.read_csv(path, sep='\t', skiprows=3)
    if '#BLOCK' in df.columns:
        df.rename(columns={'#BLOCK': 'BLOCK'}, inplace=True)

    # Reverse FICTURE's scaling and offsetting
    df['x_aligned'] = df['X'] / scale + offset_x
    df['y_aligned'] = df['Y'] / scale + offset_y

    return df[['x_aligned', 'y_aligned']].values, df['K1'].values


# ===================================================================
# EVALUATION
# ===================================================================

def evaluate_at_ficture_level(grid_coords, grid_labels,
                              ficture_coords, ficture_labels):
    """
    Map grid predictions to FICTURE pixels and compute ARI / NMI.
    Aligns coordinate centers before matching.
    """
    grid_center = grid_coords.mean(axis=0)
    ficture_center = ficture_coords.mean(axis=0)
    ficture_coords_aligned = ficture_coords + (grid_center - ficture_center)

    tree = cKDTree(grid_coords)
    dists, idx = tree.query(ficture_coords_aligned, k=1)
    ficture_predicted = grid_labels[idx]

    ari = adjusted_rand_score(ficture_labels, ficture_predicted)
    nmi = normalized_mutual_info_score(ficture_labels, ficture_predicted)

    print(f"  Nearest grid point distance: "
          f"median={np.median(dists):.1f} units ({_um(np.median(dists)):.1f} μm), "
          f"max={dists.max():.1f} units ({_um(dists.max()):.1f} μm)")

    return ari, nmi, ficture_predicted, ficture_coords_aligned


# ===================================================================
# PLOTTING
# ===================================================================

def plot_results(grid_coords, grid_labels,
                 ficture_coords_aligned, ficture_labels, ficture_predicted,
                 boundary_raw, boundary_smooth,
                 ari, nmi, output_dir, mode="coarse"):
    """Six-panel figure: FICTURE ref, predicted, errors, grid, boundary."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    gt_names = sorted(set(ficture_labels))
    n_domains = len(gt_names)
    dom_map = {d: i for i, d in enumerate(gt_names)}
    ficture_gt_int = np.array([dom_map[g] for g in ficture_labels])

    colors = plt.cm.tab20(np.linspace(0, 1, min(n_domains, 20)))
    if n_domains > 20:
        c2 = plt.cm.tab20b(np.linspace(0, 1, n_domains - 20))
        colors = np.vstack([colors, c2])

    # Hungarian matching
    n_pred = len(set(ficture_predicted))
    n_match = max(n_pred, n_domains)
    cost = np.zeros((n_match, n_match))
    for i in range(n_pred):
        for j in range(n_domains):
            cost[i, j] = -np.sum((ficture_predicted == i) & (ficture_gt_int == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    label_map = dict(zip(row_ind, col_ind))
    ficture_pred_matched = np.array([label_map.get(p, -1) for p in ficture_predicted])

    pt_size = 0.5 if len(ficture_labels) > 50000 else 2
    pt_alpha = 0.3 if len(ficture_labels) > 50000 else 0.6

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(
        f"Sheaf Boundary Detection — Stereo-seq ({mode} mode)\n"
        f"ARI = {ari:.3f},  NMI = {nmi:.3f}",
        fontsize=14, fontweight="bold",
    )

    # (0,0) FICTURE reference
    ax = axes[0, 0]
    for i, name in enumerate(gt_names):
        mask = ficture_gt_int == i
        ax.scatter(ficture_coords_aligned[mask, 0],
                   ficture_coords_aligned[mask, 1],
                   s=pt_size, c=[colors[i % len(colors)]],
                   label=str(name), alpha=pt_alpha)
    ax.set_title(f"FICTURE ({n_domains} factors, {len(ficture_labels):,} pixels)")
    if n_domains <= 20:
        ax.legend(fontsize=4, markerscale=5, loc="upper left")
    ax.set_aspect("equal")

    # (0,1) Predicted
    ax = axes[0, 1]
    for i, name in enumerate(gt_names):
        pred_cl = [p for p, g in label_map.items() if g == i]
        mask = (ficture_predicted == pred_cl[0]) if pred_cl else \
               np.zeros(len(ficture_predicted), dtype=bool)
        ax.scatter(ficture_coords_aligned[mask, 0],
                   ficture_coords_aligned[mask, 1],
                   s=pt_size, c=[colors[i % len(colors)]], alpha=pt_alpha)
    ax.set_title(f"Sheaf predicted (ARI={ari:.3f}, NMI={nmi:.3f})")
    ax.set_aspect("equal")

    # (0,2) Misclassified
    ax = axes[0, 2]
    correct = ficture_pred_matched == ficture_gt_int
    ax.scatter(ficture_coords_aligned[correct, 0],
               ficture_coords_aligned[correct, 1],
               s=pt_size, c="lightgrey", alpha=0.1)
    ax.scatter(ficture_coords_aligned[~correct, 0],
               ficture_coords_aligned[~correct, 1],
               s=pt_size, c="red", alpha=pt_alpha)
    n_wrong = (~correct).sum()
    ax.set_title(f"Misclassified: {n_wrong:,}/{len(correct):,} "
                 f"({n_wrong/len(correct)*100:.1f}%)")
    ax.set_aspect("equal")

    # (1,0) Grid predictions
    ax = axes[1, 0]
    ax.scatter(grid_coords[:, 0], grid_coords[:, 1],
               s=2, c=grid_labels, cmap="tab20", alpha=0.8)
    ax.set_title(f"Grid clusters ({len(set(grid_labels))} clusters, "
                 f"{len(grid_coords):,} pts)")
    ax.set_aspect("equal")

    # (1,1) Raw boundary
    ax = axes[1, 1]
    sc = ax.scatter(grid_coords[:, 0], grid_coords[:, 1],
                    s=2, c=boundary_raw, cmap="hot", alpha=0.8)
    plt.colorbar(sc, ax=ax)
    ax.set_title("Raw sheaf boundary score")
    ax.set_aspect("equal")

    # (1,2) Smoothed boundary
    ax = axes[1, 2]
    sc = ax.scatter(grid_coords[:, 0], grid_coords[:, 1],
                    s=2, c=boundary_smooth, cmap="hot", alpha=0.8)
    plt.colorbar(sc, ax=ax)
    ax.set_title("Smoothed sheaf boundary score")
    ax.set_aspect("equal")

    for ax in axes.flat:
        ax.set_xlabel("x (units)")
        ax.set_ylabel("y (units)")

    fig.tight_layout()
    suffix = "roi" if mode == "roi" else "coarse"
    path = os.path.join(output_dir, f"sheaf_boundary_stereoseq_{suffix}.png")
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# MAIN
# ===================================================================

def run(args):
    mode = "roi" if args.roi else "coarse"

    print("=" * 60)
    print(f"Sheaf Boundary Detection — Stereo-seq E16.5 ({mode} mode)")
    print("=" * 60)

    # --- Load, filter, crop, subsample in one pass ---
    print(f"\n[1/4] Loading and filtering from {args.mol_path} ...")
    mol = load_stereoseq_filtered(
        args.mol_path,
        top_n=args.top_genes,
        roi=tuple(args.roi) if args.roi else None,
        subsample=args.subsample,
    )
    n_genes_kept = mol['gene'].nunique()

    # --- Print parameters ---
    print(f"\n       Parameters (units → μm):")
    for name, val in [("grid_spacing", args.grid_spacing),
                      ("bandwidth", args.bandwidth),
                      ("smooth_radius", args.smooth_radius)]:
        print(f"         {name:18s} = {val:.0f} units  ({_um(val):.0f} μm)")
    print(f"         {'top_genes':18s} = {n_genes_kept}")
    print(f"         {'uniform_dim':18s} = {args.uniform_dim}")
    print(f"         {'n_clusters':18s} = {args.n_clusters}")
    print(f"         {'spatial_weight':18s} = {args.spatial_weight}")
    print(f"         {'boundary_alpha':18s} = {args.boundary_alpha}")
    print(f"         {'k_nbrs':18s} = {args.k_nbrs}")
    if args.subsample:
        print(f"         {'subsample':18s} = {args.subsample}")

    # --- Run pipeline ---
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
        verbose=args.verbose,
    )
    grid_coords = result["grid_coords"]
    grid_labels = result["labels"]
    print(f"       {len(grid_coords):,} grid points, "
          f"{len(set(grid_labels))} clusters")

    # --- Evaluation ---
    ari = nmi = float("nan")
    ficture_predicted = ficture_coords_aligned = None
    ficture_labels = None

    if args.ficture_path and os.path.exists(args.ficture_path):
        print(f"\n[3/4] FICTURE evaluation ...")
        ficture_coords, ficture_labels = load_ficture_ground_truth(
            args.ficture_path)

        # If ROI mode, crop FICTURE pixels to the same ROI
        if args.roi:
            x0, y0, w, h = args.roi
            roi_mask = ((ficture_coords[:, 0] >= x0) &
                        (ficture_coords[:, 0] < x0 + w) &
                        (ficture_coords[:, 1] >= y0) &
                        (ficture_coords[:, 1] < y0 + h))
            ficture_coords = ficture_coords[roi_mask]
            ficture_labels = ficture_labels[roi_mask]
            print(f"  FICTURE pixels in ROI: {len(ficture_labels):,}")

        ari, nmi, ficture_predicted, ficture_coords_aligned = \
            evaluate_at_ficture_level(grid_coords, grid_labels,
                                      ficture_coords, ficture_labels)
    else:
        print(f"\n[3/4] No FICTURE output — skipping evaluation")

    print(f"\n{'=' * 60}")
    print(f"RESULT:  ARI = {ari:.3f},  NMI = {nmi:.3f}")
    if result.get("n_reassigned", 0) > 0:
        print(f"         Cleanup: {result['n_reassigned']} fragments reassigned")
    print(f"{'=' * 60}")

    # --- Plot ---
    if not args.no_plot and ficture_labels is not None:
        print(f"\n[4/4] Plotting ...")
        plot_results(
            grid_coords, grid_labels,
            ficture_coords_aligned, ficture_labels, ficture_predicted,
            result["boundary_raw"], result["boundary_smooth"],
            ari, nmi,
            output_dir=args.output_dir,
            mode=mode,
        )

    # --- Boundary-only plot (useful even without FICTURE) ---
    if not args.no_plot and ficture_labels is None:
        import matplotlib.pyplot as plt
        os.makedirs(args.output_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle(
            f"Sheaf Boundary Detection — Stereo-seq E16.5 ({mode})",
            fontsize=14, fontweight="bold")
        ax = axes[0]
        ax.scatter(grid_coords[:, 0], grid_coords[:, 1],
                   s=2, c=grid_labels, cmap="tab20", alpha=0.8)
        ax.set_title(f"Clusters ({len(set(grid_labels))})")
        ax.set_aspect("equal")
        ax = axes[1]
        sc = ax.scatter(grid_coords[:, 0], grid_coords[:, 1],
                        s=2, c=result["boundary_raw"], cmap="hot", alpha=0.8)
        plt.colorbar(sc, ax=ax)
        ax.set_title("Raw boundary")
        ax.set_aspect("equal")
        ax = axes[2]
        sc = ax.scatter(grid_coords[:, 0], grid_coords[:, 1],
                        s=2, c=result["boundary_smooth"], cmap="hot", alpha=0.8)
        plt.colorbar(sc, ax=ax)
        ax.set_title("Smoothed boundary")
        ax.set_aspect("equal")
        for ax in axes:
            ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.tight_layout()
        suffix = "roi" if mode == "roi" else "coarse"
        path = os.path.join(args.output_dir,
                            f"sheaf_boundary_stereoseq_{suffix}.png")
        fig.savefig(path, dpi=400, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    return result, ari, nmi


def sweep(args):
    print("=" * 60)
    print("Parameter sweep — Stereo-seq")
    print("=" * 60)

    mol = load_stereoseq_filtered(
        args.mol_path,
        top_n=args.top_genes,
        roi=tuple(args.roi) if args.roi else None,
        subsample=args.subsample,
    )

    ficture_coords, ficture_labels = None, None
    if args.ficture_path and os.path.exists(args.ficture_path):
        ficture_coords, ficture_labels = load_ficture_ground_truth(
            args.ficture_path)
        if args.roi:
            x0, y0, w, h = args.roi
            roi_mask = ((ficture_coords[:, 0] >= x0) &
                        (ficture_coords[:, 0] < x0 + w) &
                        (ficture_coords[:, 1] >= y0) &
                        (ficture_coords[:, 1] < y0 + h))
            ficture_coords = ficture_coords[roi_mask]
            ficture_labels = ficture_labels[roi_mask]

    if args.roi:
        # ROI sweep: fine-resolution parameters
        configs = []
        for gs in [20, 30, 50]:
            for bw in [30, 50, 80]:
                for sr in [60, 120, 200]:
                    for ud in [6, 9]:
                        for sw in [0.3, 0.5]:
                            configs.append(dict(
                                grid_spacing=gs, bandwidth=bw,
                                smooth_radius=sr, uniform_dim=ud,
                                spatial_weight=sw, boundary_alpha=0.5))
    else:
        # Coarse sweep: organ-level parameters
        configs = []
        for gs in [200, 400, 600]:
            for bw in [400, 600, 800]:
                for sr in [200, 400, 600]:
                    for ud in [6, 9]:
                        for sw in [0.2, 0.3, 0.5]:
                            configs.append(dict(
                                grid_spacing=gs, bandwidth=bw,
                                smooth_radius=sr, uniform_dim=ud,
                                spatial_weight=sw, boundary_alpha=0.5))

    print(f"\n{len(configs)} configurations\n")
    hdr = (f"{'grid':>5} {'bw':>5} {'σ_s':>5} {'d':>3} "
           f"{'α_s':>5} | {'ARI':>7} {'NMI':>7}")
    print(hdr)
    print("-" * len(hdr))

    best_ari, best_cfg = -1, None
    rows = []

    for cfg in configs:
        try:
            res = pipeline.run_single_slice(
                mol=mol, slice_name=SLICE_NAME,
                n_clusters=args.n_clusters,
                k_graph=args.k_graph, k_nbrs=args.k_nbrs,
                overcluster_factor=args.overcluster_factor,
                cleanup=True, verbose=False,
                **cfg,
            )
            if ficture_coords is not None:
                ari, nmi, _, _ = evaluate_at_ficture_level(
                    res["grid_coords"], res["labels"],
                    ficture_coords, ficture_labels)
            else:
                ari = nmi = float("nan")
        except Exception as e:
            print(f"  ERROR: {e}")
            ari = nmi = float("nan")

        print(f"{cfg['grid_spacing']:5.0f} {cfg['bandwidth']:5.0f} "
              f"{cfg['smooth_radius']:5.0f} {cfg['uniform_dim']:3d} "
              f"{cfg['spatial_weight']:5.1f} | {ari:7.4f} {nmi:7.4f}")

        rows.append({**cfg, "ari": ari, "nmi": nmi})
        if ari > best_ari:
            best_ari, best_nmi, best_cfg = ari, nmi, cfg.copy()

    print(f"\n{'='*60}")
    if best_cfg:
        print(f"BEST:  ARI = {best_ari:.4f},  NMI = {best_nmi:.4f}")
        for k, v in best_cfg.items():
            if k in ("grid_spacing", "bandwidth", "smooth_radius"):
                print(f"  {k:18s} = {v}  ({_um(v):.0f} μm)")
            else:
                print(f"  {k:18s} = {v}")

    os.makedirs(args.output_dir, exist_ok=True)
    csv = os.path.join(args.output_dir, "stereoseq_sweep.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\nSaved {csv}")


# ===================================================================
# CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Sheaf boundary detection on Stereo-seq E16.5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Coarse whole embryo (Strategy 2 — default, 10%% subsample)
  python run_seqscope.py

  # Coarse with no subsampling (if you have enough RAM)
  python run_seqscope.py --subsample 1.0

  # ROI mode (Strategy 1): 2x2 mm crop, full molecules
  python run_seqscope.py --roi 6000 8000 27000 16000 \\
      --grid-spacing 30 --bandwidth 50 --smooth-radius 120 \\
      --n-clusters 12 --k-nbrs 50 --top-genes 500 --subsample 1.0
""",
    )
    p.add_argument("--mol-path",
                    default="E16.5_E1S3_GEM_bin1.tsv.gz")
    p.add_argument("--ficture-path",
                    default=None)#"DataFicture/StereoSeq_E16.5_E1S3.nF48.d_15.decode.r_5_10.pixel.sorted.tsv.gz")

    # --- Spatial parameters (native units, 1 unit ≈ 0.5 μm) ---
    #
    # COARSE defaults (whole embryo, ~3k grid points):
    #   grid=400 (200μm), bw=600 (300μm), smooth=400 (200μm)
    #
    # ROI defaults (use with --roi):
    #   grid=30 (15μm), bw=50 (25μm), smooth=120 (60μm)
    #
    p.add_argument("--grid-spacing", type=float, default=30,
                    help="Grid spacing (default 400 = 200 μm)")
    p.add_argument("--bandwidth", type=float, default=50,
                    help="KDE bandwidth (default 600 = 300 μm)")
    p.add_argument("--smooth-radius", type=float, default=120,
                    help="Boundary smoothing σ (default 400 = 200 μm)")

    # --- Sheaf parameters ---
    p.add_argument("--k-graph", type=int, default=6)
    p.add_argument("--k-nbrs", type=int, default=30,
                    help="Local PCA neighbors (30 for coarse, 50 for ROI)")
    p.add_argument("--uniform-dim", type=int, default=6)
    p.add_argument("--boundary-alpha", type=float, default=0.5)
    p.add_argument("--spatial-weight", type=float, default=0.3,
                    help="Spatial weight (0.3 for coarse, 0.5 for ROI)")
    p.add_argument("--n-clusters", type=int, default=9,
                    help="Clusters (20 for coarse tissue-level, "
                         "10-15 for ROI)")
    p.add_argument("--overcluster-factor", type=float, default=1.0)

    # --- Gene filtering ---
    p.add_argument("--top-genes", type=int, default=300,
                    help="Keep top N genes by total count (default 300)")

    # --- Subsampling ---
    p.add_argument("--subsample", type=float, default=0.1,
                    help="Subsample fraction of molecules after filtering "
                         "(default 0.1 = 10%%).  Set to 1.0 for no "
                         "subsampling.  At coarse resolution (≥200 μm)")

    # --- ROI cropping ---
    p.add_argument("--roi", type=float, nargs=4, default=[6000,8000 , 27000, 16000],#None,
                    metavar=("X0", "Y0", "WIDTH", "HEIGHT"),
                    help="Crop to ROI in native units. "
                         "E.g. --roi 16000 8000  27000 16000 ")

    # --- Flags ---
    p.add_argument("--no-cleanup", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--output-dir", default="output_Stereoseq")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.sweep:
        sweep(args)
    else:
        run(args)