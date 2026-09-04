#!/usr/bin/env python
"""
Sheaf spatial-domain inference on 10x Atera (WTA FFPE) whole-transcriptome
in-situ data.

Atera is molecule-level: the Output Bundle's ``transcripts.parquet`` is a cloud
of individual RNA detections ``(x, y, gene)`` — a direct fit for the core
molecule pipeline (same contract as MERFISH / Stereo-seq). This runner loads the
parquet, applies QV filtering and control-probe removal, keeps the top-N genes
(whole-transcriptome ≈ 18k genes → gene filtering for tractability, as for
Stereo-seq), builds the 2D fiber field, and runs boundary detection +
boundary-aware clustering.

No ground-truth annotation exists for these datasets, so we report unsupervised
metrics and — for the biological discussion — the **top marker genes per
inferred domain** (each domain's most enriched genes vs the global average),
which is what supports reading domains as tumour / stroma / immune / etc.

Schema note: column names are the 10x Xenium/in-situ convention by default
(``feature_name``, ``x_location``, ``y_location``, ``qv``). CONFIRM with
``--inspect`` first and override with --gene-col/--x-col/--y-col/--qv-col.

Usage
-----
    python run_atera.py --input transcripts.parquet --inspect

    python run_atera.py --input transcripts.parquet \
        --grid-spacing 40 --bandwidth 80 --n-clusters 12 \
        --top-genes 1000 --output-dir output_atera_breast
"""
import argparse
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)

from spsheaf import fibers, graph, sheaf, clustering

CONTROL_RE = re.compile(
    r"(?:neg[_ ]?control|blank|antisense|unassigned|deprecated|intergenic|"
    r"codeword|^control)", re.I)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="transcripts.parquet")
    p.add_argument("--output-dir", default="output_atera")
    p.add_argument("--inspect", action="store_true")
    # schema (10x convention defaults)
    p.add_argument("--gene-col", default="feature_name")
    p.add_argument("--x-col", default="x_location")
    p.add_argument("--y-col", default="y_location")
    p.add_argument("--qv-col", default="qv")
    p.add_argument("--min-qv", type=float, default=20.0,
                   help="Drop detections below this quality value (10x default 20).")
    p.add_argument("--keep-controls", action="store_true",
                   help="Do NOT drop negative-control / blank / codeword probes.")
    # scale handling
    p.add_argument("--top-genes", type=int, default=1000,
                   help="Keep top-N genes by count (0 = all). Stalk memory ~ "
                        "N_grid x N_genes x d, so keep this bounded.")
    p.add_argument("--subsample", type=float, default=1.0)
    p.add_argument("--chunk-rows", type=int, default=5_000_000,
                   help="Parquet streaming batch size (rows). Lower if memory "
                        "is tight on a very large transcripts file.")
    p.add_argument("--seed", type=int, default=0)
    # spatial / kernel (domain scale; tune to the section)
    p.add_argument("--grid-spacing", type=float, default=40.0)
    p.add_argument("--bandwidth", type=float, default=80.0)
    p.add_argument("--smooth-radius", type=float, default=150.0)
    p.add_argument("--min-molecules", type=float, default=15.0)
    p.add_argument("--block-size", type=int, default=1500)
    p.add_argument("--n-clusters", type=int, default=12)
    p.add_argument("--k-graph", type=int, default=6)
    p.add_argument("--k-nbrs", type=int, default=50)
    p.add_argument("--uniform-dim", type=int, default=6)
    p.add_argument("--spatial-weight", type=float, default=0.5)
    p.add_argument("--boundary-alpha", type=float, default=0.5)
    p.add_argument("--n-markers", type=int, default=12)
    return p


def load_atera(a, verbose=True):
    """Stream the transcripts parquet by row-group, keeping only the compact
    (gene, x, y) columns after QV filtering + control-probe removal, so peak
    memory stays bounded even on a hundreds-of-millions-of-rows file."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(a.input)
    schema_names = pf.schema.names
    for req in (a.gene_col, a.x_col, a.y_col):
        if req not in schema_names:
            raise SystemExit(f"Column '{req}' not in parquet {schema_names}. "
                             "Run --inspect and set --gene-col/--x-col/--y-col.")
    read_cols = [a.gene_col, a.x_col, a.y_col]
    has_qv = a.qv_col in schema_names
    if has_qv:
        read_cols.append(a.qv_col)
    # Xenium/Atera ship an explicit is_gene flag — cleaner than a name regex.
    has_isgene = "is_gene" in schema_names
    if has_isgene:
        read_cols.append("is_gene")

    parts, n0, n_kept = [], 0, 0
    bi = 0
    for batch in pf.iter_batches(batch_size=a.chunk_rows, columns=read_cols):
        ch = batch.to_pandas()
        n0 += len(ch)
        if has_qv and a.min_qv is not None:
            ch = ch[ch[a.qv_col] >= a.min_qv]
        if not a.keep_controls:
            if has_isgene:
                ch = ch[ch["is_gene"].astype(bool)]      # keep real genes only
            else:
                ch = ch[~ch[a.gene_col].astype(str).str.contains(CONTROL_RE)]
        # Subsample PER BATCH so the concatenated set never blows up on a
        # hundreds-of-millions-of-rows file (Atera is very dense).
        if a.subsample and 0 < a.subsample < 1.0:
            ch = ch.sample(frac=a.subsample, random_state=a.seed + bi)
        bi += 1
        n_kept += len(ch)
        parts.append(pd.DataFrame({
            "gene": ch[a.gene_col].astype(str).values,
            "x": ch[a.x_col].astype("float32").values,
            "y": ch[a.y_col].astype("float32").values}))
    mol = pd.concat(parts, ignore_index=True)
    del parts
    if verbose:
        ss = f", subsample {a.subsample:g}" if 0 < a.subsample < 1.0 else ""
        print(f"  Transcripts : {n0:,} -> {n_kept:,} after QV>={a.min_qv} "
              f"+ control removal{ss}")
        print(f"  Genes       : {mol['gene'].nunique()}")
    return mol


def prepare(mol, a, verbose=True):
    if a.top_genes and mol["gene"].nunique() > a.top_genes:
        keep = mol["gene"].value_counts().nlargest(a.top_genes).index
        mol = mol[mol["gene"].isin(keep)].copy()
        if verbose:
            print(f"  Gene filter : top {a.top_genes} genes -> {len(mol):,} molecules")
    # (subsampling already applied per-batch during streaming load)
    mol["x_local"] = mol["x"] - mol["x"].min()
    mol["y_local"] = mol["y"] - mol["y"].min()
    if verbose:
        print(f"  Extent      : {mol['x_local'].max():.0f} x "
              f"{mol['y_local'].max():.0f} um")
    return mol


def build_fibers_batched(mol, a, verbose=True):
    """Memory-safe 2D kernel fibers (block-processed radius query)."""
    genes = sorted(mol["gene"].unique())
    g2i = {g: i for i, g in enumerate(genes)}
    coords = mol[["x_local", "y_local"]].values.astype(np.float64)
    gidx = mol["gene"].map(g2i).values
    gs = a.grid_spacing
    xg = np.arange(coords[:, 0].min() + gs / 2, coords[:, 0].max(), gs)
    yg = np.arange(coords[:, 1].min() + gs / 2, coords[:, 1].max(), gs)
    xx, yy = np.meshgrid(xg, yg)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    tree = cKDTree(coords)
    radius = 3.0 * a.bandwidth
    fib = np.zeros((len(grid), len(genes)), dtype=np.float32)
    tw = np.zeros(len(grid), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(grid), a.block_size):
        e = min(s + a.block_size, len(grid))
        nbrs = tree.query_ball_point(grid[s:e], r=radius)
        for li, nb in enumerate(nbrs):
            if not nb:
                continue
            gi = s + li
            nb = np.asarray(nb)
            d = np.linalg.norm(coords[nb] - grid[gi], axis=1)
            w = np.exp(-0.5 * (d / a.bandwidth) ** 2)
            np.add.at(fib[gi], gidx[nb], w)
            tw[gi] = w.sum()
        del nbrs
    keep = tw >= a.min_molecules
    if verbose:
        print(f"  Grid        : {keep.sum():,} valid points, {len(genes)} genes "
              f"({time.time()-t0:.0f}s)")
    return grid[keep], fibers.normalize_fibers(fib[keep]), genes


def marker_genes(fibers_norm, labels, genes, n_markers):
    """Top enriched genes per domain: domain mean - global mean (log space)."""
    global_mean = fibers_norm.mean(0)
    rows = []
    for c in sorted(set(labels)):
        dm = fibers_norm[labels == c].mean(0)
        enr = dm - global_mean
        top = np.argsort(enr)[::-1][:n_markers]
        for rank, gi in enumerate(top):
            rows.append(dict(domain=int(c), rank=rank + 1, gene=genes[gi],
                             enrichment=round(float(enr[gi]), 4)))
    return pd.DataFrame(rows)


def unsup_metrics(fibers_norm, labels, edges, bsm, seed=0, max_pts=6000):
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(labels), max_pts, replace=False)
           if len(labels) > max_pts else np.arange(len(labels)))
    m = {}
    if len(set(labels[idx])) > 1:
        m["silhouette"] = round(float(silhouette_score(fibers_norm[idx], labels[idx])), 4)
        m["calinski_harabasz"] = round(float(calinski_harabasz_score(fibers_norm[idx], labels[idx])), 1)
        m["davies_bouldin"] = round(float(davies_bouldin_score(fibers_norm[idx], labels[idx])), 4)
    e = np.asarray(edges)
    same = labels[e[:, 0]] == labels[e[:, 1]]
    m["spatial_coherence"] = round(float(same.mean()), 4)
    m["n_domains"] = int(len(set(labels)))
    m["n_grid_points"] = int(len(labels))
    return m


def run(mol, a):
    print("\n[1/6] Kernel fibers"); grid, fib, genes = build_fibers_batched(mol, a)
    print("[2/6] Spatial graph"); edges = graph.build_spatial_graph(grid, k=a.k_graph, verbose=True)
    print("[3/6] Stalks"); stalks = sheaf.compute_stalks(fib, grid, uniform_dim=a.uniform_dim, k_nbrs=a.k_nbrs, verbose=True)
    print("[4/6] Restriction maps"); restr = sheaf.compute_restriction_maps(stalks, edges, a.uniform_dim, verbose=True)
    print("[5/6] Boundary scores")
    braw, bsm = sheaf.compute_boundary_scores(fib, grid, stalks, edges, restr, smooth_radius=a.smooth_radius, verbose=True)
    print("[6/6] Boundary-aware clustering")
    labels = clustering.boundary_aware_clustering(fib, grid, bsm, edges, n_clusters=a.n_clusters,
                                                  spatial_weight=a.spatial_weight, boundary_alpha=a.boundary_alpha, verbose=True)
    labels, _ = clustering.spatial_cleanup(labels, edges, grid_coords=grid, verbose=True)
    return dict(grid=grid, fib=fib, braw=braw, bsm=bsm, labels=np.asarray(labels),
                edges=edges, genes=genes)


def save_outputs(r, a):
    os.makedirs(a.output_dir, exist_ok=True)
    g = r["grid"]
    pd.DataFrame({"x": g[:, 0], "y": g[:, 1], "domain": r["labels"]}
                 ).to_csv(os.path.join(a.output_dir, "domain_labels.csv"), index=False)
    pd.DataFrame({"x": g[:, 0], "y": g[:, 1], "boundary_raw": r["braw"],
                  "boundary_smooth": r["bsm"]}
                 ).to_csv(os.path.join(a.output_dir, "boundary_field.csv"), index=False)
    mk = marker_genes(r["fib"], r["labels"], r["genes"], a.n_markers)
    mk.to_csv(os.path.join(a.output_dir, "domain_markers.csv"), index=False)
    met = unsup_metrics(r["fib"], r["labels"], r["edges"], r["bsm"], seed=a.seed)
    pd.DataFrame([met]).to_csv(os.path.join(a.output_dir, "metrics.csv"), index=False)
    _plot(r, mk, a.output_dir)
    print("\nUnsupervised metrics:", met)
    print("\nTop marker genes per domain:")
    for c in sorted(set(r["labels"])):
        top = mk[mk.domain == c].head(6)["gene"].tolist()
        print(f"  domain {c}: {', '.join(top)}")


def _plot(r, mk, out_dir):
    g, lab, bsm = r["grid"], r["labels"], r["bsm"]
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    uniq = sorted(set(lab)); cmap = plt.cm.get_cmap("tab20", max(len(uniq), 2))
    for i, c in enumerate(uniq):
        m = lab == c
        lbl = f"{c}: " + ", ".join(mk[mk.domain == c].head(3)["gene"].tolist())
        ax[0].scatter(g[m, 0], g[m, 1], s=3, color=cmap(i), label=lbl)
    ax[0].set_aspect("equal"); ax[0].invert_yaxis()
    ax[0].set_title(f"Sheaf domains (k={len(uniq)})")
    ax[0].legend(markerscale=3, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    sc = ax[1].scatter(g[:, 0], g[:, 1], c=bsm, s=3, cmap="hot",
                       vmax=np.percentile(bsm, 95))
    ax[1].set_aspect("equal"); ax[1].invert_yaxis()
    ax[1].set_title("Sheaf boundary score"); plt.colorbar(sc, ax=ax[1], shrink=0.7)
    fig.suptitle("10x Atera — sheaf spatial domains", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "domain_map.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def inspect(a):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(a.input)
    print("rows:", pf.metadata.num_rows)
    print("columns:", pf.schema.names)
    head = pd.read_parquet(a.input, columns=pf.schema.names[:min(12, len(pf.schema.names))]).head(6)
    print("\nhead:\n", head.to_string())
    if a.gene_col in pf.schema.names:
        genes = pd.read_parquet(a.input, columns=[a.gene_col])[a.gene_col].astype(str)
        ctrl = genes.str.contains(CONTROL_RE).mean()
        print(f"\n{a.gene_col}: {genes.nunique()} unique, "
              f"{100*ctrl:.1f}% control/blank probes")


def main():
    a = build_parser().parse_args()
    if a.inspect:
        inspect(a); return
    print("=" * 60); print(" 10x ATERA — sheaf spatial domains"); print("=" * 60)
    t0 = time.time()
    mol = load_atera(a); mol = prepare(mol, a)
    r = run(mol, a); del mol
    save_outputs(r, a)
    print(f"\nDone in {time.time()-t0:.0f}s -> {a.output_dir}/")


if __name__ == "__main__":
    main()
