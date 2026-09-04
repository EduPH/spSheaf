#!/usr/bin/env python
"""
Stereo-seq E16.5 whole-embryo sheaf domains with the non-tissue background
removed by a data-level tissue mask.

Segmentation-free: runs on the raw GEM molecule table. Memory-safe: two
streaming passes over the (multi-GB) gzip, subsampling per chunk so the full
filtered table is never held in RAM. A density-based tissue mask
(Otsu threshold on log grid-density + largest connected component) drops the
ambient background before clustering, so the embryo is clustered and plotted
on its own.

Usage
-----
    python run_stereoseq_masked.py \
        --gem E16.5_E1S3_GEM_bin1.tsv.gz \
        --out spsheafs_embryo_masked.png

    # tweak resolution / sampling if desired
    python run_stereoseq_masked.py --gem <path> \
        --roi 6000 8000 27000 16000 --grid 30 --bandwidth 50 \
        --smooth 120 --n-clusters 8 --subsample 0.1 --top-genes 300

Requires: numpy, pandas, scipy, scikit-learn, matplotlib, and the
``spsheaf`` package (``pip install -e package`` from the repo root).
"""
import argparse, os, sys, time, gzip
from collections import defaultdict
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
from spsheaf import fibers, graph, sheaf, clustering


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def header_skip(path):
    op = gzip.open if path.endswith(".gz") else open
    n = 0
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("#"): n += 1
            else: break
    return n


def stream_load(gem, roi, top_genes, subsample, chunk, seed):
    """Two memory-safe streaming passes: (1) top-gene selection, (2) load."""
    x0, y0, w, h = roi
    x1, y1 = x0 + w, y0 + h
    skip = header_skip(gem)
    rng = np.random.default_rng(seed)

    log("pass 1/2: gene counts over ROI ...")
    gene_count = defaultdict(np.int64); n = 0; t0 = time.time()
    for ci, ch in enumerate(pd.read_csv(gem, sep="\t", skiprows=skip,
                                        chunksize=chunk,
                                        usecols=["geneID", "x", "y", "MIDCount"],
                                        dtype={"geneID": str})):
        n += len(ch)
        gx = ch["x"].to_numpy(); gy = ch["y"].to_numpy()
        m = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)
        if m.any():
            for g, c in ch[m].groupby("geneID")["MIDCount"].sum().items():
                gene_count[g] += int(c)
        if ci % 20 == 0: log(f"  p1 {n:,} rows, {time.time()-t0:.0f}s")
    top = sorted(gene_count, key=gene_count.get, reverse=True)[:top_genes]
    dmap = {g: i for i, g in enumerate(top)}
    code_gene = np.array(top)
    log(f"pass1: {n:,} rows scanned; kept top {len(top)} genes")

    log("pass 2/2: loading subsampled top-gene molecules ...")
    xs, ys, cs = [], [], []; n = 0; t0 = time.time()
    for ci, ch in enumerate(pd.read_csv(gem, sep="\t", skiprows=skip,
                                        chunksize=chunk,
                                        usecols=["geneID", "x", "y"],
                                        dtype={"geneID": str})):
        n += len(ch)
        gx = ch["x"].to_numpy(); gy = ch["y"].to_numpy()
        m = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)
        if not m.any(): continue
        sub = ch[m]
        code = sub["geneID"].map(dmap).to_numpy()
        keep = (~pd.isna(code)) & (rng.random(len(sub)) < subsample)
        if keep.any():
            xs.append(sub["x"].to_numpy()[keep].astype(np.int32))
            ys.append(sub["y"].to_numpy()[keep].astype(np.int32))
            cs.append(code[keep].astype(np.int32))
        if ci % 20 == 0:
            log(f"  p2 {n:,} rows, kept {sum(len(a) for a in xs):,}, "
                f"{time.time()-t0:.0f}s")
    X = np.concatenate(xs); Y = np.concatenate(ys); C = np.concatenate(cs)
    mol = pd.DataFrame({"gene": code_gene[C],
                        "x": X.astype(float), "y": Y.astype(float)})
    log(f"loaded {len(mol):,} molecules over top-{top_genes} genes")
    return mol


def otsu(values, bins=256):
    lo, hi = values.min(), values.max()
    hist, edg = np.histogram(values, bins=bins, range=(lo, hi))
    p = hist / hist.sum(); omega = np.cumsum(p)
    centers = (edg[:-1] + edg[1:]) / 2
    mu = np.cumsum(p * centers); mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    return centers[np.nanargmax(sb)]


def largest_cc(n, edges, mask):
    adj = defaultdict(list)
    for u, v in edges:
        if mask[u] and mask[v]:
            adj[u].append(v); adj[v].append(u)
    seen = np.zeros(n, bool); best = []
    for s in np.where(mask)[0]:
        if seen[s]: continue
        comp = []; st = [s]; seen[s] = True
        while st:
            u = st.pop(); comp.append(u)
            for v in adj[u]:
                if not seen[v]: seen[v] = True; st.append(v)
        if len(comp) > len(best): best = comp
    out = np.zeros(n, bool); out[best] = True
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gem", default="E16.5_E1S3_GEM_bin1.tsv.gz")
    ap.add_argument("--out", default="spsheafs_embryo_masked.png")
    ap.add_argument("--roi", type=float, nargs=4, default=[6000, 8000, 27000, 16000],
                    help="x0 y0 width height (native units); default = whole embryo")
    ap.add_argument("--grid", type=float, default=30.0)
    ap.add_argument("--bandwidth", type=float, default=50.0)
    ap.add_argument("--smooth", type=float, default=120.0)
    ap.add_argument("--n-clusters", type=int, default=8)
    ap.add_argument("--k-nbrs", type=int, default=30)
    ap.add_argument("--k-graph", type=int, default=6)
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--spatial-weight", type=float, default=0.3)
    ap.add_argument("--boundary-alpha", type=float, default=0.5)
    ap.add_argument("--subsample", type=float, default=0.1)
    ap.add_argument("--top-genes", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--um-per-unit", type=float, default=0.5)
    ap.add_argument("--cache", default=None,
                    help="npz path to cache/reuse built fibers "
                         "(default: <out>_fibers.npz). Delete it to force a rebuild.")
    a = ap.parse_args()

    # fibers are the expensive step (~15 min); cache them so reruns are instant
    cache = a.cache or (os.path.splitext(a.out)[0] + "_fibers.npz")
    if os.path.exists(cache):
        log(f"loading cached fibers <- {cache}")
        z = np.load(cache, allow_pickle=True)
        grid, raw, genes = z["grid"], z["raw"], z["genes"]
    else:
        mol = stream_load(a.gem, tuple(a.roi), a.top_genes, a.subsample, a.chunk, a.seed)
        log("building fibers ...")
        grid, raw, genes = fibers.build_kernel_fibers(
            mol, grid_spacing=a.grid, bandwidth=a.bandwidth, min_molecules=15,
            x_col="x", y_col="y", verbose=False)
        np.savez(cache, grid=grid, raw=raw, genes=np.asarray(genes))
        log(f"cached fibers -> {cache}")
    dens = raw.sum(1)
    thr = np.expm1(otsu(np.log1p(dens)))
    mask = dens >= thr
    log(f"{len(grid):,} grid pts; Otsu density thr={thr:.0f} -> {mask.sum():,} tissue pts")

    edges_all = graph.build_spatial_graph(grid, k=a.k_graph, verbose=False)
    tissue = largest_cc(len(grid), edges_all, mask)
    log(f"largest tissue component: {tissue.sum():,} pts")

    gc = grid[tissue]; rf = raw[tissue]
    fn = fibers.normalize_fibers(rf)
    edges = graph.build_spatial_graph(gc, k=a.k_graph, verbose=False)
    stalks = sheaf.compute_stalks(fn, gc, uniform_dim=a.dim, k_nbrs=a.k_nbrs, verbose=False)
    restr = sheaf.compute_restriction_maps(stalks, edges, a.dim, verbose=False)
    b_raw, b_smooth = sheaf.compute_boundary_scores(
        fn, gc, stalks, edges, restr, smooth_radius=a.smooth, verbose=False)
    labels = clustering.boundary_aware_clustering(
        fn, gc, b_smooth, edges, n_clusters=a.n_clusters,
        spatial_weight=a.spatial_weight, boundary_alpha=a.boundary_alpha,
        verbose=False)
    labels = clustering.spatial_cleanup(labels, edges, grid_coords=gc, verbose=False)
    log(f"clusters: {len(set(labels))}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(1, 3, figsize=(24, 7))
    s = 6
    ax[0].scatter(gc[:, 0], gc[:, 1], c=labels, cmap="tab20", s=s, linewidths=0)
    ax[0].set_title(f"Domains ({len(set(labels))})", fontsize=15)
    r = ax[1].scatter(gc[:, 0], gc[:, 1], c=b_raw, cmap="inferno", s=s, linewidths=0)
    ax[1].set_title("Raw boundary field", fontsize=15); fig.colorbar(r, ax=ax[1], shrink=.8)
    sm = ax[2].scatter(gc[:, 0], gc[:, 1], c=b_smooth, cmap="inferno", s=s, linewidths=0)
    ax[2].set_title("Smoothed boundary field", fontsize=15); fig.colorbar(sm, ax=ax[2], shrink=.8)
    span = gc[:, 0].max() - gc[:, 0].min()
    bar = 2000.0 / a.um_per_unit  # 2 mm in native units
    for a_ in ax:
        a_.set_aspect("equal"); a_.invert_yaxis()
        a_.set_xticks([]); a_.set_yticks([])
        for sp in a_.spines.values(): sp.set_visible(False)
    x0 = gc[:, 0].min() + 0.03 * span; y0 = gc[:, 1].max() - 0.05 * (gc[:, 1].max() - gc[:, 1].min())
    ax[0].add_patch(Rectangle((x0, y0), bar, 0.01 * span, color="black"))
    ax[0].text(x0 + bar / 2, y0, "2 mm", ha="center", va="bottom", fontsize=12)
    fig.suptitle("Sheaf domains — Stereo-seq E16.5 mouse embryo (background masked)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.out, dpi=400, bbox_inches="tight", facecolor="white")
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
