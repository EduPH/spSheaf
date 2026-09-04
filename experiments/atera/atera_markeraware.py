#!/usr/bin/env python
"""
Marker-aware Atera re-analysis (breast).

Root-cause fix over the original run: (1) the curated cancer-cell-type marker
genes are ALWAYS retained (the count-500 filter had dropped PGR, KRT5/14/17,
TP63, CDKN2A, ...); (2) each gene's kernel-smoothed spatial field is
z-standardized across grid points, so low-abundance but decisive markers get
EQUAL weight in the fiber (they are no longer drowned by abundant genes). The
sheaf pipeline is otherwise unchanged.

Phases (run separately to respect the shell time cap):
  --phase fibers   build grid + per-gene smoothed z-scored fiber matrix -> npz
  --phase cluster  sheaf pipeline on the fiber matrix -> domains
  --phase report   per-marker spatial maps + domain->cancer-cell-type assignment
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from spsheaf import graph, sheaf, clustering

MARKERS = ["ESR1","PGR","ERBB2","GATA3","FOXA1","SCGB2A2","KRT5","CDK1","PLIN1","COL1A2",
           "CDKN2A","KRT14","KRT17","TP63","CD68","XBP1","PTPRC","DCN","GZMB",
           "EPCAM","KRT8","KRT18","KRT19","MKI67","TOP2A","VIM","PECAM1","VWF",
           "COL1A1","COL3A1","ACTA2","POSTN","MS4A1","CD3D","CD3E","CD8A","LYZ","MUC1","TFF1","TFF3",
           "SOX2","KRT6A","SERPINB3","SERPINB4","KRT15","DSP","IVL"]

# cancer-cell-type signatures, per tissue. Stroma/immune/vascular/proliferation
# are shared; the tumour-epithelium set is tissue-specific.
_COMMON = {
    "Stroma / CAF":     ["COL1A1","COL1A2","COL3A1","DCN","POSTN","ACTA2","VIM"],
    "Immune":           ["PTPRC","CD68","CD3D","CD3E","CD8A","MS4A1","LYZ","GZMB"],
    "Vascular":         ["PECAM1","VWF"],
    "Proliferation":    ["MKI67","TOP2A","CDK1"],
}
_TUMOUR = {
    "breast": {
        "Luminal tumour": ["ESR1","PGR","GATA3","FOXA1","ERBB2","XBP1","MUC1","TFF1","TFF3","EPCAM","KRT8","KRT18","KRT19"],
        "Basal/squamous": ["KRT5","KRT14","KRT17","TP63","CDKN2A"],
    },
    "cervical": {
        "Squamous tumour": ["KRT5","KRT14","KRT17","KRT6A","KRT15","TP63","SOX2","CDKN2A","SERPINB3","SERPINB4","DSP","IVL","EPCAM"],
    },
}
_PANEL = {  # the colleague's marker panels, for the spatial-map figure
    "breast":   ["ESR1","PGR","ERBB2","GATA3","FOXA1","SCGB2A2","KRT5","CDK1","PLIN1","COL1A2"],
    "cervical": ["CDKN2A","KRT5","KRT14","KRT17","TP63","CD68","XBP1","PTPRC","DCN","GZMB"],
}


def signatures_for(tissue):
    s = dict(_TUMOUR.get(tissue, _TUMOUR["breast"])); s.update(_COMMON); return s


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--phase", required=True, choices=["fibers","cluster","report"])
    p.add_argument("--tissue", default="breast", choices=["breast","cervical"])
    p.add_argument("--grid", type=float, default=50.0, help="grid spacing (um)")
    p.add_argument("--bandwidth", type=float, default=100.0)
    p.add_argument("--top-genes", type=int, default=250, help="non-marker genes by count")
    p.add_argument("--min-density", type=float, default=0.5, help="tissue-mask threshold (smoothed total)")
    p.add_argument("--n-domains", type=int, default=12)
    p.add_argument("--random-state", type=int, default=42)
    return p


def load_mols(cache_dir):
    return pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(cache_dir,"part_*.parquet"))],
                     ignore_index=True)


def phase_fibers(a):
    mols = load_mols(a.cache_dir)
    gc = pd.read_csv(os.path.join(a.cache_dir,"gene_counts.csv"), index_col=0).iloc[:,0]
    nonmark = [g for g in gc.index if g not in set(MARKERS)][:a.top_genes]
    genes = [g for g in dict.fromkeys(MARKERS + nonmark) if g in set(gc.index)]
    x, y = mols.x_location.values, mols.y_location.values
    x0, y0 = x.min(), y.min()
    nx = int((x.max()-x0)/a.grid)+1; ny = int((y.max()-y0)/a.grid)+1
    ix = np.clip(((x-x0)/a.grid).astype(int), 0, nx-1)
    iy = np.clip(((y-y0)/a.grid).astype(int), 0, ny-1)
    sig = a.bandwidth/a.grid
    gidx = {g:i for i,g in enumerate(genes)}
    fields = np.zeros((len(genes), ny, nx), np.float32)
    gi_all = mols.feature_name.map(gidx).values
    m = ~pd.isna(gi_all)
    gi_all = gi_all[m].astype(int); ixm, iym = ix[m], iy[m]
    for gi in range(len(genes)):
        sel = gi_all == gi
        if not sel.any(): continue
        h = np.zeros((ny,nx), np.float32)
        np.add.at(h, (iym[sel], ixm[sel]), 1.0)
        fields[gi] = gaussian_filter(h, sig)
    total = fields.sum(0)
    mask = total >= a.min_density
    yy, xx = np.where(mask)
    grid = np.column_stack([x0 + xx*a.grid, y0 + yy*a.grid]).astype(float)
    F = fields[:, yy, xx].T                                   # (n_grid, n_genes)
    # per-gene z-standardization across grid points -> equal weight per gene
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    os.makedirs(a.output_dir, exist_ok=True)
    np.savez(os.path.join(a.output_dir,"fibers.npz"),
             grid=grid, fib=Z.astype(np.float32), genes=np.array(genes),
             fields=fields.astype(np.float32), mask=mask, x0=x0, y0=y0, gridspacing=a.grid)
    print(f"genes={len(genes)} (markers {sum(g in genes for g in MARKERS)}); "
          f"grid points={len(grid)}; raster {ny}x{nx}. saved fibers.npz")


def phase_cluster(a):
    d = np.load(os.path.join(a.output_dir,"fibers.npz"), allow_pickle=True)
    grid, fib = d["grid"], d["fib"].astype(float)
    ud = 6
    edges = graph.build_spatial_graph(grid, k=6, verbose=False)
    stalks = sheaf.compute_stalks(fib, grid, uniform_dim=ud, k_nbrs=50, verbose=False)
    restr = sheaf.compute_restriction_maps(stalks, edges, ud, verbose=False)
    _, bsm = sheaf.compute_boundary_scores(fib, grid, stalks, edges, restr, smooth_radius=150.0, verbose=False)
    labels = clustering.boundary_aware_clustering(fib, grid, bsm, edges, n_clusters=a.n_domains,
                                                  n_pca=min(10,fib.shape[1]), random_state=a.random_state, verbose=False)
    labels, _ = clustering.spatial_cleanup(labels, edges, grid_coords=grid, verbose=False)
    labels = np.asarray(labels)
    within = sum(labels[i]==labels[j] for i,j in edges)/max(len(edges),1)
    np.save(os.path.join(a.output_dir,"labels.npy"), labels)
    np.save(os.path.join(a.output_dir,"boundary.npy"), bsm)
    pd.DataFrame({"x":grid[:,0],"y":grid[:,1],"domain":labels}).to_csv(
        os.path.join(a.output_dir,"domain_labels.csv"), index=False)
    print(f"{len(set(labels))} domains; spatial coherence {within:.3f}")


def phase_report(a):
    d = np.load(os.path.join(a.output_dir,"fibers.npz"), allow_pickle=True)
    grid = d["grid"]; fib = d["fib"].astype(float); genes = list(d["genes"])
    fields = d["fields"]; mask = d["mask"]; x0=float(d["x0"]); y0=float(d["y0"]); gs=float(d["gridspacing"])
    labels = np.load(os.path.join(a.output_dir,"labels.npy"))
    gidx = {g:i for i,g in enumerate(genes)}
    doms = sorted(set(labels))

    # 1) domain x signature score (mean z per gene across domains, avg within signature)
    Z = fib  # already per-gene z across grid
    dom_mean = np.vstack([Z[labels==dd].mean(0) for dd in doms])          # (n_dom, n_genes)
    SIGNATURES = signatures_for(a.tissue)
    S = {}
    for name, gl in SIGNATURES.items():
        cols = [gidx[g] for g in gl if g in gidx]
        S[name] = dom_mean[:, cols].mean(1) if cols else np.zeros(len(doms))
    Sdf = pd.DataFrame(S, index=[f"domain {d}" for d in doms])
    assign = Sdf.idxmax(1); topscore = Sdf.max(1)
    margin = (Sdf.max(1) - Sdf.apply(lambda r: r.nlargest(2).iloc[-1], axis=1))
    # domains negative on every signature are low-signal/background: do not force a label
    assign = assign.where(topscore >= 0.1, other="Unassigned (low signal)")
    comp = pd.DataFrame({"domain":doms, "cell_type":assign.values,
                         "top_score":topscore.round(3).values, "margin":margin.round(3).values})
    comp.to_csv(os.path.join(a.output_dir,"domain_celltypes.csv"), index=False)
    Sdf.round(3).to_csv(os.path.join(a.output_dir,"domain_signature_scores.csv"))

    # 2) marker spatial-pattern maps (colleague's panel) = smoothed z-scored fields
    _marker_maps(fields, mask, genes, _PANEL[a.tissue], a.output_dir)
    # 3) domain map recoloured by cancer cell type
    _celltype_map(grid, labels, comp, a.output_dir, a.tissue)
    # 4) signature heatmap
    _heatmap(Sdf, comp, a.output_dir)
    print("domain -> cancer cell type:")
    print(comp.to_string(index=False))


def _marker_maps(fields, mask, genes, panel, out_dir):
    gidx = {g:i for i,g in enumerate(genes)}
    panel = [g for g in panel if g in gidx]
    n=len(panel); fig,ax=plt.subplots(2,(n+1)//2,figsize=(3*((n+1)//2),6))
    ax=np.atleast_1d(ax).ravel()
    for k,g in enumerate(panel):
        f = fields[gidx[g]].astype(float).copy()
        f[~mask]=np.nan
        z=(f-np.nanmean(f))/(np.nanstd(f)+1e-9)
        ax[k].imshow(z, origin="lower", cmap="Reds", vmin=0, vmax=3)
        ax[k].set_title(g, fontsize=10); ax[k].axis("off")
    for k in range(len(panel),len(ax)): ax[k].axis("off")
    fig.suptitle("Marker spatial patterns (kernel-smoothed, z-scored)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"marker_maps.png"),dpi=400,bbox_inches="tight"); plt.close(fig)


def _celltype_map(grid, labels, comp, out_dir, tissue="breast"):
    m = dict(zip(comp.domain, comp.cell_type))
    ct = np.array([m[l] for l in labels])
    types = sorted(set(ct)); pal={t:plt.cm.tab10(i) for i,t in enumerate(types)}
    fig,ax=plt.subplots(figsize=(11,7))
    for t in types:
        s=ct==t; ax.scatter(grid[s,0],grid[s,1],s=8,color=pal[t],label=t)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.legend(markerscale=3,fontsize=9,loc="center left",bbox_to_anchor=(1.0,0.5))
    ax.set_title(f"Atera {tissue}: sheaf domains by inferred cancer cell type")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"celltype_map.png"),dpi=400,bbox_inches="tight"); plt.close(fig)


def _heatmap(Sdf, comp, out_dir):
    fig,ax=plt.subplots(figsize=(1.4*Sdf.shape[1]+3,0.42*Sdf.shape[0]+2))
    im=ax.imshow(Sdf.values,cmap="RdBu_r",vmin=-1.5,vmax=1.5,aspect="auto")
    ax.set_xticks(range(Sdf.shape[1])); ax.set_xticklabels(Sdf.columns,rotation=30,ha="right")
    ax.set_yticks(range(Sdf.shape[0]))
    ax.set_yticklabels([f"{r} -> {c}" for r,c in zip(Sdf.index,comp.cell_type)])
    for i in range(Sdf.shape[0]):
        ax.text(int(np.argmax(Sdf.values[i])),i,"*",ha="center",va="center")
    fig.colorbar(im,ax=ax,shrink=0.7,label="signature score (z)")
    ax.set_title("Domain x cancer-cell-type signature")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir,"signature_heatmap.png"),dpi=400,bbox_inches="tight"); plt.close(fig)


def main():
    a = build_parser().parse_args()
    {"fibers":phase_fibers,"cluster":phase_cluster,"report":phase_report}[a.phase](a)


if __name__ == "__main__":
    main()
