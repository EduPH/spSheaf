#!/usr/bin/env python
"""
PECAb carcinosarcoma, reframed for what the data actually is.

The PECAb panel is signaling-dominated (phospho + pathway proteins) and this
carcinosarcoma is a mesenchymal-dominated EMT continuum; even the paper's own
5 clusters are near-flat in protein space (max |z| ~0.2). So the honest output
is not discrete cancer cell-type domains but (i) the CONTINUOUS signaling-state
organization captured by the sheaf boundary field, and (ii) the one genuinely
discrete structure, the mesenchymal/myofibroblast compartment.

Per replicate this produces:
  domain_boundary_map.png  domains (k) + sheaf boundary field
  marker_maps.png          smoothed z-fields of lineage + signaling proteins
  emt_map.png              mesenchymal-minus-epithelial score field
  signature_heatmap.png    domain x protein (shown honestly, weak activation)
  marker_corr.csv          Pearson corr between smoothed marker fields (for
                           cross-replicate reproducibility of the EMT structure)
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score
from spsheaf import fibers, graph, sheaf, clustering

META = ["tileID","cellID","y","x","orientation","majoraxis","minoraxis","cell","rep","UMAP1","UMAP2","cluster"]
EPI = ["E-Cadherin","EpCAM"]
MES = ["N-Cadherin","VIM","a-SMA","FN1","PDPN"]
LINEAGE = EPI + MES
SIGNAL = ["pSMAD3","pERK","pAKT","pS6K"]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--grid", type=float, default=150.0)
    p.add_argument("--bandwidth", type=float, default=300.0)
    p.add_argument("--smooth-radius", type=float, default=100.0,
                   help="boundary-field smoothing; 100 = fidelity-peak sweet spot "
                        "(fidelity plateaus ~75-100; lower only adds noise)")
    p.add_argument("--n-domains", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    return p


def main():
    a = build_parser().parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    raw = pd.read_csv(a.input)
    prot = [c for c in raw.columns if c not in META]
    coords = raw[["x","y"]].to_numpy(float)
    feats = np.nan_to_num(raw[prot].to_numpy(float))

    g, fib = fibers.build_feature_fibers(coords, feats, grid_spacing=a.grid,
                                         bandwidth=a.bandwidth, standardize=True, verbose=False)
    tree = cKDTree(coords)
    keep = np.array([len(tree.query_ball_point(p, a.bandwidth)) >= 3 for p in g])
    g, fib = g[keep], fib[keep]                       # fib = per-protein z field on the grid
    pidx = {p: i for i, p in enumerate(prot)}

    edges = graph.build_spatial_graph(g, k=6, verbose=False)
    st = sheaf.compute_stalks(fib, g, uniform_dim=6, k_nbrs=40, verbose=False)
    rs = sheaf.compute_restriction_maps(st, edges, 6, verbose=False)
    _, bsm = sheaf.compute_boundary_scores(fib, g, st, edges, rs, smooth_radius=a.smooth_radius, verbose=False)
    lab = clustering.boundary_aware_clustering(fib, g, bsm, edges, n_clusters=a.n_domains,
                                               n_pca=10, random_state=a.random_state, verbose=False)
    lab, _ = clustering.spatial_cleanup(lab, edges, grid_coords=g, verbose=False)
    lab = np.asarray(lab)
    coh = sum(lab[i]==lab[j] for i,j in edges)/len(edges)

    # per-domain protein signature + identify the mesenchymal compartment
    prof = pd.DataFrame(fib, columns=prot).groupby(lab).mean()
    mes_score_dom = prof[[m for m in MES if m in pidx]].mean(1) - prof[[e for e in EPI if e in pidx]].mean(1)
    mes_dom = int(mes_score_dom.idxmax())

    # ---- outputs
    _domain_boundary(g, lab, bsm, mes_dom, a.output_dir)
    _marker_maps(g, fib, pidx, [m for m in LINEAGE+SIGNAL if m in pidx], a.output_dir)
    _emt_map(g, fib, pidx, a.output_dir)
    _heatmap(prof, mes_dom, a.output_dir)

    # marker co-localization matrix (smoothed fields) for cross-rep reproducibility
    mk = [m for m in LINEAGE if m in pidx]
    C = np.corrcoef(np.array([fib[:, pidx[m]] for m in mk]))
    pd.DataFrame(C, index=mk, columns=mk).round(3).to_csv(os.path.join(a.output_dir, "marker_corr.csv"))

    # ---- panel data for the manuscript figure
    # The grid, the labels, the boundary field and the EMT axis exist only in
    # memory during this run. The figure is drawn from them in figures/, so
    # they are saved rather than plotted here.
    from experiments.common.panel_export import export_panels
    mes = fib[:, [pidx[m] for m in MES if m in pidx]].mean(1)
    epi = fib[:, [pidx[m] for m in EPI if m in pidx]].mean(1)
    export_panels(
        a.output_dir, "pecab",
        arrays=dict(grid=g, labels=lab, boundary_smooth=bsm,
                    emt_score=mes - epi, marker_corr=C),
        meta=dict(mes_domain=mes_dom, n_domains=int(len(set(lab))),
                  coherence=round(coh, 3), markers=mk, proteins=list(prot),
                  grid_spacing=a.grid, bandwidth=a.bandwidth,
                  smooth_radius=a.smooth_radius, random_state=a.random_state),
    )

    sil = silhouette_score(fib, lab) if len(set(lab))>1 else 0
    pd.DataFrame([{"n_domains": len(set(lab)), "n_grid": len(g), "coherence": round(coh,3),
                   "silhouette": round(sil,3), "mesenchymal_domain": mes_dom,
                   "mes_score": round(float(mes_score_dom.max()),3)}]).to_csv(
        os.path.join(a.output_dir, "summary.csv"), index=False)
    print(f"{len(set(lab))} domains, coherence {coh:.3f}, silhouette {sil:.3f}; "
          f"mesenchymal compartment = domain {mes_dom} "
          f"(top: {', '.join(prof.loc[mes_dom].sort_values(ascending=False).head(4).index)})")


def _grid_scatter(ax, g, c, cmap, title, vmin=None, vmax=None):
    s = ax.scatter(g[:,0], g[:,1], s=8, c=c, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_title(title, fontsize=10); ax.axis("off")
    return s


def _domain_boundary(g, lab, bsm, mes_dom, out):
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    doms = sorted(set(lab))
    for d in doms:
        m = lab==d
        lab_txt = f"domain {d}" + ("  (mesenchymal)" if d==mes_dom else "")
        ax[0].scatter(g[m,0], g[m,1], s=8, color=plt.cm.tab10(d%10), label=lab_txt)
    ax[0].set_aspect("equal"); ax[0].invert_yaxis(); ax[0].legend(markerscale=2, fontsize=8,
        loc="center left", bbox_to_anchor=(1.0,0.5)); ax[0].set_title("Protein-state domains")
    vmax = np.percentile(bsm, 97)  # robust: clip the single hotspot so mid-range structure shows
    sc = _grid_scatter(ax[1], g, np.clip(bsm, 0, vmax), "magma", "Sheaf boundary field", vmin=0, vmax=vmax)
    fig.colorbar(sc, ax=ax[1], shrink=0.7)
    fig.tight_layout(); fig.savefig(os.path.join(out,"domain_boundary_map.png"), dpi=400, bbox_inches="tight"); plt.close(fig)


def _marker_maps(g, fib, pidx, markers, out):
    n=len(markers); ncol=5; nrow=(n+ncol-1)//ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(3*ncol, 3*nrow)); ax=np.atleast_1d(ax).ravel()
    for k,m in enumerate(markers):
        _grid_scatter(ax[k], g, fib[:, pidx[m]], "Reds", m, vmin=-1, vmax=2.5)
    for k in range(n,len(ax)): ax[k].axis("off")
    fig.suptitle("Lineage and signaling protein fields (kernel-smoothed z)")
    fig.tight_layout(); fig.savefig(os.path.join(out,"marker_maps.png"), dpi=400, bbox_inches="tight"); plt.close(fig)


def _emt_map(g, fib, pidx, out):
    mes = np.mean([fib[:,pidx[m]] for m in MES if m in pidx], axis=0)
    epi = np.mean([fib[:,pidx[e]] for e in EPI if e in pidx], axis=0)
    fig, ax = plt.subplots(figsize=(8,6))
    sc=_grid_scatter(ax, g, mes-epi, "RdBu_r", "Mesenchymal - epithelial score (EMT axis)", vmin=-2, vmax=2)
    fig.colorbar(sc, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(os.path.join(out,"emt_map.png"), dpi=400, bbox_inches="tight"); plt.close(fig)


def _heatmap(prof, mes_dom, out):
    order = prof.abs().max().sort_values(ascending=False).index[:22]
    fig, ax = plt.subplots(figsize=(11,3.5))
    im=ax.imshow(prof[order].values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(prof)))
    ax.set_yticklabels([f"domain {d}"+("  (mes)" if d==mes_dom else "") for d in prof.index])
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean z (smoothed field)")
    ax.set_title("Domain x protein (note: subtle contrast — signaling-dominated panel)")
    fig.tight_layout(); fig.savefig(os.path.join(out,"signature_heatmap.png"), dpi=400, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
