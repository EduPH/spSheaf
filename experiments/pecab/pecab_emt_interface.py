#!/usr/bin/env python
"""
Concordance of the sheaf output with Fujii et al.'s EMT finding.

Their key result on this carcinosarcoma: an active-EMT transition zone, driven
by signaling (pSMAD3) with elevated ZEB1, sits at the interface between the
epithelial and mesenchymal components. We test whether our segmentation-free
output localizes the same thing:

  (1) does the sheaf BOUNDARY FIELD coincide with the epithelial/mesenchymal
      interface (steep gradient of the EMT score)?
  (2) do ZEB1 (RNA) and pSMAD3 (protein) mark that interface?

Builds the protein grid + EMT score + boundary field, smooths ZEB1/FN1 from the
companion Seq-smFISH onto the same grid, and reports correlations + a 4-panel
overlay.
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from spsheaf import fibers, graph, sheaf, clustering

META = ["tileID","cellID","y","x","orientation","majoraxis","minoraxis","cell","rep","UMAP1","UMAP2","cluster"]
EPI = ["E-Cadherin","EpCAM"]; MES = ["N-Cadherin","VIM","a-SMA","FN1","PDPN"]


def smooth_to_grid(cells_xy, values, grid, bw):
    tree = cKDTree(cells_xy); out = np.zeros(len(grid))
    for i, g in enumerate(grid):
        idx = tree.query_ball_point(g, 3*bw)
        if not idx: continue
        d = np.linalg.norm(cells_xy[idx]-g, axis=1); w = np.exp(-(d**2)/(2*bw**2))
        out[i] = np.average(values[idx], weights=w) if w.sum() > 0 else 0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protein", required=True); ap.add_argument("--rna", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--grid", type=float, default=150.0); ap.add_argument("--bandwidth", type=float, default=300.0)
    a = ap.parse_args(); os.makedirs(a.output_dir, exist_ok=True)

    raw = pd.read_csv(a.protein); prot = [c for c in raw.columns if c not in META]
    coords = raw[["x","y"]].to_numpy(float); feats = np.nan_to_num(raw[prot].to_numpy(float))
    g, fib = fibers.build_feature_fibers(coords, feats, grid_spacing=a.grid, bandwidth=a.bandwidth, standardize=True, verbose=False)
    tree = cKDTree(coords); keep = np.array([len(tree.query_ball_point(p, a.bandwidth)) >= 3 for p in g])
    g, fib = g[keep], fib[keep]; pidx = {p:i for i,p in enumerate(prot)}
    edges = graph.build_spatial_graph(g, k=6, verbose=False)
    st = sheaf.compute_stalks(fib, g, uniform_dim=6, k_nbrs=40, verbose=False)
    rs = sheaf.compute_restriction_maps(st, edges, 6, verbose=False)
    _, bnd = sheaf.compute_boundary_scores(fib, g, st, edges, rs, smooth_radius=a.bandwidth*1.5, verbose=False)

    emt = np.mean([fib[:,pidx[m]] for m in MES if m in pidx],0) - np.mean([fib[:,pidx[e]] for e in EPI if e in pidx],0)
    psmad3 = fib[:, pidx["pSMAD3"]] if "pSMAD3" in pidx else np.zeros(len(g))
    # local EMT variation (interface = steep EMT gradient) from graph edges
    emt_var = np.zeros(len(g)); deg = np.zeros(len(g))
    for u,v in edges:
        d = abs(emt[u]-emt[v]); emt_var[u]+=d; emt_var[v]+=d; deg[u]+=1; deg[v]+=1
    emt_var /= np.maximum(deg,1)

    # ZEB1 / FN1 from the companion Seq-smFISH, smoothed onto the same grid
    rna = pd.read_csv(a.rna); rmeta = [c for c in ["tileID","cellID","y","x","orientation","majoraxis","minoraxis"] if c in rna.columns]
    rgenes = [c for c in rna.columns if c not in rmeta]; rxy = rna[["x","y"]].to_numpy(float)
    zeb1 = smooth_to_grid(rxy, rna["ZEB1"].to_numpy(float), g, a.bandwidth) if "ZEB1" in rgenes else np.zeros(len(g))
    zz = (zeb1-zeb1.mean())/(zeb1.std()+1e-9)

    def z(x): return (x-x.mean())/(x.std()+1e-9)
    R = {
        "boundary vs EMT-interface (|grad EMT|)": np.corrcoef(bnd, emt_var)[0,1],
        "boundary vs pSMAD3":                     np.corrcoef(bnd, psmad3)[0,1],
        "ZEB1 vs pSMAD3":                         np.corrcoef(zz, psmad3)[0,1],
        "ZEB1 vs EMT-interface":                  np.corrcoef(zz, emt_var)[0,1],
        "pSMAD3 vs EMT-interface":                np.corrcoef(psmad3, emt_var)[0,1],
    }
    print("Concordance with Fujii et al. EMT finding (Pearson r on grid points):")
    for k,v in R.items(): print(f"  {k:42s} {v:+.3f}")
    pd.DataFrame([{k:round(float(v),3) for k,v in R.items()}]).to_csv(os.path.join(a.output_dir,"emt_concordance.csv"), index=False)

    # overlay figure
    fields = [("Sheaf boundary field", bnd, "magma", None, None),
              ("EMT interface (local |grad EMT|)", emt_var, "magma", None, None),
              ("ZEB1 (RNA, smoothed z)", zz, "Reds", -1, 2.5),
              ("pSMAD3 (protein z)", psmad3, "Reds", -1, 2.5)]
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.5))
    for k,(t,c,cm,vmn,vmx) in enumerate(fields):
        s = ax[k].scatter(g[:,0], g[:,1], s=7, c=c, cmap=cm, vmin=vmn, vmax=vmx)
        ax[k].set_aspect("equal"); ax[k].invert_yaxis(); ax[k].set_title(t, fontsize=10); ax[k].axis("off")
        fig.colorbar(s, ax=ax[k], shrink=0.6)
    fig.suptitle("Boundary field, EMT interface, and its drivers ZEB1 / pSMAD3", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(a.output_dir,"emt_interface.png"), dpi=400, bbox_inches="tight"); plt.close(fig)
    print(f"-> {a.output_dir}/emt_interface.png, emt_concordance.csv")


if __name__ == "__main__":
    main()
