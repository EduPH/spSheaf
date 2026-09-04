"""Are ELONGATED domains recovered worse, and is alpha_s the cause?

A previous experiment measured domain WIDTH and found it explained little
(rho = +0.44 with recall, no better than cell count). Width is absolute
thinness. It does not capture elongation: a domain can be 200 um across and
still be a long thin band, which is what a cortical layer is.

The mechanism, and why this is testable rather than descriptive. Clustering is
k-means on [w * PCA(expression) | alpha_s * xy]. k-means minimises within-
cluster variance, so it prefers compact clusters; a long thin band carries
large spatial variance however coherent its expression. The spatial term
therefore penalises exactly the shape the annotation is made of.

Two predictions, and the second is what makes this more than a correlation:

  1. recall falls as elongation rises;
  2. that relationship WEAKENS as alpha_s falls, because alpha_s is the only
     thing scaling the spatial term. At alpha_s = 0 there is no compactness
     bias left and elongation should stop mattering.

If (1) holds and (2) does not, elongation is a proxy for something else and
the k-means explanation is wrong.

    python -m experiments.osmfish.elongation
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from spsheaf import pipeline, paths
from experiments.osmfish.run_osmFISH import (
    load_osmfish_molecules, load_cell_ground_truth,
    evaluate_at_cell_level, SLICE_NAME, PX_TO_UM,
)
from experiments.osmfish.resolution import per_domain_recall
from experiments.common.width import domain_widths
from experiments.common.shape import domain_shapes

BASE = dict(grid_spacing=150, bandwidth=500, smooth_radius=300, uniform_dim=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-clusters", type=int, default=11)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75])
    a = ap.parse_args()

    root = paths.raw("osmfish")
    mol = load_osmfish_molecules(str(root / "mRNA_coords_raw_counting.hdf5"))
    cell_coords, cell_gt = load_cell_ground_truth(
        str(root / "osmfish_20260512084934.h5ad"))
    gt_str = np.asarray(cell_gt).astype(str)

    rows, recalls = [], {}
    for alpha in a.alphas:
        res = pipeline.run_single_slice(
            mol=mol, slice_name=SLICE_NAME, n_clusters=a.n_clusters,
            k_graph=6, k_nbrs=50, spatial_weight=alpha, boundary_alpha=0.5,
            cleanup=True, verbose=False, **BASE)
        m, pred, coords_aligned, _, _ = evaluate_at_cell_level(
            res["grid_coords"], res["labels"], cell_coords, cell_gt,
            verbose=False)
        recalls[alpha] = per_domain_recall(gt_str, np.asarray(pred))
        rows.append(dict(alpha_s=alpha, ari=m["ari"], nmi=m["nmi"]))
        print(f"  alpha_s={alpha:.2f}  ARI={m['ari']:.4f}  NMI={m['nmi']:.4f}")

    widths = domain_widths(coords_aligned, gt_str)
    shapes = domain_shapes(coords_aligned, gt_str, widths=widths)

    tab = []
    for d, sh in shapes.items():
        if d not in widths:
            continue
        tab.append(dict(domain=d, n=sh["n"],
                        width_um=widths[d]["width"] * PX_TO_UM,
                        elongation=sh["elongation"],
                        slenderness=sh.get("slenderness", np.nan),
                        **{f"recall_a{al:g}": recalls[al].get(d, np.nan)
                           for al in a.alphas}))
    df = pd.DataFrame(tab).sort_values("elongation", ascending=False)

    print(f"\n  {'domain':30s} {'elong':>6} {'slend':>6} {'width':>7}"
          + "".join(f"  a={al:g}" for al in a.alphas))
    print("  " + "-" * (52 + 7 * len(a.alphas)))
    for _, r in df.iterrows():
        print(f"  {r.domain:30s} {r.elongation:6.2f} {r.slenderness:6.1f} "
              f"{r.width_um:7.1f}" +
              "".join(f" {r[f'recall_a{al:g}']:6.2f}" for al in a.alphas))

    print("\n  Prediction 1: recall falls as elongation rises.")
    print("  Prediction 2: that relationship weakens as alpha_s falls.\n")
    print(f"  {'alpha_s':>8} {'rho(elong,recall)':>19} {'rho(slend,recall)':>19}"
          f" {'ARI':>7}")
    for al in a.alphas:
        col = df[f"recall_a{al:g}"]
        ok = df.dropna(subset=[f"recall_a{al:g}", "elongation"])
        re_ = spearmanr(ok.elongation, ok[f"recall_a{al:g}"]).statistic
        rs_ = spearmanr(ok.slenderness, ok[f"recall_a{al:g}"]).statistic
        ari = next(r["ari"] for r in rows if r["alpha_s"] == al)
        print(f"  {al:8.2f} {re_:19.3f} {rs_:19.3f} {ari:7.4f}")

    print("\n  Reading it: prediction 1 wants the correlations NEGATIVE.")
    print("  Prediction 2 wants them closer to zero at small alpha_s. If they")
    print("  are flat across alpha_s, the k-means compactness story is wrong")
    print("  and elongation is standing in for something else.")
    print("  With 11 domains none of this is significant on its own -- read the")
    print("  trend across alpha_s, which is a within-dataset comparison and so")
    print("  more informative than any single correlation.")

    out = paths.results("osmfish") / "elongation.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
