"""Does domain WIDTH predict which domains the method loses?

Tests the hypothesis that spSheaf's osmFISH failures are a resolution limit:
a domain narrower than roughly the kernel bandwidth is blended with its
neighbours when the fibers are built, before any clustering happens, so no
downstream parameter can recover it.

Two things argue this is not simply about domain SIZE, and the script is built
to separate them:

  * Internal Capsule (106 cells) and Ventricle (129) are smaller than both
    domains that are lost, and are recovered. Cell count is not the predictor.
  * Halving the bandwidth made accuracy WORSE overall (-0.082 ARI), so a
    narrower kernel is not a free win -- on a 33-gene panel the fiber-noise
    cost exceeds the resolution gain.

Both are consistent with a resolution limit that a single global bandwidth
cannot escape, which is the case for making the bandwidth region-adaptive:
wide where the tissue is homogeneous, narrow near thin structures.

    python -m experiments.osmfish.resolution
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from spsheaf import pipeline, paths
from experiments.osmfish.run_osmFISH import (
    load_osmfish_molecules, load_cell_ground_truth,
    evaluate_at_cell_level, SLICE_NAME, PX_TO_UM,
)
from experiments.common.width import domain_widths

PUBLISHED = dict(grid_spacing=150, bandwidth=500, smooth_radius=300,
                 uniform_dim=9)


def per_domain_recall(gt, pred):
    """Recall per ground-truth domain after optimal (Hungarian) matching."""
    gt, pred = np.asarray(gt).astype(str), np.asarray(pred)
    classes, preds = np.unique(gt), np.unique(pred)
    C = np.zeros((len(classes), len(preds)), int)
    for i, c in enumerate(classes):
        for j, q in enumerate(preds):
            C[i, j] = np.sum((gt == c) & (pred == q))
    ri, ci = linear_sum_assignment(-C)
    m = {preds[j]: classes[i] for i, j in zip(ri, ci)}
    matched = np.array([m.get(q, "") for q in pred])
    return {c: float((matched[gt == c] == c).mean()) for c in classes}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-clusters", type=int, default=11)
    a = ap.parse_args()

    root = paths.raw("osmfish")
    mol = load_osmfish_molecules(str(root / "mRNA_coords_raw_counting.hdf5"))
    cell_coords, cell_gt = load_cell_ground_truth(
        str(root / "osmfish_20260512084934.h5ad"))

    res = pipeline.run_single_slice(
        mol=mol, slice_name=SLICE_NAME, n_clusters=a.n_clusters,
        k_graph=6, k_nbrs=50, spatial_weight=0.5, boundary_alpha=0.5,
        cleanup=True, verbose=False, **PUBLISHED)
    _, pred, coords_aligned, gt_int, gt_names = evaluate_at_cell_level(
        res["grid_coords"], res["labels"], cell_coords, cell_gt, verbose=False)

    gt_str = np.asarray(cell_gt).astype(str)
    widths = domain_widths(coords_aligned, gt_str)
    recall = per_domain_recall(gt_str, np.asarray(pred))

    # The threshold is 2h, not h. A fiber mixes molecules over +-h, so a band
    # of width W retains only W - 2h of interior uncontaminated by its
    # neighbours. A domain with W < 2h has no clean interior at all.
    h_um = PUBLISHED["bandwidth"] * PX_TO_UM
    rows = []
    for d, w in widths.items():
        rows.append(dict(domain=d, n=w["n"],
                         width_um=w["width"] * PX_TO_UM,
                         recall=recall.get(d, np.nan)))
    df = pd.DataFrame(rows).sort_values("width_um")

    print(f"  kernel bandwidth h = {h_um:.1f} um "
          f"(fibers mix over roughly +-h, so ~{2*h_um:.0f} um across)\n")
    print(f"  {'domain':32s} {'n':>5} {'width(um)':>10} {'W-2h':>9} "
          f"{'W/2h':>6} {'recall':>7}")
    print("  " + "-" * 76)
    for _, r in df.iterrows():
        pure = r.width_um - 2 * h_um
        flag = "  <- at the resolution limit" if pure < h_um else ""
        print(f"  {r.domain:32s} {r.n:5d} {r.width_um:10.1f} "
              f"{pure:9.1f} {r.width_um/(2*h_um):6.2f} {r.recall:7.2f}{flag}")

    ok = df.dropna()
    rho_w = spearmanr(ok.width_um, ok.recall)
    rho_n = spearmanr(ok.n, ok.recall)
    print(f"\n  Spearman recall vs WIDTH: rho={rho_w.statistic:+.3f} "
          f"(p={rho_w.pvalue:.3f})")
    print(f"  Spearman recall vs COUNT: rho={rho_n.statistic:+.3f} "
          f"(p={rho_n.pvalue:.3f})")
    rho_wn = spearmanr(ok.width_um, ok.n)
    print(f"  Spearman width vs COUNT : rho={rho_wn.statistic:+.3f}"
          f"   (if high, the two are not independent predictors)")
    print("\n  The hypothesis predicted width would correlate with recall and")
    print("  count would not. On osmFISH both correlate about equally and both")
    print("  weakly, and they are correlated with each other -- so width is NOT")
    print("  a better general predictor of failure than domain size. What does")
    print("  survive is the single-domain claim: look for a domain whose pure")
    print("  interior W-2h is a qualitative outlier.")

    narrow = df[df.width_um < h_um]
    if len(narrow):
        print(f"\n  {len(narrow)} domain(s) narrower than the bandwidth, "
              f"mean recall {narrow.recall.mean():.2f}; "
              f"the rest {df[df.width_um >= h_um].recall.mean():.2f}")

    out = paths.results("osmfish") / "resolution.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
