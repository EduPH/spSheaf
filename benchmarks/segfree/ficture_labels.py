"""Write the per-cell labels of FICTURE's best configuration, for the figures.

``eval_ficture.py`` scores every (model, mapping, seed) combination but keeps
only the metric table. This reproduces the labels of the best row UNDER THE
RULE OF THE COMPARISON -- one run per configuration at FICTURE's default seed
(``make_benchmark_table.FIC_SEED``), then the argmax; the same row Table 2
reports -- and writes them next to the SSAM and Points2Regions labels:

    python -m benchmarks.segfree.ficture_labels spacetx_BARISTASeq

reads  data/results/segfree/ficture/metrics_<tag>.csv
       data/results/segfree/ficture/<tag>/analysis/<model>/*.pixel.sorted.tsv.gz
       data/cache/segfree/ficture_input/<tag>/{cell_xy_um.npy,ground_truth.csv}
writes data/results/segfree/<tag>/labels_ficture_best.csv   (gt,pred)
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from benchmarks.segfree import common as bc
from benchmarks.segfree import make_benchmark_table as bt
from benchmarks.segfree.eval_ficture import (argmax_labels, cell_factor_profiles,
                                             cluster_profiles, read_pixel_output)
from spsheaf import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="e.g. spacetx_BARISTASeq")
    ap.add_argument("--radius", type=float, default=30.0)
    a = ap.parse_args()
    res = paths.results("segfree")
    metrics = pd.read_csv(res / "ficture" / f"metrics_{a.tag}.csv")
    best = metrics.loc[bt.ficture_one_run(metrics).ARI.idxmax()]
    in_dir = paths.cache("segfree") / "ficture_input" / a.tag
    cell_xy = np.load(in_dir / "cell_xy_um.npy")
    gt_table = pd.read_csv(in_dir / "ground_truth.csv")
    gt = gt_table["gt"].astype(str).values
    k_gt = len(set(gt))
    hits = glob.glob(str(res / "ficture" / a.tag / "analysis" / best.model / "*.pixel.sorted.tsv.gz"))
    if len(hits) != 1:
        raise SystemExit(f"expected one pixel file for {best.model}, found {hits}")
    pix_xy, kmat, pmat, K, _ = read_pixel_output(hits[0])
    if best.strategy == "argmax":
        lab = bc.fill_background(argmax_labels(cell_xy, pix_xy, kmat, radius=a.radius), cell_xy, bg=-1)
    else:
        prof, _ = cell_factor_profiles(cell_xy, pix_xy, kmat, pmat, K, radius=a.radius)
        lab = cluster_profiles(prof, k_gt, cell_xy=cell_xy if best.strategy == "spatial" else None,
                               seed=int(best.seed))
    m = bc.all_metrics(gt, lab, cell_xy)
    print(f"{a.tag}: {best.strategy} {best.model} seed={best.seed} -> ARI {m['ARI']:.3f} "
          f"(table {best.ARI:.3f}) NMI {m['NMI']:.3f}")
    if abs(m["ARI"] - best.ARI) > 1e-6:
        raise SystemExit("reproduced labels do not match the metric table")
    out = res / a.tag
    os.makedirs(out, exist_ok=True)
    pd.DataFrame({"cell": gt_table["cell"].values, "gt": gt, "pred": lab}).to_csv(out / "labels_ficture_best.csv", index=False)
    print("wrote", out / "labels_ficture_best.csv")


if __name__ == "__main__":
    main()
