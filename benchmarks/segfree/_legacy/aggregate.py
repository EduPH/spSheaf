"""Collect the per-slice benchmark JSONs into comparison tables.

Outputs
  results/merfish_per_slice.csv   one row per (method, slice, setting)
  results/merfish_summary.csv     mean +- sd across the 5 slices per method
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from benchmarks.segfree import common as bc

METRICS = ["ARI", "NMI", "HOM", "COM", "CHAOS", "PAS", "ASW", "n_pred_domains"]

# Published numbers for the sheaf method on this benchmark, Table 1 of the
# manuscript (Cellular Sheaves for Spatial Domain Inference, Aug 2026 draft).
SHEAF_TABLE1 = pd.DataFrame([
    dict(slice="0.04", ARI=0.528, NMI=0.617, HOM=0.623, COM=0.611, CHAOS=0.0287, PAS=0.0116, ASW=-0.043, time_s=8.80),
    dict(slice="0.09", ARI=0.513, NMI=0.626, HOM=0.645, COM=0.608, CHAOS=0.0286, PAS=0.0120, ASW=0.035, time_s=8.73),
    dict(slice="0.14", ARI=0.442, NMI=0.554, HOM=0.573, COM=0.536, CHAOS=0.0288, PAS=0.0116, ASW=-0.066, time_s=9.12),
    dict(slice="0.19", ARI=0.650, NMI=0.682, HOM=0.693, COM=0.672, CHAOS=0.0287, PAS=0.0120, ASW=-0.017, time_s=8.89),
    dict(slice="0.24", ARI=0.654, NMI=0.674, HOM=0.681, COM=0.668, CHAOS=0.0288, PAS=0.0122, ASW=-0.073, time_s=8.99),
])
# Best published competitor in the SDMBench comparison quoted by the manuscript.
SCANIT_MEAN_NMI = 0.554


def rows_from_ssam(d, method):
    out = []
    for setting, m in d.get("domains", {}).items():
        r = {k: m.get(k) for k in METRICS}
        r.update(method=method, setting=setting, slice=d["slice"],
                 time_s=m.get("total_s"), params=json.dumps(m.get("params", {})))
        out.append(r)
    return out


def rows_from_ssam_guided(d):
    out = []
    for setting in ("published", "best_sweep"):
        if setting not in d:
            continue
        m = d[setting]
        r = {k: m.get(k) for k in METRICS}
        r.update(method="SSAM (guided, GT cell types)", setting=setting, slice=d["slice"],
                 time_s=m.get("find_domains_s"), params=json.dumps(m.get("params", {})))
        out.append(r)
    return out


def rows_from_p2r(d):
    out = []
    best = d["best"]
    r = {k: best.get(k) for k in METRICS}
    r.update(method="Points2Regions", setting="best_grid", slice=d["slice"],
             time_s=best.get("total_s"),
             params=json.dumps({k: best[k] for k in ("pixel_width", "pixel_smoothing")}))
    out.append(r)
    # seed-averaged score at the best setting: the honest number to quote, since
    # a single favourable seed overstates a k-means-based method.
    seeds = d.get("seeds", [])
    if seeds:
        r2 = {k: float(np.mean([s[k] for s in seeds])) for k in METRICS}
        r2.update(method="Points2Regions", setting="best_grid_seedmean", slice=d["slice"],
                  time_s=best.get("total_s"),
                  params=json.dumps({k: best[k] for k in ("pixel_width", "pixel_smoothing")}),
                  ARI_sd=float(np.std([s["ARI"] for s in seeds])),
                  NMI_sd=float(np.std([s["NMI"] for s in seeds])))
        out.append(r2)
    return out


def main():
    rows = []
    for f in sorted(glob.glob("results/ssam/metrics_*.json")):
        rows += rows_from_ssam(json.load(open(f)), "SSAM (de novo)")
    for f in sorted(glob.glob("results/ssam_guided/metrics_*.json")):
        rows += rows_from_ssam_guided(json.load(open(f)))
    for f in sorted(glob.glob("results/p2r/metrics_*.json")):
        rows += rows_from_p2r(json.load(open(f)))

    sheaf = SHEAF_TABLE1.copy()
    sheaf["method"] = "Sheaf boundary (published Table 1)"
    sheaf["setting"] = "published"
    sheaf["n_pred_domains"] = 8
    rows += sheaf.to_dict("records")

    per = pd.DataFrame(rows)
    per = per[["method", "setting", "slice"] + METRICS + ["time_s", "params"]
              + [c for c in per.columns if c.endswith("_sd")]]
    per = per.sort_values(["method", "setting", "slice"])
    per.to_csv("results/merfish_per_slice.csv", index=False)

    agg = (per.groupby(["method", "setting"])[["ARI", "NMI", "HOM", "COM", "CHAOS", "PAS", "ASW", "n_pred_domains", "time_s"]]
           .agg(["mean", "std", "count"]))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index().sort_values("ARI_mean", ascending=False)
    agg.to_csv("results/merfish_summary.csv", index=False)

    show = agg[["method", "setting", "ARI_mean", "ARI_std", "NMI_mean", "NMI_std",
                "n_pred_domains_mean", "ARI_count"]].round(3)
    print(show.to_string(index=False))
    print(f"\nSCAN-IT (best published in SDMBench, quoted by the manuscript): mean NMI = {SCANIT_MEAN_NMI}")
    return per, agg


if __name__ == "__main__":
    main()
