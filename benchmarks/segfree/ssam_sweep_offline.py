"""SSAM domain sweep from the exported store (phase 2 of run_ssam.py).

Phase 1 (``SSAM_PIPELINE_ONLY=1 python -m benchmarks.segfree.run_ssam``)
runs KDE -> local maxima -> cell types -> binned cell-type maps and writes
``data/results/segfree/ssam/store_<slice>.npz``. This phase evaluates the
published ``find_domains`` setting and the 24-setting sweep from that file
alone, with SSAM's algorithm re-implemented in ``ssam_domains.py``. It needs
no ``ssam`` install, only numpy/scipy/scikit-learn/scikit-image, and about
10 GB of RAM: at norm_thres=1000 the Ward clustering runs on ~28k windows.

    python -m benchmarks.segfree.ssam_sweep_offline 0.04 0.09 0.14 0.19 0.24

Writes metrics_<slice>.json (same layout as run_ssam.py) and the label tables.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from benchmarks.segfree import common as bc
from benchmarks.segfree import ssam_domains as sd
from benchmarks.segfree.run_ssam import DOMAIN_SETTINGS, SWEEP, OUT, SAMPLING_DISTANCE


def run(slice_id, tag=""):
    z = np.load(f"{OUT}/store_{slice_id}{tag}.npz", allow_pickle=True)   # our own file; str arrays
    store = dict(binned=z["binned"], vf_norm_shape=tuple(int(v) for v in z["vf_norm_shape"]),
                 filtered=z["filtered"])
    cell_xy, gt, names = z["cell_xy"], z["gt"].astype(str), z["cell_names"].astype(str)
    x0, y0 = float(z["x0"]), float(z["y0"])
    timings = json.loads(str(z["timings"]))
    out = {"slice": slice_id, "tag": tag, "n_localmax": int(z["n_localmax"]), "n_celltypes": int(z["n_celltypes"]),
           "n_molecules": int(z["n_molecules"]), "timings": timings, "domains": {}}

    def evaluate(kw, name, save=True):
        t = time.time()
        try:
            dom = sd.find_domains(store, **kw)
        except ValueError as e:
            return dict(ARI=float("nan"), NMI=float("nan"), n_pred_domains=0, params=kw, error=str(e))
        dt = time.time() - t
        zc = dom.shape[2] // 2
        pred = bc.nearest_pixel_labels(dom[..., zc], cell_xy, x0, y0, scale=SAMPLING_DISTANCE)
        pred_filled = bc.fill_background(pred, cell_xy, bg=-1)
        m = bc.all_metrics(gt, pred_filled, cell_xy)
        m.update(find_domains_s=dt, frac_background=float(np.mean(pred == -1)), params=kw)
        if save:
            pd.DataFrame({"cell": names, "gt": gt, "pred_raw": pred, "pred": pred_filled}).to_csv(
                f"{OUT}/labels_{slice_id}{tag}_{name}.csv", index=False)
        return m

    for name, kw in DOMAIN_SETTINGS.items():
        m = evaluate(kw, name); out["domains"][name] = m
        print(f"[ssam {slice_id}{tag} {name}] ARI={m['ARI']:.3f} NMI={m['NMI']:.3f} k={m['n_pred_domains']}", flush=True)
    sweep = []
    for kw in SWEEP:
        sweep.append(evaluate(kw, "sweep", save=False)); out["sweep"] = sweep
        json.dump(out, open(f"{OUT}/metrics_{slice_id}{tag}.partial.json", "w"), indent=1)
        print(f"[ssam {slice_id} sweep {len(sweep)}/{len(SWEEP)}] ARI={sweep[-1]['ARI']:.3f} "
              f"k={sweep[-1]['n_pred_domains']} {kw}", flush=True)
    ok = [r for r in sweep if r["ARI"] == r["ARI"]]
    best = max(ok, key=lambda r: r["ARI"])
    out["domains"]["best_sweep"] = evaluate(best["params"], "best_sweep")
    print(f"[ssam {slice_id} best_sweep] ARI={best['ARI']:.3f} NMI={best['NMI']:.3f} "
          f"k={best['n_pred_domains']} params={best['params']}", flush=True)
    json.dump(out, open(f"{OUT}/metrics_{slice_id}{tag}.json", "w"), indent=1)
    os.remove(f"{OUT}/metrics_{slice_id}{tag}.partial.json")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slices", nargs="*", default=bc.MER_SLICES)
    ap.add_argument("--tags", nargs="+", default=[""],
                    help="store suffixes, e.g. _sct_r4 _sct_r8 (ssam_resolution.py)")
    a = ap.parse_args()
    for s in a.slices:
        for t in a.tags:
            run(s, t)
