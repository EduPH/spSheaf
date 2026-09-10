"""Leave-one-slice-out selection of (sigma_s, alpha_b, alpha_s) on MERFISH.

The manuscript (Methods, "Hyperparameters" and "Matched model selection")
states that the three sheaf parameters were fixed by leave-one-slice-out
cross-validation across the five Moffitt slices. Until this script existed
there was no artifact behind that sentence. It runs the same 27-configuration
grid used for osmFISH (``experiments/osmfish/tune.py``), rescaled to MERFISH
units, through ``pipeline.run_single_slice`` -- the exact code path of
Table 1, input policy of ``spsheaf.io.MERFISH_INPUT``, cell-level scoring -- and applies
``experiments.common.cv.cross_validate``: for each held-out slice the
configuration maximising mean ARI on the other four is selected and its score
on the held-out slice is reported. The defaults of Table 3
(sigma_s 60 um, alpha_b 0.5, alpha_s 0.5) are one of the 27 configurations.

    python -m experiments.merfish.loo_cv            # ~135 runs x ~15 s

Writes data/results/merfish/loo_cv_table.csv (every config x slice) and
data/results/merfish/loo_cv.json (held-out scores, per-fold choice, final).
Resumable: rows already in the CSV are not recomputed.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from experiments.common.cv import cross_validate, summarise
from spsheaf import paths, pipeline

RAW_CSV = "moffitt2018_export_full/molecules_full_add.csv.gz"
SLICES = ["MERFISH_0.04.h5ad", "MERFISH_0.09.h5ad", "MERFISH_0.14.h5ad",
          "MERFISH_0.19.h5ad", "MERFISH_0.24.h5ad"]
GRID = [dict(smooth_radius=sr, boundary_alpha=ab, spatial_weight=sw)
        for sr in (30.0, 60.0, 120.0)
        for ab in (0.25, 0.5, 1.0)
        for sw in (0.3, 0.5, 0.7)]
DEFAULT = dict(smooth_radius=60.0, boundary_alpha=0.5, spatial_weight=0.5)
COLS = ["gene", "x", "y", "z", "matched_h5ad_file", "domain",
        "cell_id", "cell_x", "cell_y", "cell_z"]


def load_slice(slice_name):
    src = paths.raw("merfish") / RAW_CSV
    bregma = -float(slice_name.replace("MERFISH_", "").replace(".h5ad", ""))
    parts = []
    for ch in pd.read_csv(src, usecols=lambda c: c in COLS, chunksize=2_000_000,
                          low_memory=False):
        # the whole section; select_slice then applies io.MERFISH_INPUT
        parts.append(ch[np.isclose(ch.cell_z.to_numpy(dtype=float), bregma)])
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-clusters", type=int, default=8)
    a = ap.parse_args()
    out_dir = paths.results("merfish")
    table_path = out_dir / "loo_cv_table.csv"
    rows = pd.read_csv(table_path).to_dict("records") if table_path.exists() else []
    done = {(r["slice"], r["smooth_radius"], r["boundary_alpha"], r["spatial_weight"])
            for r in rows}

    for s in SLICES:
        todo = [c for c in GRID if (s, c["smooth_radius"], c["boundary_alpha"],
                                    c["spatial_weight"]) not in done]
        if not todo:
            continue
        print(f"{s}: {len(todo)} configurations to run", flush=True)
        mol = load_slice(s)
        for c in todo:
            t0 = time.time()
            res = pipeline.run_single_slice(
                mol, s, n_clusters=a.n_clusters, random_state=a.seed,
                verbose=False, **c)
            m = res["metrics"]
            rows.append(dict(slice=s, **c, ari=float(m["ari"]), nmi=float(m["nmi"]),
                             time_s=time.time() - t0))
            pd.DataFrame(rows).to_csv(table_path, index=False)
            print(f"  sr={c['smooth_radius']:5.0f} ab={c['boundary_alpha']:.2f} "
                  f"sw={c['spatial_weight']:.1f}  ARI {m['ari']:.4f}  NMI {m['nmi']:.4f}"
                  f"  ({time.time() - t0:.0f} s)", flush=True)
        del mol

    df = pd.DataFrame(rows)
    key = lambda c: (c["smooth_radius"], c["boundary_alpha"], c["spatial_weight"])
    lookup = {(r["slice"], r["smooth_radius"], r["boundary_alpha"], r["spatial_weight"]):
              dict(ari=r["ari"], nmi=r["nmi"]) for r in df.to_dict("records")}

    def run_fn(cfg, fold):
        return lookup[(fold, *key(cfg))]

    cv = cross_validate(run_fn, GRID, SLICES)
    summ = summarise(cv)
    default_mean = df[(df.smooth_radius == DEFAULT["smooth_radius"])
                      & (df.boundary_alpha == DEFAULT["boundary_alpha"])
                      & (df.spatial_weight == DEFAULT["spatial_weight"])].ari.mean()
    report = dict(
        summary=summ,
        held_out=[dict(slice=f, **m) for f, m in zip(SLICES, cv["held_out"])],
        chosen_per_fold=cv["chosen_per_fold"],
        final_config=cv["final_config"],
        default_config=DEFAULT,
        default_mean_ari=float(default_mean),
        default_is_final=(cv["final_config"] == DEFAULT),
        grid=GRID, seed=a.seed,
    )
    (out_dir / "loo_cv.json").write_text(json.dumps(report, indent=2))
    print("\nheld-out (LOO) mean ARI {cv_ari_mean:.4f} +- {cv_ari_sd:.4f}, "
          "NMI {cv_nmi_mean:.4f}".format(**summ))
    print(f"final config on all five: {cv['final_config']}")
    print(f"defaults (Table 3) mean ARI over slices: {default_mean:.4f}; "
          f"defaults are the all-slice argmax: {report['default_is_final']}")
    print(f"selection stable across folds: {summ['selection_stable_across_folds']}")
    print("wrote", out_dir / "loo_cv.json")


if __name__ == "__main__":
    main()
