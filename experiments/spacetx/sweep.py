"""In-sample parameter grid for spSheaf on a SpaceTx dataset -- the BOUND, not
the headline. The headline is the MERFISH-carried default (run_spacetx.py).
Points2Regions and SSAM are reported at their best in-sample setting on a
single section, so this grid is what makes the comparison symmetric: it says
what spSheaf would reach under the same permissive rule.

    python -m experiments.spacetx.sweep MERFISH_Allen_Prototype [--budget-s 500]

Appends to data/results/spacetx/<dataset>/sweep.csv and skips settings already
there, so it can be run repeatedly until the grid is complete.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spsheaf import evaluation, paths, pipeline
from experiments.spacetx import run_spacetx as RS

GRID = dict(bandwidth=(25.0, 40.0, 60.0), spatial_weight=(0.25, 0.5, 0.75),
            smooth_radius=(60.0, 120.0), boundary_alpha=(0.5,))


def main():
    p = RS.build_parser()
    p.add_argument("--budget-s", type=float, default=500.0)
    a = p.parse_args()
    mol, cxy, cid, gt, cbg = RS.load(a)
    valid = pd.notna(gt); names = sorted(set(gt[valid])); k = a.n_clusters or len(names)
    gti = np.array([names.index(g) if isinstance(g, str) else -1 for g in gt])
    meta = json.loads((paths.results("spacetx") / a.dataset / "panels" / "spsheaf.json").read_text())
    t = np.asarray(meta["registration"]["translation"])       # the same label-free registration
    out = paths.results("spacetx") / a.dataset / "sweep.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    t0 = time.time()
    for h, a_s, s_s, a_b in itertools.product(*GRID.values()):
        if len(done) and ((done.bandwidth == h) & (done.spatial_weight == a_s) &
                          (done.smooth_radius == s_s) & (done.boundary_alpha == a_b)).any():
            continue
        if time.time() - t0 > a.budget_s:
            print("budget reached; run again to continue"); break
        res = pipeline.run_single_slice(
            mol=mol, slice_name=f"spacetx_{a.dataset}", n_clusters=k, grid_spacing=a.grid_spacing,
            bandwidth=h, smooth_radius=s_s, k_graph=6, k_nbrs=50, uniform_dim=a.uniform_dim,
            spatial_weight=a_s, boundary_alpha=a_b, cleanup=True, eval_at_cell_level=False,
            random_state=a.seed, verbose=False)
        _, idx = cKDTree(res["grid_coords"]).query(cxy + t, k=1)
        pred = res["labels"][idx]
        row = dict(bandwidth=h, spatial_weight=a_s, smooth_radius=s_s, boundary_alpha=a_b,
                   ari=evaluation.compute_ari(pred, gti, valid), nmi=evaluation.compute_nmi(pred, gti, valid))
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True); done.to_csv(out, index=False)
        print(f"h={h:.0f} a_s={a_s} s_s={s_s:.0f} a_b={a_b}  ARI {row['ari']:.3f} NMI {row['nmi']:.3f}", flush=True)
    if len(done):
        b = done.sort_values("ari", ascending=False).iloc[0]
        print(f"\n{len(done)} settings; best in-sample ARI {b.ari:.3f} NMI {b.nmi:.3f} at "
              f"h={b.bandwidth:.0f} a_s={b.spatial_weight} s_s={b.smooth_radius:.0f}; "
              f"median ARI {done.ari.median():.3f}")


if __name__ == "__main__":
    main()
