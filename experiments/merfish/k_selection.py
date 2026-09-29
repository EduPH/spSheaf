"""The same k-selection procedure, on a real annotated section.

``experiments/synthetic/k_selection.py`` establishes the procedure where K is a
fact. This applies it, unchanged and importing the same helpers, to a MERFISH
section where the annotation says 8 -- which is one expert's granularity rather
than a fact, and so is the interesting test: what the criteria return here is
evidence about the criteria only in so far as the annotation is right.

Everything is at the manuscript's MERFISH defaults (Table of reference
hyperparameters) and reads the per-slice molecule table the rest of the
revision experiments read, so the k = 8 row is comparable with Table 1.

Cost is one full pipeline run per (k, perturbation): about 15 k values times
three runs. It is resumable -- completed k values are skipped -- so it can be
run in chunks:

    python -m experiments.merfish.k_selection --k 2 6
    python -m experiments.merfish.k_selection --k 7 11
    python -m experiments.merfish.k_selection --k 12 16
    make merfish-k-selection

Writes ``merfish/k_selection_<slice>.csv`` under the results root. The
silhouette here is computed on every grid point, matching
``k_sensitivity.py``; a 4000-point subsample disagreed with it by up to 0.014,
which is enough to move the argmax of a curve whose top is flat to within
0.001 -- see the note in that script.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from spsheaf import evaluation, paths, pipeline
from experiments.merfish import _cache
from experiments.synthetic.k_selection import bracket, transfer

#: the manuscript's MERFISH configuration
CFG = dict(grid_spacing=15.0, bandwidth=25.0, smooth_radius=60.0,
           spatial_weight=0.5, boundary_alpha=0.5, uniform_dim=6,
           k_graph=6, k_nbrs=50)
SLICE_COL = "matched_h5ad_file"


def run(df, k, offset, seed):
    res = pipeline.run_single_slice(
        mol=df, slice_name=df[SLICE_COL].iloc[0], n_clusters=k,
        cleanup=True, eval_at_cell_level=False, grid_offset=offset,
        random_state=seed, verbose=False, **CFG)
    return (np.asarray(res["grid_coords"]), np.asarray(res["labels"]),
            np.asarray(res["fibers_norm"]))


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--slice", default="MERFISH_0.19.h5ad")
    a.add_argument("--k", type=int, nargs=2, default=[2, 16], metavar=("MIN", "MAX"))
    a.add_argument("--shifts", type=float, nargs="+", default=[0.5],
                   help="grid-phase offsets, as fractions of the grid spacing")
    a.add_argument("--seed", type=int, default=42)
    a = a.parse_args()

    p = _cache.parquet_path(a.slice)
    npz = p.with_suffix(".npz")
    if p.exists():
        df = pd.read_parquet(p)
    elif npz.exists():                      # portable form, no parquet engine
        z = np.load(npz, allow_pickle=False)
        g = z["gene_names"]; dn = z["domain_names"]
        dom = z["domain"]
        df = pd.DataFrame({
            "gene": g[z["gene"]],
            # run_single_slice derives x_local from x; x_local already has
            # its minimum at zero, so passing it as x is the same table
            "x": z["x_local"], "y": z["y_local"],
            "x_local": z["x_local"], "y_local": z["y_local"],
            "domain": np.where(dom >= 0, dn[np.clip(dom, 0, None)], None)})
    else:
        raise SystemExit(f"neither {p} nor {npz} found")
    if SLICE_COL not in df.columns:
        df[SLICE_COL] = a.slice
    print(f"{a.slice}: {len(df):,} molecules, {df.gene.nunique()} genes, "
          f"{df.domain.nunique()} annotated domains")

    out = (paths.results("merfish")
           / f"k_selection_{a.slice.replace('.h5ad', '')}.csv")
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
    done = set(prev.k) if len(prev) else set()
    if done:
        print(f"  resuming: k = {sorted(done)} already done")
    rows = prev.to_dict("records") if len(prev) else []
    offsets = [(f * CFG["grid_spacing"], f * CFG["grid_spacing"])
               for f in a.shifts]

    print(f"\n{'k':>3s} {'stability':>10s} {'silhouette':>11s} "
          f"{'ARI vs annot.':>14s} {'seed stab.':>11s} {'min':>6s}")
    for k in range(a.k[0], a.k[1] + 1):
        if k in done:
            continue
        t0 = time.time()
        g0, l0, fib0 = run(df, k, None, a.seed)
        gt, valid, _ = evaluation.map_ground_truth(df, g0)
        stab = [adjusted_rand_score(l0, transfer(*run(df, k, off, a.seed)[:2], g0))
                for off in offsets]
        _, l_seed, _ = run(df, k, None, a.seed + 1)
        rows.append(dict(
            slice=a.slice, k=k,
            stability=float(np.mean(stab)), stability_sd=float(np.std(stab)),
            seed_stability=adjusted_rand_score(l0, l_seed),
            silhouette=(float(silhouette_score(fib0, l0))
                        if len(np.unique(l0)) > 1 else np.nan),
            ari_vs_annotation=adjusted_rand_score(gt[valid], l0[valid]),
            n_grid=len(g0)))
        r = rows[-1]
        print(f"{k:3d} {r['stability']:10.3f} {r['silhouette']:11.3f} "
              f"{r['ari_vs_annotation']:14.3f} {r['seed_stability']:11.3f} "
              f"{(time.time() - t0) / 60:5.1f}m", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).sort_values("k").to_csv(out, index=False)

    d = pd.DataFrame(rows).sort_values("k")
    print(f"\n-> {out}  ({len(d)} of {a.k[1] - a.k[0] + 1} k values)")
    if len(d) < 4:
        return
    lo, hi = bracket(d.rename(columns={"ari_vs_annotation": "ari_vs_truth"}))
    m = d.set_index("k")
    print(f"  annotated k                   = {df.domain.nunique()}")
    print(f"  argmax ARI against annotation = {int(m.ari_vs_annotation.idxmax())}")
    print(f"  argmax silhouette             = {int(m.silhouette.idxmax())}")
    print(f"  argmax grid-phase stability   = {int(m.stability.idxmax())}")
    print(f"  bracket [silhouette, stability bound] = {lo}-{hi}")
    print(f"  seed stability, min over k    = {m.seed_stability.min():.3f}")


if __name__ == "__main__":
    main()
