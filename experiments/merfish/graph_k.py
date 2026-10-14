"""Does the square grid with k_graph = 6 bias the result?

On a square lattice the four axis neighbours are at delta and the four
diagonals at delta*sqrt(2), so k = 6 keeps the axis neighbours and two of the
four tied diagonals; which two is decided by the tree's tie order, vertex by
vertex, and symmetrisation puts most of the others back (interior degree 6-8,
both diagonal orientations equally represented). k = 4 (axis only) and k = 8
(all diagonals) are the two isotropic alternatives. This runs the five slices
at the reference configuration with k_graph in {4, 6, 8} over three seeds and
writes data/results/merfish/graph_k.csv.

    python -m experiments.merfish.graph_k [--seeds 3]
"""
from __future__ import annotations

import argparse, time
import numpy as np, pandas as pd

from spsheaf import paths, pipeline
from experiments.merfish import _cache
from experiments.merfish.k_selection import CFG, SLICE_COL

SLICES = ["0.04", "0.09", "0.14", "0.19", "0.24"]


def main():
    a = argparse.ArgumentParser(); a.add_argument("--seeds", type=int, default=3)
    a.add_argument("--k-graph", type=int, nargs="+", default=[4, 6, 8]); a = a.parse_args()
    out = paths.results("merfish") / "graph_k.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for s in SLICES:
        df = pd.read_parquet(_cache.parquet_path(f"MERFISH_{s}.h5ad"))
        if SLICE_COL not in df.columns:
            df[SLICE_COL] = f"MERFISH_{s}.h5ad"
        for kg in a.k_graph:
            for seed in range(a.seeds):
                if len(done) and ((done.slice == float(s)) & (done.k_graph == kg) & (done.seed == seed)).any():
                    continue
                cfg = dict(CFG); cfg["k_graph"] = kg
                t0 = time.time()
                r = pipeline.run_single_slice(mol=df, slice_name=f"MERFISH_{s}.h5ad", n_clusters=8,
                                              cleanup=True, random_state=seed, verbose=False, **cfg)
                rows.append(dict(slice=s, k_graph=kg, seed=seed, ari=r["metrics"]["ari"],
                                 nmi=r["metrics"]["nmi"], sec=time.time() - t0))
                pd.DataFrame(rows).to_csv(out, index=False)
                print(f"{s} k_graph={kg} seed={seed}: ARI {r['metrics']['ari']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    d = pd.DataFrame(rows)
    print(d.groupby("k_graph").ari.agg(["mean", "std"]).round(4))
    print(d.pivot_table(index="slice", columns="k_graph", values="ari").round(3))


if __name__ == "__main__":
    main()
