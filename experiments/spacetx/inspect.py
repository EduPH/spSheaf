"""Look at a downloaded SpaceTx dataset before writing anything else.

Prints the columns and first rows of the three tables, the spatial extent and
unit guess of spots and cells, the candidate annotation columns (anything
whose name contains layer / region / class / cluster), and whether spot and
cell coordinates share a frame. Read the Readme too; this only reports.

    python -m experiments.spacetx.inspect MERFISH_Allen_Prototype
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from spsheaf import paths


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "MERFISH_Allen_Prototype"
    try:
        root = paths.raw("spacetx") / ds
    except FileNotFoundError:
        raise SystemExit("no data/raw/spacetx yet: run  bash experiments/spacetx/download.sh " + ds)
    tabs = {}
    for name in ("s3_spot_table.csv", "s3_mapped_cell_table.csv", "s3_cell_by_gene.csv"):
        f = root / name
        if not f.exists():
            print(f"missing: {f}")
            continue
        df = pd.read_csv(f, nrows=200000) if name == "s3_spot_table.csv" else pd.read_csv(f)
        tabs[name] = df
        print(f"\n== {name}  ({len(df):,} rows read{' (first 200k)' if name == 's3_spot_table.csv' else ''})")
        print("columns:", list(df.columns))
        print(df.head(3).to_string())
    for name, df in tabs.items():
        xs = [c for c in df.columns if c.lower() in ("x", "x_um", "global_x", "xc", "x_coord", "centroid_x")]
        ys = [c for c in df.columns if c.lower() in ("y", "y_um", "global_y", "yc", "y_coord", "centroid_y")]
        if xs and ys:
            x, y = df[xs[0]].values, df[ys[0]].values
            print(f"\n{name}: {xs[0]}/{ys[0]} extent x [{x.min():.1f}, {x.max():.1f}]  y [{y.min():.1f}, {y.max():.1f}]  "
                  f"span ({x.max()-x.min():.0f}, {y.max()-y.min():.0f})")
            if "cell" in name:
                from scipy.spatial import cKDTree
                d, _ = cKDTree(np.c_[x, y]).query(np.c_[x, y], k=2)
                print(f"  median nearest-cell spacing {np.median(d[:, 1]):.2f} (units of the table; ~10-20 um in VISp)")
        cand = [c for c in df.columns if any(k in c.lower() for k in ("layer", "region", "class", "cluster", "type", "label", "annot"))]
        if cand:
            print(f"{name}: candidate annotation columns {cand}")
            for c in cand[:4]:
                vc = df[c].value_counts(dropna=False)
                print(f"   {c}: {len(vc)} values; top: {dict(vc.head(12))}")
    if "s3_spot_table.csv" in tabs:
        g = [c for c in tabs["s3_spot_table.csv"].columns if c.lower() in ("gene", "target", "gene_name", "feature")]
        if g:
            print(f"\nspot gene column: {g[0]}; {tabs['s3_spot_table.csv'][g[0]].nunique()} genes in the first 200k spots")


if __name__ == "__main__":
    main()
