"""Re-run each SpaceTx section at the argmax of its own in-sample grid.

Table 2's in-sample row is the best row of ``sweep.csv``; ED Fig. 6's in-sample
column is a map, and a map needs a full run with its per-cell arrays dumped.
Reading the configuration out of the sweep table rather than writing it down
again is the point of this script: the two cannot disagree, and when the grid is
extended the panel follows without anybody editing a command.

    python -m experiments.spacetx.insample_panels                # both sections
    python -m experiments.spacetx.insample_panels BARISTASeq      # just one
    python -m experiments.spacetx.insample_panels --dry-run       # print, don't run

Each section is one full pipeline run (a few minutes). The dump lands in
``spacetx/<ds>/panels/spsheaf_insample.{npz,json}``, which is where
``figures/ed_spacetx.py`` looks for it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import pandas as pd

from spsheaf import paths

PANEL_NAME = "spsheaf_insample"

#: the column names run_spacetx takes on the command line
FLAGS = {"bandwidth": "--bandwidth", "smooth_radius": "--smooth-radius",
         "uniform_dim": "--uniform-dim", "spatial_weight": "--spatial-weight",
         "boundary_alpha": "--boundary-alpha", "reference": "--reference"}

#: column names differ between the SpaceTx tables; these two are what we use
COLS = ["--spot-x", "xc", "--spot-y", "yc", "--spot-gene", "gene",
        "--cell-id", "sample_name", "--cell-x", "rotated_x",
        "--cell-y", "rotated_y", "--label", "layer"]

DATASETS = ["MERFISH_Allen_Prototype", "BARISTASeq"]


def argmax_row(ds):
    p = paths.results("spacetx") / ds / "sweep.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found; run 'make spacetx-insample-grid' first")
    d = pd.read_csv(p).dropna(subset=["ari"])
    if d.empty:
        raise SystemExit(f"{p} has no scored rows")
    return d.loc[d.ari.idxmax()], len(d)


def command(ds, row):
    flags = []
    for col, flag in FLAGS.items():
        if col not in row or pd.isna(row[col]):
            continue
        v = row[col]
        flags += [flag, f"{v:g}" if isinstance(v, float) else str(v)]
    return [sys.executable, "-m", "experiments.spacetx.run_spacetx", ds, *COLS,
            "--grid-spacing", "10", *flags, "--panel-name", PANEL_NAME]


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("datasets", nargs="*", default=None, help=f"default: {' '.join(DATASETS)}")
    a.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    a = a.parse_args()

    failed = []
    for ds in (a.datasets or DATASETS):
        row, n = argmax_row(ds)
        cmd = command(ds, row)
        print(f"\n{ds}: best of {n} configurations, ARI {row.ari:.3f}")
        print("  " + " ".join(cmd[2:]))
        if a.dry_run:
            continue
        if subprocess.call(cmd) != 0:
            failed.append(ds)
    if failed:
        raise SystemExit("failed: " + ", ".join(failed))
    if not a.dry_run:
        print("\nnow rebuild the figure:  cd figures && python ed_spacetx.py")


if __name__ == "__main__":
    main()
