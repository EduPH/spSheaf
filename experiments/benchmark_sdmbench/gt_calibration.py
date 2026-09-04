#!/usr/bin/env python
"""
Ground-truth calibration for the Moran's I / Geary's C ("marker score") panels.

Computes those two metrics on the TRUE domain labels of each MERFISH slice,
using the identical path our pipeline uses for its own labels
(``pipeline._marker_autocorr`` -> squidpy ``compute_marker_score`` if available,
else the manual fallback). The ground truth is the correct partition, so on a
comparable metric it should land at or above the best benchmark method
(baseline MERFISH Moran's I range ~0.15-0.25).

Interpretation:
  * GT Moran's I lands near the top of the baselines  -> metric is comparable;
    any low value for our method is a real (or fixable) property.
  * GT Moran's I lands low (like our value)            -> the metric is NOT
    computed comparably to the feather; the two panels should be footnoted or
    dropped rather than reported as a ranking.

Usage
-----
    python gt_calibration.py \
        --mol-path ../MERFISH/moffitt2018_export_full/molecules_full_add.csv.gz \
        --output-dir output_benchmark
"""
import argparse
import os
import numpy as np
import pandas as pd

from spsheaf import io, pipeline


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mol-path",
                   default="../MERFISH/moffitt2018_export_full/molecules_full_add.csv.gz")
    p.add_argument("--output-dir", default="output_benchmark")
    return p


def main():
    a = build_parser().parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    mol = io.load_data(a.mol_path)
    slices = sorted(io.identify_slices(mol, verbose=False)["slice_name"].tolist())

    rows = []
    for s in slices:
        mol_single = io.select_slice(mol, s, verbose=False)
        cell_coords, cell_gt_str, cell_ids = pipeline._extract_cells(mol_single)
        if cell_coords is None or cell_gt_str is None:
            print(f"  {s}: no cell/domain metadata — skipped")
            continue
        valid = pd.notna(cell_gt_str)
        mI, gC = pipeline._marker_autocorr(
            mol_single, cell_ids[valid], cell_coords[valid],
            cell_gt_str[valid], verbose=False)
        rows.append(dict(slice=s, gt_morani=mI, gt_gearyc=gC,
                         n_cells=int(valid.sum())))
        print(f"  {s}: GT Moran's I={mI:.3f}  Geary's C={gC:.3f}  "
              f"({int(valid.sum())} cells)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.output_dir, "gt_calibration.csv"), index=False)
    print("\n=== ground-truth marker autocorrelation (median over slices) ===")
    print(f"  GT Moran's I = {df.gt_morani.median():.3f}   "
          f"(baseline range 0.15-0.25)")
    print(f"  GT Geary's C = {df.gt_gearyc.median():.3f}   "
          f"(baseline range 0.73-0.86)")
    print(f"\n-> {os.path.join(a.output_dir, 'gt_calibration.csv')}")
    print("If GT sits at/above the best baseline, the metric is comparable; "
          "if GT is low too, footnote/drop the Moran/Geary panels.")


if __name__ == "__main__":
    main()
