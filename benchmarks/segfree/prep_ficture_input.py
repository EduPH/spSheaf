"""Convert the benchmark molecule tables into FICTURE's input format.

Run as: python prep_ficture_input.py merfish [slice ...]
        python prep_ficture_input.py osmfish
        python prep_ficture_input.py spacetx <dataset>   (e.g. MERFISH_Allen_Prototype)

FICTURE needs three files per dataset:
  transcripts.tsv.gz     X, Y, gene, Count  -- sorted along the major axis
  coordinate_minmax.tsv  xmin/xmax/ymin/ymax
  feature.clean.tsv.gz   gene, Count  -- the gene whitelist with total counts

The molecule tables are one row per molecule, so Count is 1 throughout; FICTURE
aggregates them into hexagons itself. Coordinates are written in micron (the
osmFISH table is in camera pixels at 15.3846 px/um and is converted here), and
shifted so the minimum is at the origin, because FICTURE's decode step writes
integer-scaled coordinates and negative values complicate the back-transform.
The shift is recorded in coordinate_minmax.tsv and in offset.json so predictions
can be mapped back onto the annotated cell centroids.
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

from benchmarks.segfree import common as bc

from spsheaf import paths as _paths
OUT_ROOT = str(_paths.cache("segfree/ficture_input"))
PIXEL_PER_UM_OSM = 15.3846


def write_dataset(name, x_um, y_um, genes, cell_xy_um=None, extra=None):
    """Write FICTURE's three input files, with coordinates shifted to the origin."""
    d = f"{OUT_ROOT}/{name}"
    os.makedirs(d, exist_ok=True)
    x0, y0 = float(np.min(x_um)), float(np.min(y_um))
    df = pd.DataFrame({"X": np.round(x_um - x0, 2), "Y": np.round(y_um - y0, 2),
                       "gene": genes, "Count": 1})
    # FICTURE reads the transcript file streaming along the major axis, so it
    # must be sorted on that axis (Y here, matching --major-axis Y).
    df = df.sort_values("Y", kind="mergesort")
    df.to_csv(f"{d}/transcripts.tsv.gz", sep="\t", index=False, compression="gzip")

    with open(f"{d}/coordinate_minmax.tsv", "w") as f:
        f.write(f"xmin\t{df.X.min():.2f}\nxmax\t{df.X.max():.2f}\n"
                f"ymin\t{df.Y.min():.2f}\nymax\t{df.Y.max():.2f}\n")

    feat = df.groupby("gene", as_index=False)["Count"].sum()
    feat.to_csv(f"{d}/feature.clean.tsv.gz", sep="\t", index=False, compression="gzip")

    meta = {"x_offset": x0, "y_offset": y0, "n_molecules": int(len(df)),
            "n_genes": int(feat.shape[0]), "units": "micron"}
    if cell_xy_um is not None:
        np.save(f"{d}/cell_xy_um.npy", np.asarray(cell_xy_um) - [x0, y0])
        meta["n_cells"] = int(len(cell_xy_um))
    if extra:
        meta.update(extra)
    with open(f"{d}/offset.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[{name}] {len(df):,} molecules, {feat.shape[0]} genes, "
          f"extent {df.X.max():.0f} x {df.Y.max():.0f} um", flush=True)
    return d


def prep_merfish(slices):
    molecules = bc.load_merfish_export()
    for s in slices:
        adata, mols = bc.load_merfish_slice(s, molecules=molecules)
        gt = adata.obs["domain"].astype(str).values
        d = write_dataset(f"merfish_{s}", mols.x.values, mols.y.values, mols.gene.values,
                          cell_xy_um=adata.obsm["spatial"],
                          extra={"n_gt_domains": int(len(set(gt)))})
        pd.DataFrame({"cell": adata.obs_names, "gt": gt}).to_csv(f"{d}/ground_truth.csv", index=False)


def prep_osmfish():
    from benchmarks.segfree import osm_register as osr
    o, om, _, cell_xy_px, _, _, _ = osr.load_cells()
    gt = o.obs["Region"].astype(str).values
    d = write_dataset("osmfish",
                      om.x.values / PIXEL_PER_UM_OSM, om.y.values / PIXEL_PER_UM_OSM,
                      om.gene.values, cell_xy_um=cell_xy_px / PIXEL_PER_UM_OSM,
                      extra={"n_gt_domains": int(len(set(gt))),
                             "pixel_per_um": PIXEL_PER_UM_OSM})
    pd.DataFrame({"cell": o.obs_names, "gt": gt}).to_csv(f"{d}/ground_truth.csv", index=False)


def prep_spacetx(dataset, spot_x="xc", spot_y="yc", spot_gene="gene",
                 cell_x="rotated_x", cell_y="rotated_y", label="layer", unit_um=1.0):
    """Same frame and registration as benchmarks/segfree/run_spacetx.py."""
    root = _paths.raw("spacetx") / dataset
    spots = pd.read_csv(root / "s3_spot_table.csv", usecols=[spot_x, spot_y, spot_gene])
    cells = pd.read_csv(root / "s3_mapped_cell_table.csv")
    mol = np.c_[spots[spot_x].values, spots[spot_y].values] * unit_um
    cxy = cells[[cell_x, cell_y]].values * unit_um
    meta = _paths.results("spacetx") / dataset / "panels" / "spsheaf.json"
    if meta.exists():
        t = np.asarray(json.loads(meta.read_text())["registration"]["translation"], dtype=float)
        cxy = cxy + t + mol.min(0)
    gt = cells[label].astype(str).values
    d = write_dataset(f"spacetx_{dataset}", mol[:, 0], mol[:, 1], spots[spot_gene].astype(str).values,
                      cell_xy_um=cxy, extra={"n_gt_domains": int(len(set(gt)))})
    pd.DataFrame({"cell": cells.index, "gt": gt}).to_csv(f"{d}/ground_truth.csv", index=False)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "merfish"
    if which == "merfish":
        prep_merfish(sys.argv[2:] or bc.MER_SLICES)
    elif which == "osmfish":
        prep_osmfish()
    elif which == "spacetx":
        prep_spacetx(sys.argv[2] if len(sys.argv) > 2 else "MERFISH_Allen_Prototype")
    else:
        raise SystemExit("usage: prep_ficture_input.py {merfish|osmfish|spacetx <dataset>} [slice ...]")
