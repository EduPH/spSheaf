"""Dump what Extended Data Fig. 7 (multi-slice alignment) is drawn from.

Runs the per-slice pipeline on the five MERFISH slices, the gene-centroid
registration of :mod:`multislice_alignment` and the self-alignment control,
and saves the arrays the figure needs: per-slice grid coordinates (native and
aligned), Hungarian-matched labels, per-slice ARI/NMI, the recovered shifts,
the four adjacent-pair agreement rates and the self-alignment upper bound.

    python -m experiments.merfish.multislice_panels

Writes ``data/results/merfish/panels/multislice.npz`` + ``.json``.
"""

from __future__ import annotations

import numpy as np

from spsheaf import evaluation, io, paths, pipeline
from experiments.common.panel_export import export_panels
from experiments.merfish import multislice_alignment as msa

RAW_CSV = "moffitt2018_export_full/molecules_full_add.csv.gz"

#: The columns the per-slice pipeline and the registration read.
COLS = ["gene", "x", "y", "z", "cell_id", "cell_x", "cell_y", "cell_z",
        "matched_h5ad_file", "domain"]
BREGMAS = [-0.04, -0.09, -0.14, -0.19, -0.24]      # the five annotated sections


def load_annotated(path):
    """The molecule table restricted to the five annotated slices.

    ``io.load_data`` reads all 20.5 M rows and 21 columns, which needs more
    than 8 GB; only the annotated slices and nine columns are used here, so
    read in chunks and keep just those.
    """
    import pandas as pd
    parts = []
    for ch in pd.read_csv(path, usecols=COLS, chunksize=2_000_000, low_memory=False):
        if io.MERFISH_INPUT == "section":
            ch = ch[np.isclose(ch.cell_z.to_numpy(dtype=float)[:, None], BREGMAS).any(1)]
        else:
            ch = ch[ch.matched_h5ad_file.notna()]
        for c in ("gene", "cell_id", "matched_h5ad_file", "domain"):
            ch[c] = ch[c].astype("category")
        parts.append(ch)
    mol = pd.concat(parts, ignore_index=True)
    for c in ("gene", "cell_id", "matched_h5ad_file", "domain"):
        mol[c] = mol[c].astype(object)
    return mol


def main():
    mol = load_annotated(str(paths.raw("merfish") / RAW_CSV))
    print(f"{len(mol):,} molecules on the annotated slices", flush=True)
    names = sorted(io.identify_slices(mol)["slice_name"].tolist())
    results = []
    for sn in names:
        r = pipeline.run_single_slice(
            mol, sn, n_clusters=8, grid_spacing=15, bandwidth=25,
            k_graph=6, k_nbrs=50, uniform_dim=6, smooth_radius=60.0,
            spatial_weight=0.5, boundary_alpha=0.5, cleanup=True,
            verbose=False)
        print(f"{sn}: ARI {r['ari']:.3f}  NMI {r['nmi']:.3f}")
        results.append(r)

    out = msa.run_multislice_alignment(
        slice_results=results, mol=mol, grid_spacing=15, bandwidth=25,
        k_graph=6, k_nbrs=50, uniform_dim=6, interslice_radius=45.0, plot=False)
    self_al = msa.run_self_alignment_experiment(
        results[2], shift=(50.0, -70.0), interslice_radius=45.0, grid_spacing=15)

    arrays, meta = {}, dict(slices=[], ari=[], nmi=[], shift_um=[], inlier_frac=[])
    for k, r in enumerate(results):
        vis = evaluation.hungarian_match_labels(r["labels"], r["gt_int"], r["gt_valid"])
        arrays[f"grid_{k}"] = r["grid_coords"]
        arrays[f"grid_aligned_{k}"] = r["grid_coords_aligned"]
        arrays[f"labels_{k}"] = vis
        meta["slices"].append(r["slice_name"])
        meta["ari"].append(float(r["ari"])); meta["nmi"].append(float(r["nmi"]))
        meta["shift_um"].append([float(x) for x in out["transforms"][k]["shift_um"]])
        meta["inlier_frac"].append(float(out["transforms"][k]["inlier_frac"]))
    meta["ref_idx"] = int(out["ref_idx"])
    meta["pairs"] = [list(m["pair"]) for m in out["pair_metrics"]]
    meta["agreement"] = [float(m["agreement_rate"]) for m in out["pair_metrics"]]
    meta["n_matched"] = [int(m["n_matched"]) for m in out["pair_metrics"]]
    meta["mean_agreement"] = float(np.mean(meta["agreement"]))
    meta["self_alignment"] = dict(
        slice=results[2]["slice_name"], shift=list(self_al["shift"]),
        recovered=[float(x) for x in self_al["recovered"]],
        err_um=float(self_al["err"]), agreement=float(self_al["agreement"]))
    npz = export_panels(str(paths.results("merfish")), "multislice", arrays=arrays, meta=meta)
    print("mean agreement", round(meta["mean_agreement"], 3),
          "self-alignment", round(meta["self_alignment"]["agreement"], 3), "->", npz)


if __name__ == "__main__":
    main()
