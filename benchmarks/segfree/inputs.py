"""Declared inputs for the segmentation-free benchmark.

The comparison was originally run on the collaborator's machine, where the
input files sat under a hardcoded absolute path with names that differ from
the copies in this repository's data tree. Rather than silently guess, every
required file is declared here and resolved through :mod:`spsheaf.paths`.

Run ``python -m benchmarks.segfree.inputs`` to see which are present.

If a file is missing you have two options: obtain it from the collaborator, or
point the entry at your local equivalent and confirm the loader still agrees
with it. Do not simply substitute a similarly-named file — the osmFISH
molecule table in particular exists here as HDF5 while the loader expects CSV,
and the two are not interchangeable without a conversion step.
"""

from __future__ import annotations

from spsheaf import paths

# name -> (dataset, relative path under data/raw/<dataset>/, what it is)
REQUIRED = {
    "merfish_molecules": (
        "merfish",
        "moffitt2018_hypothalamus_molecules.csv.gz",
        "Raw molecule table, 20.5M rows. Local candidate: "
        "moffitt2018_export_full/molecules_full.csv.gz (VERIFY column names "
        "before substituting).",
    ),
    "merfish_annotation": (
        "merfish",
        "MERFISH_{slice_id}.h5ad",
        "Per-slice SDMBench AnnData carrying the 8-domain annotation. "
        "Not present locally; slice_id is one of 0.04/0.09/0.14/0.19/0.24.",
    ),
    "osmfish_molecules": (
        "osmfish",
        "mRNA_coords_raw_counting.csv",
        "Molecule coordinates in the raw imaging frame. Local copy is "
        "mRNA_coords_raw_counting.hdf5 — needs conversion, not renaming.",
    ),
    "osmfish_annotation": (
        "osmfish",
        "osmfish_remove_excluded.h5ad",
        "SDMBench AnnData with the 11 cortical regions. Local candidate: "
        "osmfish_20260512084934.h5ad (VERIFY the Region column matches).",
    ),
    "osmfish_loom": (
        "osmfish",
        "osmFISH_SScortex_mouse_all_cells.loom",
        "Published loom file; defines the coordinate frame the annotation "
        "lives in. Present locally.",
    ),
}


def path(name: str, **fmt) -> "object":
    dataset, rel, _ = REQUIRED[name]
    return paths.data_root() / "raw" / dataset / rel.format(**fmt)


def check() -> int:
    """Report which declared inputs exist. Returns the number missing."""
    missing = 0
    try:
        root = paths.data_root()
    except FileNotFoundError as exc:
        print(f"! {exc}")
        return len(REQUIRED)

    print(f"data root: {root}\n")
    for name, (dataset, rel, note) in REQUIRED.items():
        p = root / "raw" / dataset / rel
        if "{slice_id}" in rel:
            found = [
                (root / "raw" / dataset / rel.format(slice_id=s)).exists()
                for s in ("0.04", "0.09", "0.14", "0.19", "0.24")
            ]
            ok = all(found)
            shown = f"{sum(found)}/5 slices"
        else:
            ok = p.exists()
            shown = "present" if ok else "MISSING"
        if not ok:
            missing += 1
        print(f"  [{'ok' if ok else '  '}] {name:22s} {shown}")
        print(f"       {rel}")
        if not ok:
            print(f"       -> {note}")
    print()
    if missing:
        print(f"{missing} of {len(REQUIRED)} inputs unavailable; the benchmark "
              f"cannot be rerun end to end until these are resolved.")
    else:
        print("all declared inputs present.")
    return missing


if __name__ == "__main__":
    raise SystemExit(1 if check() else 0)
