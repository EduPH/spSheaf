"""
Data loading and slice selection for MERFISH spatial transcriptomics.

Handles Imoto's pre-merged molecule table (molecules_full_add.csv.gz)
with columns: gene, x, y, z, cell_class, neuron_cluster_id,
matched_h5ad_file (slice ID), and domain (ground truth).

Tissue slices are identified via the ``matched_h5ad_file`` column
(e.g. ``MERFISH_0.14.h5ad``), not spatial jump detection.
"""

import pandas as pd
import os

import numpy as np


def load_data(mol_path, verbose=True):
    """
    Load Imoto's pre-merged molecule table.

    Parameters
    ----------
    mol_path : str or Path
        Path to ``molecules_full_add.csv.gz``.
    verbose : bool
        Print summary statistics.

    Returns
    -------
    mol : DataFrame
        Full molecule table with one row per detected RNA molecule.
    """
    mol = pd.read_csv(mol_path)

    if verbose:
        print(f"  Molecules : {len(mol):,}")
        print(f"  Genes      : {mol['gene'].nunique()}")
        if "matched_h5ad_file" in mol.columns:
            slices = mol["matched_h5ad_file"].dropna().unique()
            print(f"  Slices     : {len(slices)}")
        if "domain" in mol.columns:
            domains = mol["domain"].dropna().unique()
            print(f"  Domains    : {len(domains)}")

    return mol


def identify_slices(mol, verbose=True):
    """
    Identify tissue slices from the ``matched_h5ad_file`` column.

    Each unique value (e.g. ``MERFISH_0.14.h5ad``) corresponds to
    one tissue slice.

    Parameters
    ----------
    mol : DataFrame
        Full molecule table.
    verbose : bool
        Print per-slice summary.

    Returns
    -------
    slice_info : DataFrame
        Columns: slice_name, n_molecules, x_min, x_max, y_min, y_max.
    """
    col = "matched_h5ad_file"
    if col not in mol.columns:
        raise ValueError(
            f"Column '{col}' not found. Available: {list(mol.columns)}"
        )

    slice_names = sorted(mol[col].dropna().unique())
    rows = []
    for s_name in slice_names:
        ms = mol[mol[col] == s_name]
        rows.append({
            "slice_name": s_name,
            "n_molecules": len(ms),
            "x_min": ms["x"].min(),
            "x_max": ms["x"].max(),
            "y_min": ms["y"].min(),
            "y_max": ms["y"].max(),
        })
        if verbose:
            print(f"  {s_name}: {len(ms):>10,} molecules")

    return pd.DataFrame(rows)


#: Which molecules of a Moffitt slice spSheaf reads.
#:
#: ``section``      every molecule of the physical section (rows whose
#:                  ``cell_z`` equals the slice's bregma coordinate), all
#:                  seven focal planes. Identical to the comparators' input
#:                  (benchmarks/segfree/common.py). Default since 8 Sep 2026.
#: ``annotated_z0`` the molecules of the annotated cells only
#:                  (``matched_h5ad_file == slice``) on the z = 0 plane; the
#:                  input behind the results reported up to 8 Sep 2026.
#: Set with the environment variable SPSHEAF_MERFISH_INPUT.
MERFISH_INPUT = os.environ.get("SPSHEAF_MERFISH_INPUT", "section")


def section_rows(mol, slice_name):
    """Boolean mask of the molecules of ``slice_name``'s physical section.

    The export lays the twelve bregma sections of one animal out in
    overlapping xy blocks, so the section must be selected by ``cell_z``, not
    by coordinates; ``matched_h5ad_file`` marks only the annotated cells.
    """
    ann = mol["matched_h5ad_file"] == slice_name
    if "cell_z" not in mol.columns:
        return ann
    bregma = mol.loc[ann, "cell_z"].iloc[0]
    return np.isclose(mol["cell_z"].to_numpy(dtype=float), bregma)


def select_slice(mol, slice_name, filter_z0=None, verbose=True, policy=None):
    """
    Extract molecules for a single slice with local coordinates.

    Spatial coordinates are shifted so that the slice origin is at (0, 0).
    If ``filter_z0`` is True and multiple z-planes exist, only z=0 is kept.

    Parameters
    ----------
    mol : DataFrame
        Full molecule table.
    slice_name : str
        Value from the ``matched_h5ad_file`` column.
    filter_z0 : bool or None
        Keep only z=0 when multiple z-planes are present. ``None`` follows
        the policy (all planes for ``section``, z=0 for ``annotated_z0``).
    policy : str or None
        ``"section"`` or ``"annotated_z0"``; default :data:`MERFISH_INPUT`.
    verbose : bool
        Print slice summary.

    Returns
    -------
    mol_slice : DataFrame
        Filtered molecules with added ``x_local``, ``y_local`` columns.
    """
    policy = policy or MERFISH_INPUT
    if policy == "section":
        mol_slice = mol[section_rows(mol, slice_name)].copy()
        if filter_z0 is None:
            filter_z0 = False
    elif policy == "annotated_z0":
        mol_slice = mol[mol["matched_h5ad_file"] == slice_name].copy()
        if filter_z0 is None:
            filter_z0 = True
    else:
        raise ValueError(f"unknown MERFISH input policy {policy!r}")
    mol_slice["x_local"] = mol_slice["x"] - mol_slice["x"].min()
    mol_slice["y_local"] = mol_slice["y"] - mol_slice["y"].min()

    if filter_z0 and "z" in mol_slice.columns:
        if mol_slice["z"].nunique() > 1:
            mol_slice = mol_slice[mol_slice["z"] == 0].copy()

    if verbose:
        print(
            f"  Slice: {slice_name} — {len(mol_slice):,} molecules, "
            f"{mol_slice['gene'].nunique()} genes"
        )

    return mol_slice
