"""Label-free registration of the osmFISH annotation onto the molecule grid.

``run_osmFISH.evaluate_at_cell_level`` aligns the benchmark cell centroids to
the molecule grid by matching the two centroids. The molecule cloud extends
beyond the annotated cells asymmetrically (31.6k x 51.6k px against
23.9k x 48.3k px), so that alignment is biased: on 5 Sep 2026 it was found to
be off by about (107, -72) um.

This module recovers the translation without the domain labels, the same way
``benchmarks/segfree/osm_register.py`` does for the competitor benchmark: the
gene composition of the molecules within a cell radius of each centroid is
correlated with that cell's own count vector from the AnnData, and the
translation maximising the mean correlation over 1,500 sampled cells is kept.
The optimum is the same at 100, 150 and 250 px radii.

    python -m experiments.osmfish.register            # fits and writes
                                                      # data/results/osmfish/registration.json

The stored quantity is the full translation from the AnnData cell frame into
the molecule grid frame (molecule coordinates minus their minimum), so it does
not depend on the grid spacing; ``run_osmFISH.evaluate_at_cell_level`` reads it
and every osmFISH experiment inherits it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

GENE_FIXES = {"Kcnip": "Kcnip2", "Tmem6": "Tmem2"}


def composition_score(cells_xy, mol_xy, gcode, counts, n_genes, radius=150.0,
                      n=1500, seed=0, tree=None):
    """Mean per-cell Pearson correlation between the gene composition of the
    molecules within ``radius`` of a cell centroid and the cell's own counts."""
    tree = tree or cKDTree(mol_xy)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cells_xy), min(n, len(cells_xy)), replace=False)
    lists = tree.query_ball_point(cells_xy[pick], radius)
    lens = np.array([len(l) for l in lists])
    keep = lens >= 20
    if keep.sum() == 0:
        return float("nan"), 0
    ids = np.concatenate([lists[i] for i in np.where(keep)[0]])
    rows = np.repeat(np.arange(keep.sum()), lens[keep])
    g = gcode[ids]
    ok = g >= 0
    C = np.zeros((keep.sum(), n_genes))
    np.add.at(C, (rows[ok], g[ok]), 1.0)
    T = counts[pick[keep]].astype(float)
    good = (C.sum(1) > 0) & (T.sum(1) > 0)
    C, T = C[good], T[good]
    C = C / C.sum(1, keepdims=True); T = T / T.sum(1, keepdims=True)
    C = C - C.mean(1, keepdims=True); T = T - T.mean(1, keepdims=True)
    r = (C * T).sum(1) / (np.linalg.norm(C, axis=1) * np.linalg.norm(T, axis=1) + 1e-12)
    return float(np.nanmean(r)), int(good.sum())


def fit(mol, adata, coarse=300, fine=75, span=(-900, 2700, -2100, 900),
        radius=150.0, verbose=True):
    """Return ``(translation, r, r0)``: the translation (px) taking AnnData cell
    coordinates into the molecule grid frame (molecule coordinates minus their
    minimum, as ``io.select_slice`` builds it), the composition correlation it
    reaches, and the correlation of the centroid-matched alignment.

    ``mol`` is the molecule table from ``run_osmFISH.load_osmfish_molecules``
    (raw frame), ``adata`` the benchmark AnnData.
    """
    xy = mol[["x", "y"]].values
    xy = xy - xy.min(0)
    genes = list(adata.var_names)
    gi = {g: i for i, g in enumerate(genes)}
    gcode = mol["gene"].replace(GENE_FIXES).map(lambda g: gi.get(g, -1)).values
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    cells = adata.obsm["spatial"]
    centroid_t = xy.mean(0) - cells.mean(0)
    c0 = cells + centroid_t

    tree = cKDTree(xy)

    def s(dx, dy, n):
        return composition_score(c0 + [dx, dy], xy, gcode, X, len(genes), radius, n=n, tree=tree)[0]

    r0 = s(0, 0, 1500)
    # coarse pass on 400 cells, fine pass on 1500
    best = max(((s(dx, dy, 400), dx, dy) for dx in range(span[0], span[1] + 1, coarse)
                for dy in range(span[2], span[3] + 1, coarse)))
    _, dx0, dy0 = best
    best = max(((s(dx, dy, 1500), dx, dy) for dx in range(dx0 - coarse, dx0 + coarse + 1, fine)
                for dy in range(dy0 - coarse, dy0 + coarse + 1, fine)))
    r, dx, dy = best
    if verbose:
        print(f"  osmFISH registration: composition r = {r0:.3f} at centroid alignment, "
              f"{r:.3f} at shift ({dx}, {dy}) px = ({dx*0.065:.0f}, {dy*0.065:.0f}) um")
    return centroid_t + np.array([dx, dy], dtype=float), r, r0


def main():
    import json
    import anndata as ad
    from spsheaf import paths
    from experiments.osmfish import run_osmFISH as osm
    mol = osm.load_osmfish_molecules(str(paths.raw("osmfish") / "mRNA_coords_raw_counting.hdf5"))
    adata = ad.read_h5ad(str(paths.raw("osmfish") / "osmfish_20260512084934.h5ad"))
    translation, r, r0 = fit(mol, adata)
    out = paths.results("osmfish") / osm.REGISTRATION_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        translation=[float(t) for t in translation],
        composition_r=r, composition_r_centroid=r0, radius_px=150.0,
        note="cells_aligned = adata.obsm['spatial'] + translation, in the frame "
             "of molecule coordinates minus their minimum (run_osmFISH frame)"),
        indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
