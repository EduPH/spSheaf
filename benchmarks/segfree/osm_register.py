"""Register the osmFISH molecule table onto the benchmark cell coordinates.

Why this step exists. The SDMBench osmFISH AnnData carries cell centroids in
the coordinate frame of the published loom file (verified below: correlation
1.0 and zero offset against osmFISH_SScortex_mouse_all_cells.loom, with 100%
agreement of the Region annotation). The molecule table distributed for
segmentation-free reanalysis is in the raw imaging frame, which is both shifted
and flipped on both axes relative to that frame. Any comparison that feeds
molecules to SSAM / Points2Regions and then scores against the annotated cells
must undo that transform first, otherwise every method is scored against
misaligned ground truth.

The transform is recovered without using the domain labels: candidate
orientations are ranked by how well the gene composition of the molecules
within a radius of a cell centroid reproduces that cell's own count vector from
the loom matrix. The chosen transform reaches mean per-cell composition
correlation 0.83 at a 60 um radius (n = 1380 cells with >= 20 nearby
molecules), versus 0.07 for the unflipped alignment.

Outputs, in the molecule frame:
    cell_xy_all   (6471, 2)  every loom cell
    cell_xy       (4839, 2)  the benchmark cells, in AnnData row order
"""
import numpy as np
import pandas as pd

from spsheaf import paths as _paths
OSM_DIR = str(_paths.data_root() / "raw" / "osmfish") + "/"
MOL_CSV = OSM_DIR + "mRNA_coords_raw_counting.csv"
H5AD = OSM_DIR + "osmfish_remove_excluded.h5ad"
LOOM = "osmrepo/osmFISH_SScortex_mouse_all_cells.loom"

# Recovered by the label-free grid search in osm_register_fit(); both axes are
# flipped, then a residual translation is applied.
RESIDUAL_OFFSET = np.array([-165.0, -785.0])

# The molecule table uses two legacy gene labels that the osmFISH paper's
# supplement corrects; SSAM's own notebook applies the same two renames.
GENE_FIXES = {"Kcnip": "Kcnip2", "Tmem6": "Tmem2"}
# Hybridisation rounds excluded as low quality by the osmFISH supplement. The
# distributed CSV has already had these dropped (it carries 35 gene labels with
# no Hybridization suffix), so nothing to filter here; kept for the record.
BLACKLIST = ["Cnr1_Hybridization4", "Plp1_Hybridization4", "Vtn_Hybridization4",
             "Klk6_Hybridization5", "Lum_Hybridization9", "Tbr1_Hybridization11"]


def load_molecules():
    om = pd.read_csv(MOL_CSV)
    om["gene"] = om["gene"].replace(GENE_FIXES)
    return om


def load_cells():
    """Benchmark AnnData plus loom cell coordinates, in the molecule frame."""
    import anndata as ad
    import h5py
    o = ad.read_h5ad(H5AD)
    with h5py.File(LOOM, "r") as f:
        LX, LY = f["col_attrs/X"][:], f["col_attrs/Y"][:]
        cid = np.array([s.decode() for s in f["col_attrs/CellID"][:]])
        region = np.array([s.decode() for s in f["col_attrs/Region"][:]])
        counts = f["matrix"][:]
        genes_loom = np.array([s.decode() for s in f["row_attrs/Gene"][:]])
    Lall = np.c_[LX, LY]
    pos = pd.Series(np.arange(len(cid)), index=cid)
    row_idx = pos[[c.replace("cell_", "") for c in o.obs_names]].values

    # sanity: the AnnData coordinates must be the loom coordinates
    assert np.allclose(o.obsm["spatial"], Lall[row_idx]), "AnnData/loom coordinate mismatch"
    assert (region[row_idx] == o.obs["Region"].astype(str).values).all(), "Region mismatch"

    om = load_molecules()
    mx1, my1 = om.x.max(), om.y.max()
    flipped = np.c_[mx1 - (Lall[:, 0] - Lall[:, 0].min()),
                    my1 - (Lall[:, 1] - Lall[:, 1].min())]
    cell_xy_all = flipped + RESIDUAL_OFFSET
    return o, om, cell_xy_all, cell_xy_all[row_idx], counts, genes_loom, row_idx


def composition_score(cell_xy, om, counts, genes_loom, radius=60.0, n=1500, seed=7):
    """Mean per-cell correlation between local molecule composition and loom counts."""
    from scipy.spatial import cKDTree
    gi = {g: i for i, g in enumerate(genes_loom)}
    gcode = om.gene.map(lambda g: gi.get(g, -1)).values
    tree = cKDTree(om[["x", "y"]].values)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cell_xy), min(n, len(cell_xy)), replace=False)
    cors = []
    for k, ids in zip(pick, tree.query_ball_point(cell_xy[pick], radius)):
        if len(ids) < 20:
            continue
        gc = gcode[ids]
        gc = gc[gc >= 0]
        c = np.bincount(gc, minlength=len(genes_loom)).astype(float)
        t = counts[:, k].astype(float)
        if c.sum() == 0 or t.sum() == 0:
            continue
        cors.append(np.corrcoef(c / c.sum(), t / t.sum())[0, 1])
    return float(np.nanmean(cors)), len(cors)


if __name__ == "__main__":
    o, om, all_xy, cell_xy, counts, genes_loom, _ = load_cells()
    r, n = composition_score(all_xy, om, counts, genes_loom)
    print(f"molecules {len(om):,}  genes {om.gene.nunique()}  benchmark cells {o.shape[0]}")
    print(f"composition correlation at 60 um: {r:.3f} over {n} cells")
    print(f"benchmark cell bbox in molecule frame: {cell_xy.min(0).round(0)} .. {cell_xy.max(0).round(0)}")
    print(f"regions: {o.obs['Region'].nunique()}")
