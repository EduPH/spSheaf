"""Shared utilities for the segmentation-free domain-inference benchmark.

Metric definitions follow the SDMBench implementation
(Yuan et al. 2024, SDMBench; https://github.com/zhaofangyuan98/SDMBench)
so numbers are directly comparable to the published benchmark table.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             homogeneity_score, completeness_score, silhouette_score)

from spsheaf import paths as _paths
MER_DIR = str(_paths.data_root() / "raw" / "merfish") + "/"
MER_SLICES = ["0.04", "0.09", "0.14", "0.19", "0.24"]
MER_MOL = MER_DIR + "moffitt2018_hypothalamus_molecules.csv.gz"


# ---------------------------------------------------------------- metrics
# The SDMBench metric panel lives in spsheaf.evaluation and is imported here so
# that spSheaf and its comparators are scored by exactly one implementation.
# The previous local copies of CHAOS/PAS/ASW were verified equivalent to these
# (tests/test_metric_parity.py, 25 cases, agreement to 1e-9) before removal.
from spsheaf.evaluation import (          # noqa: E402
    compute_chaos as compute_CHAOS,
    compute_pas as compute_PAS,
    compute_asw as compute_ASW,
)


def all_metrics(gt, pred, location):
    """SDMBench metric panel. gt/pred are 1-D label arrays, location is (n,2)."""
    gt = np.asarray(gt).astype(str)
    pred = np.asarray(pred).astype(str)
    return dict(
        ARI=float(adjusted_rand_score(gt, pred)),
        NMI=float(normalized_mutual_info_score(gt, pred)),
        HOM=float(homogeneity_score(gt, pred)),
        COM=float(completeness_score(gt, pred)),
        CHAOS=float(compute_CHAOS(pred, location)),
        PAS=float(compute_PAS(pred, location)),
        ASW=compute_ASW(pred, location),
        n_pred_domains=int(len(np.unique(pred))),
    )


# ---------------------------------------------------------------- data loading
MER_EXPORT = MER_DIR + "moffitt2018_export_full/molecules_full_add.csv.gz"
MER_EXPORT_COLS = ["gene", "x", "y", "z", "cell_id", "cell_x", "cell_y", "cell_z",
                   "matched_h5ad_file", "domain"]


def load_merfish_export():
    """The Moffitt molecule export with the SDMBench slice/domain columns."""
    import os
    if not os.path.exists(MER_EXPORT):
        raise FileNotFoundError(MER_EXPORT)
    parts = [ch for ch in pd.read_csv(MER_EXPORT, usecols=MER_EXPORT_COLS,
                                      chunksize=3_000_000, low_memory=False)]
    return pd.concat(parts, ignore_index=True)


def load_merfish_section(slice_id):
    """Only the rows of one physical section (bregma = -slice_id), read in
    chunks so the 20.5 M-row export is never held in memory at once."""
    bregma = -float(slice_id)
    parts = []
    for ch in pd.read_csv(MER_EXPORT, usecols=MER_EXPORT_COLS, chunksize=3_000_000, low_memory=False):
        parts.append(ch[np.isclose(ch.cell_z, bregma)])
    return pd.concat(parts, ignore_index=True)


class _Cells:
    """The three attributes of the SDMBench AnnData the runners read."""
    def __init__(self, cells):
        self.obs_names = cells.index
        self.obs = cells[["domain"]]
        self.obsm = {"spatial": cells[["cell_x", "cell_y"]].values}


def load_merfish_slice(slice_id, pad=60.0, molecules=None, legacy_bbox=False):
    """Return (cells, molecules of the physical section).

    ``molecules`` is the export of :func:`load_merfish_export` (or None to
    load it). The section is selected by its bregma coordinate
    (``cell_z``): the export lays the twelve bregma sections of animal 1 out
    in overlapping xy blocks, so the bounding box of the annotated cells,
    which the benchmark used before 8 Sep 2026 (``legacy_bbox=True``), also
    captures a second, unrelated section (bregma 0.11 lies on top of -0.19,
    0.16 on -0.14, 0.21 on -0.09, 0.26 on -0.04, 0.01 on -0.24) and every
    competitor MERFISH number computed that way was scored on two sections
    superimposed. All focal planes are kept. Cells are those of the
    annotation (``matched_h5ad_file``), scored as before.
    """
    if molecules is None:
        molecules = load_merfish_export()
    name = f"MERFISH_{slice_id}.h5ad"
    ann = molecules[molecules.matched_h5ad_file == name]
    if len(ann) == 0:
        raise ValueError(f"no molecules matched to {name}")
    if legacy_bbox:
        cx, cy = ann.cell_x, ann.cell_y
        m = molecules[(molecules.x >= cx.min() - pad) & (molecules.x <= cx.max() + pad) &
                      (molecules.y >= cy.min() - pad) & (molecules.y <= cy.max() + pad)]
    else:
        m = molecules[molecules.cell_z == ann.cell_z.iloc[0]]
    cells = (ann.drop_duplicates("cell_id").set_index("cell_id")[["cell_x", "cell_y", "domain"]])
    cells = cells[cells.domain.notna()]
    return _Cells(cells), m.copy()


def nearest_pixel_labels(label_img, cell_xy, x0, y0, scale=1.0):
    """Sample a 2-D pixel label image at cell centroid positions.

    label_img[i, j] is indexed by (x, y) pixel; cell coordinates are shifted by
    (x0, y0) and divided by `scale` (um per pixel) before rounding.
    """
    xi = np.clip(np.round((cell_xy[:, 0] - x0) / scale).astype(int), 0, label_img.shape[0] - 1)
    yi = np.clip(np.round((cell_xy[:, 1] - y0) / scale).astype(int), 0, label_img.shape[1] - 1)
    return np.asarray(label_img)[xi, yi]


def fill_background(labels, cell_xy, bg=-1):
    """Assign background/unlabelled cells to their nearest labelled cell.

    Domain-inference metrics require a label for every cell; leaving a
    method's background as its own pseudo-domain would penalise methods that
    mask low-density pixels (SSAM, Points2Regions) for a reason unrelated to
    domain quality. Nearest-labelled-neighbour transfer is the same
    convention used for grid-to-cell transfer of the other methods.
    """
    labels = np.asarray(labels).copy()
    bad = labels == bg
    if bad.all() or not bad.any():
        return labels
    from scipy.spatial import cKDTree
    tree = cKDTree(cell_xy[~bad])
    _, j = tree.query(cell_xy[bad])
    labels[bad] = labels[~bad][j]
    return labels
