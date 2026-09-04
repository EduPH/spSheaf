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
def load_merfish_slice(slice_id, pad=60.0, molecules=None):
    """Return (adata, molecule DataFrame restricted to that slice, origin offset).

    The molecule table is the raw Moffitt et al. molecule export; slice
    membership is recovered from the bounding box of the annotated cells,
    which is unambiguous because slices occupy disjoint coordinate blocks.
    """
    import anndata as ad
    a = ad.read_h5ad(f"{MER_DIR}MERFISH_{slice_id}.h5ad")
    xy = a.obsm["spatial"]
    if molecules is None:
        molecules = pd.read_csv(MER_MOL)
    m = molecules[(molecules.x >= xy[:, 0].min() - pad) & (molecules.x <= xy[:, 0].max() + pad) &
                  (molecules.y >= xy[:, 1].min() - pad) & (molecules.y <= xy[:, 1].max() + pad)]
    return a, m.copy()


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
