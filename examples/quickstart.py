"""spSheaf quickstart: from a molecule table to spatial domains in one script.

Downloads one public section (the SpaceTx BaristaSeq mouse visual cortex
section, 0.32 million molecules, 81 genes, ~25 MB), runs the eight steps of
spSheaf at the settings used for it in the paper, draws the domains next to
the authors' layer annotation and prints the ARI. About one minute on a
laptop; no other data needed.

    pip install spsheaf matplotlib
    python examples/quickstart.py            # writes quickstart_domains.png

Every step is a plain function call so each intermediate (fibers, graph,
stalks, boundary field) can be inspected or replaced. Units are those of the
input coordinates (micrometres here).
"""
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

from spsheaf import clustering, evaluation, fibers, graph, sheaf

# ---------------------------------------------------------------- 1. data
BASE = "https://s3.amazonaws.com/starfish.data.spacetx/spacetx-website/data/BARISTASeq"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data_quickstart"
DATA.mkdir(exist_ok=True)
for name in ("s3_spot_table.csv", "s3_mapped_cell_table.csv"):
    if not (DATA / name).exists():
        print(f"downloading {name} ...")
        urlretrieve(f"{BASE}/{name}", DATA / name)

spots = pd.read_csv(DATA / "s3_spot_table.csv", usecols=["xc", "yc", "gene"])
mol = pd.DataFrame({"x": spots.xc.values, "y": spots.yc.values, "gene": spots.gene.astype(str).values})
print(f"{len(mol):,} molecules, {mol.gene.nunique()} genes")

# ---------------------------------------------------------------- 2. parameters
# the configuration selected for this section in the paper (Supplementary Table 3)
GRID_SPACING = 10.0     # um between fiber centres
BANDWIDTH = 40.0        # um, Gaussian kernel of the fibers
STALK_DIM = 9           # dimension of every local principal subspace
SMOOTH_RADIUS = 180.0   # um, smoothing of the disagreement field
SPATIAL_WEIGHT = 0.5    # alpha_s: weight of the coordinates in the clustering feature
BOUNDARY_ALPHA = 0.5    # alpha_b: strength of the attenuation at transitions
K = 6                   # number of domains (six annotated layers here)

# ---------------------------------------------------------------- 3. the eight steps
grid, raw, genes = fibers.build_kernel_fibers(mol, grid_spacing=GRID_SPACING, bandwidth=BANDWIDTH,
                                              x_col="x", y_col="y")            # (1) fibers on a grid
fib = fibers.normalize_fibers(raw)
edges = graph.build_spatial_graph(grid, k=6)                                     # (2) spatial graph
stalks = sheaf.compute_stalks(fib, grid, uniform_dim=STALK_DIM, k_nbrs=50)      # (3) stalks: local PCA bases
rs = sheaf.compute_restriction_maps(stalks, edges, STALK_DIM)                    # (4) restriction maps by SVD
b_raw, b_smooth = sheaf.compute_boundary_scores(fib, grid, stalks, edges, rs,
                                                smooth_radius=SMOOTH_RADIUS)     # (5-6) disagreement, field
labels = clustering.boundary_aware_clustering(fib, grid, b_smooth, edges, n_clusters=K,
                                              spatial_weight=SPATIAL_WEIGHT,
                                              boundary_alpha=BOUNDARY_ALPHA)    # (7) attenuated clustering
labels, _ = clustering.spatial_cleanup(labels, edges, grid_coords=grid)         # (8) remove fragments

# ---------------------------------------------------------------- 4. against the authors' annotation
cells = pd.read_csv(DATA / "s3_mapped_cell_table.csv")
cxy = cells[["rotated_x", "rotated_y"]].values
cxy = cxy - cxy.min(0) + grid.min(0)      # cell frame onto the grid frame by their lower-left corners
from scipy.spatial import cKDTree
_, idx = cKDTree(grid).query(cxy, k=1)
pred = labels[idx]
names = sorted(cells.layer.dropna().unique())
gti = np.array([names.index(l) if isinstance(l, str) else -1 for l in cells.layer])
valid = gti >= 0
ari = evaluation.compute_ari(pred, gti, valid)
print(f"ARI against the layer annotation: {ari:.3f}  (the paper reports 0.757 at this configuration "
      "with a label-free registration of the cell frame; a corner alignment is used here)")

# ---------------------------------------------------------------- 5. picture
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
axes[0].scatter(cxy[valid, 0], cxy[valid, 1], c=gti[valid], cmap="tab10", s=2, linewidths=0)
axes[0].set_title("annotation (cells)")
axes[1].scatter(grid[:, 0], grid[:, 1], c=b_smooth, cmap="magma", s=5, marker="s", linewidths=0)
axes[1].set_title("sheaf disagreement field")
axes[2].scatter(grid[:, 0], grid[:, 1], c=labels, cmap="tab10", s=5, marker="s", linewidths=0)
axes[2].set_title(f"spSheaf domains, k = {K}")
for ax in axes:
    ax.set_aspect("equal"); ax.set_axis_off()
fig.tight_layout()
fig.savefig(HERE / "quickstart_domains.png", dpi=150)
print(f"wrote {HERE / 'quickstart_domains.png'}")
