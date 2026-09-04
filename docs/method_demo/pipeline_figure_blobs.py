#!/usr/bin/env python
"""
Harder Fig. 1 synthetic: several irregular blobs on a background, each blob a
distinct expression *subspace*, run through the full sheaf pipeline.

Same kernel-blind construction as the two-region demo, generalized to K+1
regions: a background plus several circular/elliptical blobs (different sizes,
one deliberately small and hard). Every region gets its own orthogonal d-dim
subspace of the G-dim feature space, with i.i.d. coefficients everywhere, so:

  * mean expression and neighbour distance are identical across every region
    (the kernel / intensity view is blind), while
  * the sheaf disagreement (restriction-map misalignment, 1 - cos principal
    angle between neighbouring stalk bases) lights up every blob boundary.

Domains are then recovered by clustering the stalks on their local subspace
(k-means on the stalk projectors U U^T), followed by a connectivity cleanup.
"""
import os
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from spsheaf import graph, sheaf, clustering

OUT = os.path.join(os.path.dirname(__file__) or ".", "pipeline_panels_blobs")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(3)

# ------------------------------------------------------------ geometry (blobs)
n, G, d = 48, 30, 5
xs, ys = np.meshgrid(np.arange(n), np.arange(n))
grid = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
region = np.zeros(len(grid), dtype=int)                 # 0 = background
# (cx, cy, rx, ry) ; later blobs overwrite earlier where they overlap
blobs = [
    (12, 13, 8, 8),      # 1  large circle
    (34, 15, 9, 6),      # 2  wide ellipse
    (25, 35, 9, 9),      # 3  large circle (near blob 2 -> shared-ish border)
    (12, 36, 4, 4),      # 4  SMALL circle (hard case)
    (39, 37, 6, 5),      # 5  medium ellipse
]
for k, (cx, cy, rx, ry) in enumerate(blobs, start=1):
    inside = ((grid[:, 0] - cx) / rx) ** 2 + ((grid[:, 1] - cy) / ry) ** 2 <= 1.0
    region[inside] = k
K = len(blobs) + 1

# ------------------------------------------------- fibers: per-region subspace
# exact orthogonal blocks (K*d <= G) -> kernel-blind; random sign/rotation of Z
subspaces = []
for k in range(K):
    P = np.zeros((G, d)); P[k * d:(k + 1) * d] = np.eye(d)
    subspaces.append(P)
Z = rng.normal(size=(len(grid), d))
fib = np.zeros((len(grid), G))
for k in range(K):
    m = region == k
    fib[m] = Z[m] @ subspaces[k].T
fib += rng.normal(scale=0.12, size=fib.shape)           # more noise = harder

# ------------------------------------------------------------- pipeline stages
edges = graph.build_spatial_graph(grid, k=6, verbose=False)
e_arr = np.array(edges)
stalks = sheaf.compute_stalks(fib, grid, uniform_dim=d, k_nbrs=25, verbose=False)

mis = np.array([1.0 - np.linalg.svd(stalks[u].T @ stalks[v], compute_uv=False).mean()
                for u, v in edges])
dis = np.zeros(len(grid)); deg = np.zeros(len(grid))
for (u, v), m in zip(edges, mis):
    dis[u] += m; dis[v] += m; deg[u] += 1; deg[v] += 1
dis /= np.maximum(deg, 1)
tree = cKDTree(grid); rad = 1.8
boundary = np.array([dis[tree.query_ball_point(p, rad)].mean() for p in grid])

# domains: cluster the stalks by their local subspace. Each stalk U_i defines a
# projector P_i = U_i U_i^T; points sharing a subspace have near-identical
# projectors, so k-means on the projectors groups the grid into domains by
# subspace. The background stays a single domain (no pinching), and even the
# small blob is recovered -- the sheaf signal (local subspaces) drives the
# clustering where an intensity/PCA view of the fibers cannot. A connectivity
# cleanup removes stray boundary points.
Pr = np.array([(U @ U.T).ravel() for U in stalks])
feat = PCA(n_components=25, random_state=0).fit_transform(Pr)
labels_raw = KMeans(K, n_init=10, random_state=0).fit_predict(feat)
labels_clean, _ = clustering.spatial_cleanup(labels_raw, edges, grid_coords=grid, verbose=False)

# align predicted labels to the ground-truth region ids (Hungarian on the
# overlap matrix) so a region keeps the same colour in every panel
def align_to_region(pred):
    C = np.zeros((K, K), int)
    for p, r in zip(pred, region):
        C[int(p), int(r)] += 1
    rows, cols = linear_sum_assignment(-C)
    m = {int(a): int(b) for a, b in zip(rows, cols)}
    return np.array([m[int(p)] for p in pred])
labels = align_to_region(labels_clean)
labels_raw_col = align_to_region(labels_raw)

# exponential boundary attenuation used by the boundary-aware clustering:
# interior points (low boundary) keep weight ~1, boundary points are damped.
alpha = 0.5
w_att = np.exp(-alpha * boundary / (np.median(boundary) + 1e-9))

# ------------------------------------------------------------ kernel baselines
meanexpr = fib.mean(1)
naive = np.zeros(len(grid)); degn = np.zeros(len(grid))
for u, v in edges:
    dd = np.linalg.norm(fib[u] - fib[v]); naive[u] += dd; naive[v] += dd
    degn[u] += 1; degn[v] += 1
naive /= np.maximum(degn, 1)
pairs = rng.integers(0, len(grid), size=(5000, 2)); pairs = pairs[pairs[:, 0] != pairs[:, 1]]
cos_pair = np.array([np.linalg.svd(stalks[i].T @ stalks[j], compute_uv=False).mean()
                     for i, j in pairs])
same = region[pairs[:, 0]] == region[pairs[:, 1]]

# ------------------------------------------------------------------ rendering
SZ = 3.3
def field(name, c, cmap="magma", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(SZ, SZ))
    ax.scatter(grid[:, 0], grid[:, 1], c=c, s=22, cmap=cmap, vmin=vmin, vmax=vmax,
               marker="s", linewidths=0)
    ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)

field("00_ground_truth.png", region, cmap="tab10", vmin=0, vmax=9)

# 01 fibers: honest raw view -- three random feature channels as RGB. The
# domain structure is invisible to any such ordinary view of the data (it lives
# in *which* subspace varies, not in any fixed set of channels); only the sheaf
# comparison of local subspaces recovers it.
Wproj = rng.normal(size=(G, 3))
proj = fib @ Wproj
lo3, hi3 = np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)
rgb = np.clip((proj - lo3) / (hi3 - lo3 + 1e-9), 0, 1)
fig, ax = plt.subplots(figsize=(SZ, SZ))
ax.scatter(grid[:, 0], grid[:, 1], c=rgb, s=22, marker="s", linewidths=0)
ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(OUT, "01_fibers.png"), dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)

# 02 graph zoom
lo, hi = 18, 30
win = (grid[:, 0] >= lo) & (grid[:, 0] <= hi) & (grid[:, 1] >= lo) & (grid[:, 1] <= hi)
segs = [grid[[u, v]] for u, v in edges if win[u] and win[v]]
fig, ax = plt.subplots(figsize=(SZ, SZ))
ax.add_collection(LineCollection(segs, colors="0.6", linewidths=0.9, alpha=0.85))
ax.scatter(grid[win, 0], grid[win, 1], s=45, c="steelblue", zorder=3, linewidths=0)
ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(lo - 0.5, hi + 0.5)
ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(OUT, "02_graph.png"), dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)

field("03_disagreement.png", dis, cmap="magma")
field("04_boundary.png", boundary, cmap="magma")
field("04b_attenuation.png", w_att, cmap="viridis")
field("05_kmeans_raw.png", labels_raw_col, cmap="tab10", vmin=0, vmax=9)
field("05_domains.png", labels, cmap="tab10", vmin=0, vmax=9)
field("kernel_mean_expression.png", meanexpr, cmap="viridis")
field("kernel_neighbour_distance.png", naive, cmap="magma")

fig, ax = plt.subplots(figsize=(SZ + 0.6, SZ))
bins = np.linspace(0, 1, 25)
ax.hist(cos_pair[same], bins=bins, alpha=0.75, density=True, color="steelblue", label="same region")
ax.hist(cos_pair[~same], bins=bins, alpha=0.75, density=True, color="firebrick", label="different regions")
ax.set_xlabel("mean cos(principal angle) between stalks"); ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sheaf_principal_angle_hist.png"), dpi=220,
            bbox_inches="tight", transparent=True); plt.close(fig)

# --------------------------------------------------------------------- report
def contrast(f):
    fe = (f[e_arr[:, 0]] + f[e_arr[:, 1]]) / 2
    cr = region[e_arr[:, 0]] != region[e_arr[:, 1]]
    return fe[cr].mean() / max(fe[~cr].mean(), 1e-9)
sizes = [int((region == k).sum()) for k in range(K)]
print(f"regions (bg + {len(blobs)} blobs), sizes: {sizes}")
print(f"neighbour-distance contrast    : {contrast(naive):.3f}   (~1 => kernel blind)")
print(f"disagreement (subspace) contrast: {contrast(dis):.3f}")
print(f"stalk-pair cos  same={cos_pair[same].mean():.2f}  diff={cos_pair[~same].mean():.2f}")
print(f"recovered domains: {len(set(labels))} (true {K}); ARI vs regions: {adjusted_rand_score(region, labels):.3f}")
print(f"-> panels in {OUT}")
