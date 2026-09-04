#!/usr/bin/env python
"""
Fig. 1 pipeline panels on the synthetic two-region construction.

Same construction as ``sheaf_vs_kernel.py``: a square grid split into two
regions whose fibers have identical mean and identical neighbour-distance
statistics, differing only in the *subspace* along which expression varies
(region A in feature dims 0-3, region B in orthogonal dims 8-11). Every stage
of the sheaf pipeline is run on it and saved as a clean, standalone panel so a
Fig. 1 schematic can be composed manually:

  00_ground_truth      the two latent regions actually present in the data
  01_fibers            high-dimensional fibers on the grid (active-channel RGB)
  02_graph             spatial k-NN graph over grid points
  03_stalks            local-PCA stalk orientation (subspace) field
  04_disagreement      restriction-map misalignment across edges (raw D)
  05_boundary          smoothed sheaf boundary field
  06_domains           boundary-aware clustering -> two regions recovered

Kernel-vs-sheaf contrast (split out from the composite figure):
  kernel_mean_expression       mean intensity  -> kernel sees NO boundary
  kernel_neighbour_distance    ||f_u - f_v||    -> kernel sees NO boundary
  sheaf_stalk_orientation      local subspace   -> two regions appear
  sheaf_principal_angle_hist   subspaces aligned within, orthogonal across

The sheaf disagreement here is the restriction-map misalignment
1 - mean cos(principal angle) between neighbouring stalk bases -- the sheaf
difference operator itself. It fires at the interface (orthogonal subspaces =>
90 deg) even though the coefficients are i.i.d. everywhere, which is precisely
why the kernel/intensity views stay blind.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from spsheaf import graph, sheaf, clustering

OUT = os.path.join(os.path.dirname(__file__) or ".", "pipeline_panels")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- construction
rng = np.random.default_rng(0)
G, d, n, sigma = 16, 4, 40, 1.0
xs, ys = np.meshgrid(np.arange(n), np.arange(n))
grid = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
region = (grid[:, 0] >= n / 2).astype(int)               # 0 = A (left), 1 = B (right)

PA = np.zeros((G, d)); PA[0:d] = np.eye(d)               # region A subspace: dims 0-3
PB = np.zeros((G, d)); PB[8:8 + d] = np.eye(d)           # region B subspace: dims 8-11
Z = rng.normal(scale=sigma, size=(len(grid), d))         # i.i.d. coefficients everywhere
fib = np.where(region[:, None] == 0, Z @ PA.T, Z @ PB.T)
fib += rng.normal(scale=0.05, size=fib.shape)

# --------------------------------------------------------------- pipeline stages
edges = graph.build_spatial_graph(grid, k=6, verbose=False)
stalks = sheaf.compute_stalks(fib, grid, uniform_dim=d, k_nbrs=25, verbose=False)

# stalk orientation vs a fixed region-A reference subspace (per-point scalar)
Uref = np.linalg.qr(PA)[0][:, :d]
orient = np.array([np.linalg.svd(U.T @ Uref, compute_uv=False).mean() for U in stalks])

# restriction-map misalignment per edge = 1 - mean cos(principal angle)
e_arr = np.array(edges)
mis = np.array([1.0 - np.linalg.svd(stalks[u].T @ stalks[v], compute_uv=False).mean()
                for u, v in edges])
# aggregate edge disagreement to nodes (mean over incident edges) -> raw field
dis = np.zeros(len(grid)); deg = np.zeros(len(grid))
for (u, v), m in zip(edges, mis):
    dis[u] += m; dis[v] += m; deg[u] += 1; deg[v] += 1
dis /= np.maximum(deg, 1)
# smooth the node disagreement in space -> boundary field
tree = cKDTree(grid); rad = 2.0
boundary = np.array([dis[tree.query_ball_point(p, rad)].mean() for p in grid])

# boundary-aware domains: the smoothed sheaf boundary field forms a high ridge
# along the interface. Remove that ridge, take connected components of the two
# interiors, then backfill the ridge nodes to their nearest interior. On this
# construction the fiber mean is identical across regions, so an intensity/PCA
# split cannot separate them; the sheaf boundary field can.
from collections import deque
def watershed(boundary_field, keep_pct):
    keep = boundary_field <= np.percentile(boundary_field, keep_pct)
    adj = [[] for _ in range(len(grid))]
    for u, v in edges:
        if keep[u] and keep[v]:
            adj[u].append(v); adj[v].append(u)
    lab = -np.ones(len(grid), dtype=int); cid = 0
    for s in np.where(keep)[0]:
        if lab[s] != -1:
            continue
        q = deque([s]); lab[s] = cid
        while q:
            nd = q.popleft()
            for nb in adj[nd]:
                if lab[nb] == -1:
                    lab[nb] = cid; q.append(nb)
        cid += 1
    return lab, keep

# pick the keep-fraction that yields exactly two sizeable interiors
for kp in (85, 80, 78, 75, 70, 65):
    lab, keep = watershed(boundary, kp)
    big = [c for c in set(lab[keep]) if (lab == c).sum() >= 0.1 * keep.sum()]
    if len(big) == 2:
        break
# backfill ridge nodes to nearest kept-interior label
kept_idx = np.where(keep & np.isin(lab, big))[0]
ktree = cKDTree(grid[kept_idx])
labels = lab.copy()
fillmask = ~np.isin(labels, big)
if fillmask.any():
    _, jj = ktree.query(grid[fillmask])
    labels[fillmask] = lab[kept_idx[jj]]
# relabel by descending size for stable colors
order = {c: i for i, c in enumerate(
    sorted(set(labels), key=lambda c: -(labels == c).sum()))}
labels = np.array([order[c] for c in labels])

# --------------------------------------------------------------- kernel baselines
meanexpr = fib.mean(1)
naive = np.zeros(len(grid)); degn = np.zeros(len(grid))
for u, v in edges:
    dd = np.linalg.norm(fib[u] - fib[v]); naive[u] += dd; naive[v] += dd
    degn[u] += 1; degn[v] += 1
naive /= np.maximum(degn, 1)

# principal angles between random stalk pairs, same- vs different-region
pairs = rng.integers(0, len(grid), size=(4000, 2)); pairs = pairs[pairs[:, 0] != pairs[:, 1]]
cos_pair = np.array([np.linalg.svd(stalks[i].T @ stalks[j], compute_uv=False).mean()
                     for i, j in pairs])
same = region[pairs[:, 0]] == region[pairs[:, 1]]

# ------------------------------------------------------------------- rendering
SZ = 3.3
def field(name, c, cmap="magma", vmin=None, vmax=None, s=30):
    fig, ax = plt.subplots(figsize=(SZ, SZ))
    ax.scatter(grid[:, 0], grid[:, 1], c=c, s=s, cmap=cmap, vmin=vmin, vmax=vmax,
               marker="s", linewidths=0)
    ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight",
                transparent=True); plt.close(fig)

# 00 ground truth
field("00_ground_truth.png", region, cmap="coolwarm")

# 01 fibers: RGB of active-channel energy (A -> red, B -> teal); shows fibers
# carry the distinction that the scalar kernel views throw away
aE = np.linalg.norm(fib[:, 0:d], axis=1); bE = np.linalg.norm(fib[:, 8:8 + d], axis=1)
aE = aE / aE.max(); bE = bE / bE.max()
rgb = np.column_stack([aE, 0.55 * bE, bE])               # red = A energy, teal = B energy
fig, ax = plt.subplots(figsize=(SZ, SZ))
ax.scatter(grid[:, 0], grid[:, 1], c=np.clip(rgb, 0, 1), s=30, marker="s", linewidths=0)
ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(OUT, "01_fibers.png"), dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)

# 02 spatial graph (zoom to a small window so the k-NN lattice is legible)
lo, hi = 14, 26
win = (grid[:, 0] >= lo) & (grid[:, 0] <= hi) & (grid[:, 1] >= lo) & (grid[:, 1] <= hi)
segs = [grid[[u, v]] for u, v in edges if win[u] and win[v]]
fig, ax = plt.subplots(figsize=(SZ, SZ))
ax.add_collection(LineCollection(segs, colors="0.6", linewidths=0.9, alpha=0.85))
ax.scatter(grid[win, 0], grid[win, 1], s=45, c="steelblue", zorder=3, linewidths=0)
ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(lo - 0.5, hi + 0.5)
ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout(pad=0.1)
fig.savefig(os.path.join(OUT, "02_graph.png"), dpi=220, bbox_inches="tight", transparent=True)
plt.close(fig)

# 03 stalk orientation (sheaf subspace field)
field("03_stalks.png", orient, cmap="coolwarm", vmin=0, vmax=1)
# 04 raw restriction-map disagreement
field("04_disagreement.png", dis, cmap="magma")
# 05 smoothed boundary field
field("05_boundary.png", boundary, cmap="magma")
# 06 recovered domains
field("06_domains.png", labels, cmap="coolwarm")

# split kernel-vs-sheaf panels
field("kernel_mean_expression.png", meanexpr, cmap="viridis")
field("kernel_neighbour_distance.png", naive, cmap="magma")
field("sheaf_stalk_orientation.png", orient, cmap="coolwarm", vmin=0, vmax=1)

fig, ax = plt.subplots(figsize=(SZ + 0.6, SZ))
bins = np.linspace(0, 1, 25)
ax.hist(cos_pair[same], bins=bins, alpha=0.75, density=True, color="steelblue", label="same region")
ax.hist(cos_pair[~same], bins=bins, alpha=0.75, density=True, color="firebrick", label="different regions")
ax.set_xlabel("mean cos(principal angle) between stalks")
ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sheaf_principal_angle_hist.png"), dpi=220,
            bbox_inches="tight", transparent=True); plt.close(fig)

# --------------------------------------------------------------------- report
from sklearn.metrics import adjusted_rand_score
def contrast(f):
    fe = (f[e_arr[:, 0]] + f[e_arr[:, 1]]) / 2
    cr = region[e_arr[:, 0]] != region[e_arr[:, 1]]
    return fe[cr].mean() / max(fe[~cr].mean(), 1e-9)
print(f"mean expression by region     : {meanexpr[region==0].mean():+.3f} vs {meanexpr[region==1].mean():+.3f}")
print(f"neighbour-distance contrast    : {contrast(naive):.3f}   (~1 => kernel blind)")
print(f"disagreement (subspace) contrast: {contrast(dis):.3f}   (>>1 => sheaf sees interface)")
print(f"stalk orientation  A={orient[region==0].mean():.2f}  B={orient[region==1].mean():.2f}")
print(f"stalk-pair cos  same={cos_pair[same].mean():.2f}  diff={cos_pair[~same].mean():.2f}")
print(f"recovered domains ARI vs region: {adjusted_rand_score(region, labels):.3f}")
print(f"-> panels in {OUT}")
