#!/usr/bin/env python
"""Compose the multi-blob pipeline panels into a single overview figure:
a pipeline row (fibers -> graph -> disagreement -> boundary -> domains) with
stage arrows, plus a kernel-vs-sheaf contrast row."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(__file__) or "."
P = os.path.join(HERE, "pipeline_panels_blobs")
img = lambda n: mpimg.imread(os.path.join(P, n + ".png"))

fig = plt.figure(figsize=(16, 8.2))
fig.patch.set_facecolor("white")

def panel(rect, name, title, letter, fs=12):
    ax = fig.add_axes(rect)
    ax.imshow(img(name)); ax.axis("off")
    ax.set_title(title, fontsize=fs, pad=4)
    ax.text(-0.02, 1.05, letter, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="bottom", ha="right")
    return ax

def arrow(x0, x1, y):
    a = FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                        arrowstyle="-|>", mutation_scale=20, lw=1.8, color="0.35")
    fig.add_artist(a)

# ---- row 1 : pipeline -------------------------------------------------------
w, h, y = 0.135, 0.30, 0.55
xs = [0.035, 0.19, 0.345, 0.50, 0.655, 0.81]
names = ["00_ground_truth", "01_fibers", "02_graph",
         "03_disagreement", "04_boundary", "05_domains"]
titles = ["True domains\n(hidden from method)", "Fibers\n(input data)",
          "Spatial graph", "Sheaf disagreement $D(e)$",
          "Boundary field", "Domains  (ARI 0.89)"]
letters = list("abcdef")
for x, nm, ti, le in zip(xs, names, titles, letters):
    panel([x, y, w, h], nm, ti, le)
# pipeline arrows between fibers -> ... -> domains (skip the a->b gap; a is a
# reference, not a processing step)
ymid = y + h / 2
for i in range(1, 5):
    arrow(xs[i] + w + 0.001, xs[i + 1] - 0.001, ymid)
# a gentle divider note between (a) reference and (b) input
fig.text(xs[1] - 0.012, ymid, "≈", fontsize=20, color="0.6",
         ha="center", va="center")

fig.text(0.5, 0.95, "Cellular-sheaf spatial-domain pipeline on a synthetic multi-blob field",
         fontsize=16, fontweight="bold", ha="center")
fig.text(0.035, 0.885, "Pipeline", fontsize=13, fontweight="bold", color="0.25")

# ---- row 2 : kernel vs sheaf ------------------------------------------------
y2, h2 = 0.10, 0.30
panel([0.075, y2, 0.215, h2], "kernel_mean_expression",
      "Mean expression", "g")
panel([0.315, y2, 0.215, h2], "kernel_neighbour_distance",
      "Neighbour distance $\\|f_u-f_v\\|$", "h")
axh = fig.add_axes([0.63, y2 + 0.02, 0.31, h2 - 0.04])
axh.imshow(img("sheaf_principal_angle_hist")); axh.axis("off")
axh.set_title("Stalk principal angles", fontsize=12, pad=4)
axh.text(-0.02, 1.06, "i", transform=axh.transAxes, fontsize=15,
         fontweight="bold", va="bottom", ha="right")

# group labels + divider
fig.text(0.30, 0.435, "Kernel / intensity view — blind to the domains",
         fontsize=12.5, fontweight="bold", color="#8B1A1A", ha="center")
fig.text(0.785, 0.435, "Sheaf difference operator — recovers them",
         fontsize=12.5, fontweight="bold", color="#1F4E79", ha="center")
fig.add_artist(plt.Line2D([0.58, 0.58], [0.085, 0.46], transform=fig.transFigure,
                          color="0.8", lw=1.2, ls="--"))
fig.text(0.5, 0.485, "Why the sheaf, not the kernel:  identical mean expression and "
         "neighbour distances across regions, yet the local subspaces differ",
         fontsize=11.5, ha="center", color="0.25")

out = os.path.join(HERE, "fig1_blobs_overview.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print("->", out)
