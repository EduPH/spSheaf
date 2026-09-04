#!/usr/bin/env python
"""Full step-by-step overview of the cellular-sheaf pipeline on the multi-blob
synthetic: every stage from fibers to cleaned domains, including the sheaf
machinery (stalks, restriction maps), the boundary score, the exponential
attenuation weight and the connectivity cleanup, plus the kernel-vs-sheaf
justification strip."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(__file__) or "."
P = os.path.join(HERE, "pipeline_panels_blobs")
img = lambda n: mpimg.imread(os.path.join(P, n + ".png"))

fig = plt.figure(figsize=(19, 12.5))
fig.patch.set_facecolor("white")

def imgpanel(rect, name, title, letter, formula=""):
    ax = fig.add_axes(rect); ax.imshow(img(name)); ax.axis("off")
    ax.set_title(title, fontsize=11.5, pad=3)
    ax.text(-0.03, 1.06, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right")
    if formula:
        ax.text(0.5, -0.10, formula, transform=ax.transAxes, fontsize=9.2,
                color="0.25", ha="center", va="top")
    return ax

def blankpanel(rect, title, letter, formula=""):
    ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11.5, pad=3)
    ax.text(-0.03, 1.06, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right")
    if formula:
        ax.text(0.5, -0.10, formula, transform=ax.transAxes, fontsize=9.2,
                color="0.25", ha="center", va="top")
    return ax

def arrow(x0, x1, y):
    fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=20, lw=1.8, color="0.35"))

def line(x0, y0, x1, y1):
    fig.add_artist(plt.Line2D([x0, x1], [y0, y1], transform=fig.transFigure,
                   color="0.35", lw=1.8, solid_capstyle="round"))

# ----- schematic: local PCA stalk -------------------------------------------
def draw_stalk(ax):
    rng = np.random.default_rng(1)
    th = np.deg2rad(35)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    pts = (rng.normal(size=(16, 2)) * np.array([0.22, 0.07])) @ R.T + np.array([0.5, 0.5])
    ax.scatter(pts[:, 0], pts[:, 1], s=22, color="steelblue", zorder=2)
    c = np.array([0.5, 0.5])
    e1 = R @ np.array([1, 0]); e2 = R @ np.array([0, 1])
    ax.annotate("", c + 0.34 * e1, c, arrowprops=dict(arrowstyle="-|>", lw=2, color="firebrick"))
    ax.annotate("", c + 0.17 * e2, c, arrowprops=dict(arrowstyle="-|>", lw=2, color="darkorange"))
    ax.text(*(c + 0.40 * e1), "$u_1$", color="firebrick", fontsize=11, ha="center", va="center")
    ax.text(*(c + 0.24 * e2), "$u_2$", color="darkorange", fontsize=11, ha="center", va="center")

# ----- schematic: restriction map / principal angle -------------------------
def draw_restriction(ax):
    o = np.array([0.30, 0.34])
    ax.annotate("", o + [0.42, 0], o, arrowprops=dict(arrowstyle="-|>", lw=2, color="0.15"))
    ax.annotate("", o + [0, 0.42], o, arrowprops=dict(arrowstyle="-|>", lw=2, color="0.15"))
    ax.text(*(o + [0.45, 0.07]), "$U_u$", fontsize=11, color="0.15", va="center")
    th = np.deg2rad(42)
    d1 = np.array([np.cos(th), np.sin(th)]); d2 = np.array([-np.sin(th), np.cos(th)])
    ax.annotate("", o + 0.42 * d1, o, arrowprops=dict(arrowstyle="-|>", lw=2, color="steelblue"))
    ax.annotate("", o + 0.42 * d2, o, arrowprops=dict(arrowstyle="-|>", lw=2, color="steelblue"))
    ax.text(*(o + 0.47 * d1), "$U_v$", fontsize=11, color="steelblue", va="center")
    ang = np.linspace(0, th, 30)
    ax.plot(o[0] + 0.2 * np.cos(ang), o[1] + 0.2 * np.sin(ang), color="0.4", lw=1.3)
    ax.text(o[0] + 0.29 * np.cos(th / 2), o[1] + 0.29 * np.sin(th / 2), r"$\theta_e$",
            fontsize=12, color="0.3")

fig.text(0.5, 0.965, "Cellular-sheaf spatial-domain pipeline — all steps",
         fontsize=18, fontweight="bold", ha="center")

# =================== BAND 1 : sheaf construction ============================
y1, h1, w1 = 0.75, 0.165, 0.15
xs1 = [0.035, 0.222, 0.409, 0.596, 0.783]
fig.text(0.035, 0.948, "1.  Sheaf construction", fontsize=13.5,
         fontweight="bold", color="0.2")
imgpanel([xs1[0], y1, w1, h1], "01_fibers", "Fibers", "a",
         "$F(v)$: kernel-smoothed\nexpression on a grid")
imgpanel([xs1[1], y1, w1, h1], "02_graph", "Spatial graph", "b",
         "$k$-NN graph  $G=(V,E)$")
axs = blankpanel([xs1[2], y1, w1, h1], "Local-PCA stalks", "c",
                 "$U_v$ = top-$d$ PCA of\n$\\{F(u):u\\in N(v)\\}$")
draw_stalk(axs)
axr = blankpanel([xs1[3], y1, w1, h1], "Restriction maps", "d",
                 "align via SVD$(U_u^{\\!\\top}U_v)$;\nprincipal angle $\\theta_e$")
draw_restriction(axr)
imgpanel([xs1[4], y1, w1, h1], "03_disagreement", "Sheaf disagreement", "e",
         "$D(e)=\\|F_{u\\triangleleft e}\\xi_u-F_{v\\triangleleft e}\\xi_v\\|^2$")
ym1 = y1 + h1 / 2
for i in range(4):
    arrow(xs1[i] + w1 + 0.002, xs1[i + 1] - 0.002, ym1)

# =================== BAND 2 : boundary -> domains ===========================
y2, h2, w2 = 0.42, 0.165, 0.15
xs2 = [0.13, 0.33, 0.53, 0.73]
fig.text(0.035, 0.67, "2.  Boundary field  $\\rightarrow$  domains",
         fontsize=13.5, fontweight="bold", color="0.2")
imgpanel([xs2[0], y2, w2, h2], "04_boundary", "Boundary score", "f",
         "$b(v)=$ smoothed $\\sum_{e\\ni v} D(e)$")
imgpanel([xs2[1], y2, w2, h2], "04b_attenuation", "Exponential attenuation", "g",
         "$w(v)=\\exp(-\\alpha\\,b(v)/\\mathrm{med}\\,b)$")
imgpanel([xs2[2], y2, w2, h2], "05_kmeans_raw", "Boundary-aware $k$-means", "h",
         "cluster $[\\,w\\!\\cdot\\!\\mathrm{PCA}(F)\\ \\|\\ \\lambda\\,xy\\,]$")
imgpanel([xs2[3], y2, w2, h2], "05_domains", "Connectivity cleanup", "i",
         "reassign fragments\n$\\rightarrow$ domains (ARI 0.89)")
ym2 = y2 + h2 / 2
for i in range(3):
    arrow(xs2[i] + w2 + 0.002, xs2[i + 1] - 0.002, ym2)

# return connector e -> f, routed through the clear whitespace band (y=0.645)
gy = 0.645
line(xs1[4] + w1, y1 - 0.002, xs1[4] + w1, gy)          # down the right margin
line(xs1[4] + w1, gy, xs2[0] + w2 / 2, gy)              # left across the gap
fig.add_artist(FancyArrowPatch((xs2[0] + w2 / 2, gy), (xs2[0] + w2 / 2, y2 + h2 + 0.004),
               transform=fig.transFigure, arrowstyle="-|>", mutation_scale=20,
               lw=1.8, color="0.35"))                    # down into f

# =================== BAND 3 : why the sheaf, not the kernel =================
y3, h3 = 0.075, 0.15
fig.text(0.035, 0.315, "3.  Why the sheaf, not the kernel",
         fontsize=13.5, fontweight="bold", color="0.2")
fig.text(0.5, 0.285,
         "the true domains are invisible to kernel/intensity views (identical mean "
         "expression and neighbour distances) — only the sheaf subspace comparison recovers them",
         fontsize=11, ha="center", color="0.3")
imgpanel([0.06, y3, 0.13, h3], "00_ground_truth", "True domains", "j")
imgpanel([0.24, y3, 0.13, h3], "kernel_mean_expression", "Mean expression", "k")
imgpanel([0.42, y3, 0.13, h3], "kernel_neighbour_distance", "Neighbour distance", "l")
axh = fig.add_axes([0.62, y3 + 0.012, 0.30, h3 - 0.024])
axh.imshow(img("sheaf_principal_angle_hist")); axh.axis("off")
axh.set_title("Stalk principal angles", fontsize=11.5, pad=3)
axh.text(-0.03, 1.06, "m", transform=axh.transAxes, fontsize=15,
         fontweight="bold", va="bottom", ha="right")

out = os.path.join(HERE, "fig1_blobs_full.png")
fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print("->", out)

# ---- also export every panel of the composition as its own file ------------
import shutil
PANELS = os.path.join(HERE, "fig1_panels")
os.makedirs(PANELS, exist_ok=True)
copy_map = {
    "a_fibers": "01_fibers", "b_spatial_graph": "02_graph",
    "e_sheaf_disagreement": "03_disagreement", "f_boundary_score": "04_boundary",
    "g_exponential_attenuation": "04b_attenuation", "h_boundary_aware_kmeans": "05_kmeans_raw",
    "i_connectivity_cleanup_domains": "05_domains", "j_true_domains": "00_ground_truth",
    "k_mean_expression": "kernel_mean_expression",
    "l_neighbour_distance": "kernel_neighbour_distance",
    "m_stalk_principal_angles": "sheaf_principal_angle_hist",
}
for dst, src in copy_map.items():
    shutil.copy(os.path.join(P, src + ".png"), os.path.join(PANELS, dst + ".png"))
# the two schematic panels are drawn, not data -> render them standalone
for name, drawer in [("c_local_pca_stalks", draw_stalk), ("d_restriction_maps", draw_restriction)]:
    f, a = plt.subplots(figsize=(3.3, 3.3))
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal"); a.axis("off")
    drawer(a)
    f.savefig(os.path.join(PANELS, name + ".png"), dpi=220, bbox_inches="tight", transparent=True)
    plt.close(f)
print("-> panels in", PANELS)
