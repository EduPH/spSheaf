"""Figure 1 — spSheaf, as a diagram.

One scene throughout: the tissue plane at the bottom of every panel, with the
construction stacking above it in z.

  a  molecules on the plane; one Gaussian-smoothed field per gene floating
     above. Reading up the vertical at a grid point gives that vertex's fiber.
  b  the spatial graph floating over the plane, with a stalk -- the local
     expression subspace, drawn as a tilted plane -- at each vertex.
  c  the same graph with each edge scored by its disagreement, and the
     smoothed boundary field above it.
  d  the partition on the plane, with the boundary field above it: the ridges
     sit over the domain borders.

Between a-b and c-d, a strip carries the algebra: the overlap of two stalks,
its singular value decomposition, and the residual that remains between the
transported coordinates once both stalks are in one frame.

Illustrative throughout; nothing plots measured data. Notation follows the
Methods -- M_uv = U_u^T U_v = L S R^T, F_{u<e} = L^T, F_{v<e} = R^T,
D(e) = ||F_{u<e} xi_u - F_{v<e} xi_v||^2.

    python fig1_pipeline.py
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import sheafstyle as ss

HERE = Path(__file__).parent
OUT = HERE / "out" / "fig1.pdf"

W_MM, H_MM = 180.0, 96.0

#: Margin left around a panel when it is exported on its own, so that text
#: reaching past the panel's box is not clipped at the file's edge.
PAD_MM = 1.5

#: The canvas being drawn into, and the shift applied to every coordinate.
#: Composing the whole figure leaves these at the defaults. Exporting one
#: panel sets the canvas to that panel's own box and shifts the coordinates
#: so the panel lands at the origin -- every draw function keeps writing the
#: composed-figure coordinates either way, so the two paths cannot drift.
_CW, _CH = W_MM, H_MM
_OX, _OY = 0.0, 0.0


@contextmanager
def canvas(w: float, h: float, ox: float = 0.0, oy: float = 0.0):
    global _CW, _CH, _OX, _OY
    prev = (_CW, _CH, _OX, _OY)
    _CW, _CH, _OX, _OY = w, h, ox, oy
    try:
        yield
    finally:
        _CW, _CH, _OX, _OY = prev


HEAD = ss.TYPE_PT["normal"]["text"]     # 8 pt
BODY = ss.TYPE_PT["normal"]["tick"]     # 6 pt

INK = "#242B33"
GREY = "#78848F"
RULE = "#AEB8C2"
FAINT = "#DCE2E8"
HOT = "#B03A18"

DOMAIN = ["#7A5CA8", "#0072B2", "#009E73"]
GENES = ["#7A5CA8", "#0072B2", "#009E73"]
GENE_CMAP = ["Purples", "Blues", "Greens"]

ELEV, AZIM = 22, -62

#: Edge scores run from "these two stalks agree" to "they do not". Grey to red
#: says that; a sequential yellow-to-red map paints agreement yellow as well.
def _edge_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    # Grey at the low end, not tan: most edges are interior, and the panel-b
    # caption calls them grey edges, so the ramp has to actually say grey.
    # The interior scores cluster below 0.3, so the ramp holds grey across
    # that range and only warms once edges genuinely disagree.
    return LinearSegmentedColormap.from_list(
        "agree_disagree", [(0.00, "#CDD5DC"), (0.38, "#C6C8CA"),
                           (0.70, "#D0793F"), (1.00, "#B03A18")])

rng = np.random.default_rng(11)


# ------------------------------------------------------------------ geometry

def outline_xy(n=260):
    t = np.linspace(0, 2 * np.pi, n)
    r = 1 + 0.16 * np.sin(3 * t + 0.7) + 0.06 * np.sin(5 * t - 0.4)
    return np.c_[r * np.cos(t), r * np.sin(t)]


OUTLINE = outline_xy()


def inside(p):
    t = np.arctan2(p[..., 1], p[..., 0])
    lim = 1 + 0.16 * np.sin(3 * t + 0.7) + 0.06 * np.sin(5 * t - 0.4)
    return np.hypot(p[..., 0], p[..., 1]) < lim * 0.96


def scatter_points(n, seed=0):
    r = np.random.default_rng(seed)
    p = r.uniform(-1.25, 1.25, (int(n * 2.8), 2))
    return p[inside(p)][:n]


def grid_points(step=0.215):
    g = np.arange(-1.08, 1.09, step)
    gx, gy = np.meshgrid(g, g)
    p = np.c_[gx.ravel(), gy.ravel()]
    return p[inside(p)]


def domain_of(p):
    x, y = p[..., 0], p[..., 1]
    return np.where(y > 0.28 + 0.26 * x, 0, np.where(x < -0.08, 1, 2))


CENTRES = [(-0.30, 0.62), (-0.62, -0.42), (0.55, -0.28)]


def gene_field(X, Y, g):
    cx, cy = CENTRES[g]
    f = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * 0.42 ** 2))
    f += 0.35 * np.exp(-((X - cx * 0.2) ** 2 + (Y - cy * 0.3) ** 2) / (2 * 0.55 ** 2))
    return f / f.max()


def knn_edges(pts, k=4):
    e = set()
    for i, p in enumerate(pts):
        for j in np.argsort(np.hypot(*(pts - p).T))[1:k + 1]:
            e.add((min(i, j), max(i, j)))
    return np.array(sorted(e))


def edge_scores(gp, ed):
    """Disagreement per edge, as a field rather than a test.

    Labelling an edge by whether its two endpoints fall in different domains
    makes the score binary, and leaves gaps wherever a border passes between
    two grid rows without separating them: 21 of 146 edges came out red and
    the boundary read as broken. D(e) is not a test, it rises as the two
    stalks stop agreeing, so it is drawn from the same smooth boundary field
    the rest of the figure uses, sampled at each edge's midpoint.
    """
    mid = 0.5 * (gp[ed[:, 0]] + gp[ed[:, 1]])
    b = boundary_field(mid[:, 0], mid[:, 1])
    r = np.random.default_rng(21)  # own seed: a panel drawn alone must match
    return np.clip(0.08 + 0.88 * b + 0.04 * r.standard_normal(len(ed)), 0, 1)


def boundary_field(X, Y):
    """A smooth ridge along the domain borders, for the surfaces in c and d."""
    d1 = np.abs(Y - (0.28 + 0.26 * X)) / np.sqrt(1 + 0.26 ** 2)
    d2 = np.abs(X + 0.08)
    below = Y <= 0.28 + 0.26 * X
    d = np.where(below, np.minimum(d1, d2), d1)
    return np.exp(-(d ** 2) / (2 * 0.13 ** 2))


# ------------------------------------------------------------------- helpers

def mm_axes(fig, x, y, w, h, **kw):
    x, y = x + _OX, y + _OY
    return fig.add_axes([x / _CW, 1 - (y + h) / _CH, w / _CW, h / _CH], **kw)


def ftext(fig, x_mm, y_mm, text, **kw):
    """Figure-level text placed in millimetres from the top-left corner."""
    return fig.text((x_mm + _OX) / _CW, 1 - (y_mm + _OY) / _CH, text, **kw)


def flat(ax):
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return ax


def scene(fig, x, y, w, h, zmax=4.1):
    ax = mm_axes(fig, x, y, w, h, projection="3d")
    ax.set_axis_off()
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((1.20, 1.00, 1.55))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_zlim(0, zmax)
    return ax


def base_plane(ax, colour_by=None, s=1.3, z=0.0, faint=False):
    """The tissue outline, with molecules on it."""
    ax.plot(OUTLINE[:, 0], OUTLINE[:, 1], z, color=RULE, lw=0.6, zorder=1)
    pts = scatter_points(360, seed=4)
    if faint:
        c, alpha = FAINT, 1.0
    elif colour_by == "gene":
        lab = domain_of(pts)
        r = np.random.default_rng(31)  # own seed: a panel drawn alone must match
        noisy = r.random(len(pts)) < 0.14
        gi = np.where(noisy, r.integers(0, 3, len(pts)), lab)
        c, alpha = [GENES[i] for i in gi], 1.0
    else:
        c, alpha = [DOMAIN[i] for i in domain_of(pts)], 1.0
    ax.scatter(pts[:, 0], pts[:, 1], np.full(len(pts), z), c=c, s=s,
               alpha=alpha, depthshade=False, linewidths=0, zorder=2)


def graph_layer(ax, gp, ed, z, colours=None, lw=0.55, node_s=2.4):
    segs = [[(gp[i, 0], gp[i, 1], z), (gp[j, 0], gp[j, 1], z)] for i, j in ed]
    lc = Line3DCollection(segs, linewidths=lw, zorder=3)
    if colours is None:
        lc.set_color(RULE)
    else:
        cmap = _edge_cmap()
        lc.set_color([cmap(float(c)) for c in colours])
        # the disagreeing edges are the point of the panel, so they are drawn
        # heavier as well as redder
        lc.set_linewidths([lw + (1.1 if c > 0.5 else 0.25) for c in colours])
    ax.add_collection3d(lc)
    ax.scatter(gp[:, 0], gp[:, 1], np.full(len(gp), z), color="white",
               edgecolor=GREY, linewidths=0.3, s=node_s, depthshade=False,
               zorder=4)


def stalk_fibers(ax, gp, z0, height=0.85, every=2, lw=0.7):
    """A stalk over every vertex, drawn the way sheaves usually are: a fiber
    standing on the base space. Height is not meaningful; the fiber is."""
    sel = gp[::every]
    lab = domain_of(sel)
    segs = [[(p[0], p[1], z0), (p[0], p[1], z0 + height)] for p in sel]
    lc = Line3DCollection(segs, linewidths=lw,
                          colors=[DOMAIN[l] for l in lab], zorder=5)
    lc.set_alpha(0.85)
    ax.add_collection3d(lc)
    ax.scatter(sel[:, 0], sel[:, 1], np.full(len(sel), z0 + height),
               c=[DOMAIN[l] for l in lab], s=3.0, depthshade=False,
               linewidths=0, zorder=6)


def restriction_callout(ax, u, v, z0, height=0.85, rise=0.95, stalk=0.90,
                        labels=True):
    """One edge in full: F(u) -> F(e) <- F(v), the maps pointing up into the
    edge stalk, with what is left between the two images inside it.

    ``rise`` and ``stalk`` shrink the construction to fit a small vignette;
    ``labels`` drops the names where the panel is too narrow to hold them.
    """
    mid = (u + v) / 2
    ze = z0 + height + rise
    for p, col in ((u, DOMAIN[0]), (v, DOMAIN[1])):
        ax.plot([p[0], p[0]], [p[1], p[1]], [z0, z0 + height], color=col,
                lw=2.0, zorder=7)
        ax.scatter([p[0]], [p[1]], [z0 + height], c=col, s=6,
                   depthshade=False, zorder=8)
        ax.plot([p[0], mid[0]], [p[1], mid[1]], [z0 + height, ze],
                color=col, lw=1.0, ls=(0, (2.2, 1.4)), zorder=7)
    # the edge stalk, and the two images inside it
    ax.plot([u[0], v[0]], [u[1], v[1]], [z0, z0], color=INK, lw=1.1, zorder=6)
    ax.plot([mid[0], mid[0]], [mid[1], mid[1]], [ze, ze + stalk],
            color=INK, lw=2.0, zorder=8)
    lo, hi = ze + 0.24 * stalk, ze + 0.78 * stalk
    ax.scatter([mid[0]], [mid[1]], [lo], c=DOMAIN[0], s=8, depthshade=False,
               zorder=9)
    ax.scatter([mid[0]], [mid[1]], [hi], c=DOMAIN[1], s=8, depthshade=False,
               zorder=9)
    ax.plot([mid[0], mid[0]], [mid[1], mid[1]], [lo, hi], color=HOT, lw=2.6,
            zorder=10)
    ax.text(mid[0] + 0.20, mid[1], (lo + hi) / 2, r"$D(e)$", fontsize=BODY,
            color=HOT, zorder=11)
    if labels:
        ax.text(mid[0] - 0.05, mid[1], ze + stalk + 0.15,
                r"$\mathcal{F}(e)$", fontsize=BODY, color=INK, zorder=11)
        ax.text(u[0] - 0.46, u[1] - 0.10, z0 + height + 0.02,
                r"$\mathcal{F}(u)$", fontsize=BODY, color=DOMAIN[0], zorder=11)
        ax.text(v[0] + 0.14, v[1] + 0.02, z0 + height + 0.02,
                r"$\mathcal{F}(v)$", fontsize=BODY, color=DOMAIN[1], zorder=11)


def domain_surface(ax, z, n=150, alpha=1.0):
    """The partition as filled regions rather than a scatter of labels."""
    from matplotlib.colors import to_rgba
    g = np.linspace(-1.25, 1.25, n)
    X, Y = np.meshgrid(g, g)
    mask = inside(np.dstack([X, Y]))
    lab = domain_of(np.dstack([X, Y]))
    cols = np.zeros(X.shape + (4,))
    for k, c in enumerate(DOMAIN):
        r, gg, b, _ = to_rgba(c)
        sel = (lab == k) & mask
        cols[sel] = (r, gg, b, alpha)
    s = ax.plot_surface(X, Y, np.where(mask, z, np.nan), facecolors=cols,
                        linewidth=0, antialiased=False, shade=False,
                        rstride=1, cstride=1, zorder=3)
    s.set_rasterized(True)


def field_surface(ax, z0, amp, cmap, alpha=0.92, n=44):
    g = np.linspace(-1.25, 1.25, n)
    X, Y = np.meshgrid(g, g)
    mask = inside(np.dstack([X, Y]))
    F = boundary_field(X, Y)
    Z = np.where(mask, z0 + amp * F, np.nan)
    s = ax.plot_surface(X, Y, Z, cmap=cmap, vmin=z0 - 0.55 * amp,
                        vmax=z0 + 1.05 * amp, linewidth=0, antialiased=True,
                        alpha=alpha, rstride=1, cstride=1, shade=True, zorder=6)
    s.set_rasterized(True)


def label(fig, x_mm, y_mm, letter, title, sub=None):
    ftext(fig, x_mm, y_mm, letter, fontsize=ss.TYPE_PT["normal"]["label"],
          color=INK, ha="left", va="top", fontweight="bold")
    ftext(fig, x_mm + 5.2, y_mm, title, fontsize=HEAD, color=INK,
          ha="left", va="top", fontweight="semibold")
    if sub:
        ftext(fig, x_mm + 5.2, y_mm + 4.8, sub, fontsize=BODY, color=GREY,
              ha="left", va="top")


def layer_label(fig, x_mm, y_mm, text, color=GREY):
    """Name a layer of the stack, at the right edge of its scene."""
    ftext(fig, x_mm, y_mm, text, fontsize=BODY, color=color, ha="left",
          va="center", linespacing=1.45)


def caption3d(fig, x_mm, y_mm, text, color=GREY):
    ftext(fig, x_mm, y_mm, text, fontsize=BODY, color=color, ha="center",
          va="top", linespacing=1.45)


def block(ax, x, y, w, h, n, colour, label_txt=None):
    for i in range(n):
        ax.add_patch(Polygon(
            [[x + i * w / n, y], [x + (i + 1) * w / n, y],
             [x + (i + 1) * w / n, y + h], [x + i * w / n, y + h]],
            closed=True, facecolor=colour, alpha=0.55, edgecolor="white",
            lw=0.5, zorder=3))
    ax.add_patch(Polygon([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                         closed=True, facecolor="none", edgecolor=INK, lw=0.6,
                         zorder=4))
    if label_txt:
        ax.text(x + w / 2, y - 0.10, label_txt, fontsize=BODY, color=INK,
                ha="center", va="top", zorder=5)


# ------------------------------------------------------------------ storyline
# Figure 1 is one story read left to right, not four boxes. The five stages sit
# on a single band; the stage where the sheaf actually does its work is opened
# out underneath, connected by a wedge, so the glance and the mechanism are the
# same picture.

STAGE_W, STAGE_H = 31.3, 52.0
STAGE_Y = 11.0

#: The title row is placed absolutely rather than measured off the scene box.
#: A 3-D box is drawn as a parallelogram, so its top corner is empty whatever
#: the content reaches; letting the headings sit against that empty corner
#: costs nothing and saves six millimetres.
TITLE_Y, SUB_Y = 4.6, 9.6
#: The row starts clear of the left margin so the panel letter has its own
#: column and does not sit under stage 1's badge.
STAGE_XS = [6.0 + i * (STAGE_W + 4.0) for i in range(5)]
BAND_Y = STAGE_Y + 0.56 * STAGE_H      # the height the chevrons ride at

#: The detail is a zoom-up of stage 3, so it is drawn smaller than the story
#: it explains and inset from the full width. Anything full-bleed down here
#: competes with the sequence above instead of serving it.
MECH_X, MECH_W = 26.0, 128.0
MECH_Y, MECH_H = 62.0, 32.0
#: The magnification is not of a stage but of the step between two of them:
#: stage 3 has the stalks, stage 4 has a score on every edge, and the block
#: below shows how the first becomes the second. So the ring sits on the arrow
#: joining them, which is the thing being explained.
LENS = (STAGE_XS[2] + STAGE_W + 2.25, BAND_Y, 4.4)

#: mpl reserves a wide margin inside a 3-D axes, so a vignette drawn at its
#: natural scale fills about half the box it was given. `zoom` on the box
#: aspect reclaims it; this is what makes the stages readable at 32 mm.
SCENE_ZOOM = 1.5

#: Where the tissue plane sits in each vignette, as a fraction of the box
#: height from the bottom. Each stage needs its own zmax -- the gene sheets
#: stack far higher than the graph does -- and with a shared floor that put
#: the plane at a different height in every stage, so the sequence had no
#: baseline. Fixing the fraction instead aligns the planes and still lets each
#: stage fill its own box.
PLANE_FRAC = 0.15

#: One vertical range for all five vignettes. Tuning zmax per stage would fill
#: each box, but it also rescales z, so the same structure -- the graph, say --
#: would sit at a different height in stage 3 than in stage 4. Instead the
#: range is shared and each stage's content is placed to use it.
Z_TOP = 3.8

ACCENT = "#5B7DB1"


def overlay(fig, zorder):
    """A transparent axes covering the whole canvas, in millimetres.

    Used for everything that belongs to the figure rather than to a scene --
    the band, the chevrons, the stage badges, the wedge down to the mechanism.
    Drawn before the scenes for the background, after them for the marks.
    """
    ax = fig.add_axes([0, 0, 1, 1], zorder=zorder)
    ax.set_axis_off()
    ax.set_xlim(0, _CW); ax.set_ylim(_CH, 0)   # y downward, like the layout
    ax.patch.set_alpha(0.0)
    return ax


def _mm(x, y):
    return x + _OX, y + _OY


def chevron(ax, x, y, size=2.6, color=ACCENT):
    x, y = _mm(x, y)
    ax.add_patch(Polygon([[x - size * 0.5, y - size],
                          [x + size * 0.75, y],
                          [x - size * 0.5, y + size]],
                         closed=True, facecolor=color, edgecolor="none",
                         zorder=6))


def badge(ax, x, y, n, r=2.5):
    """A numbered stage marker: what the eye counts before it reads."""
    x, y = _mm(x, y)
    ax.add_patch(Circle((x, y), r, facecolor=ACCENT, edgecolor="none",
                        zorder=7))
    ax.text(x, y, str(n), fontsize=BODY, color="white", ha="center",
            va="center", zorder=8, fontweight="bold")


def lens(ax):
    """A ring on stage 3, opening down into the detail: this, magnified."""
    cx, cy, r = LENS
    cx, cy = _mm(cx, cy)
    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=ACCENT,
                        lw=0.7, zorder=9))
    xa, yt = _mm(MECH_X, MECH_Y)
    xb, _ = _mm(MECH_X + MECH_W, 0)
    # tangents from the ring to the top corners of the detail block
    for x_end, sign in ((xa, -1), (xb, +1)):
        ax.plot([cx + sign * r * 0.72, x_end], [cy + r * 0.69, yt],
                color=ACCENT, lw=0.5, ls=(0, (2.5, 1.8)), zorder=2)


def stage_title(fig, i, title, sub):
    """Number, name and one line, above each vignette."""
    x = STAGE_XS[i]
    ftext(fig, x + 11.0, TITLE_Y, title, fontsize=HEAD, color=INK,
          ha="left", va="center", fontweight="semibold")
    ftext(fig, x + STAGE_W / 2, SUB_Y, sub, fontsize=BODY, color=GREY,
          ha="center", va="center", linespacing=1.35)


def vignette(fig, i, zmax=Z_TOP):
    ax = scene(fig, STAGE_XS[i], STAGE_Y, STAGE_W, STAGE_H, zmax=zmax)
    ax.set_box_aspect((1.20, 1.00, 1.35), zoom=SCENE_ZOOM)
    ax.set_xlim(-1.22, 1.22); ax.set_ylim(-1.22, 1.22)
    # The tissue plane sits at z = 0, and zooming enlarges the box about its
    # centre, so with 0 as the floor the plane would be the first thing
    # cropped. Solving for the floor that puts z = 0 at PLANE_FRAC gives both
    # the headroom and one baseline across the five stages.
    ax.set_zlim(-PLANE_FRAC * zmax / (1.0 - PLANE_FRAC), zmax)
    return ax


# --- the five stages -------------------------------------------------------

def stage1(fig):
    """Molecules on the plane, with the grid laid over them."""
    ax = vignette(fig, 0)
    base_plane(ax, colour_by="gene", s=1.5)
    z = 2.60
    ax.plot(OUTLINE[:, 0], OUTLINE[:, 1], z, color=RULE, lw=0.5, zorder=5)
    ax.scatter(GP[:, 0], GP[:, 1], np.full(len(GP), z), color="white",
               edgecolor=GREY, linewidths=0.3, s=1.8, depthshade=False,
               zorder=6)
    for q in GP[::11]:
        ax.plot([q[0], q[0]], [q[1], q[1]], [0.04, z - 0.04], color=RULE,
                lw=0.4, ls=(0, (1.6, 1.4)), zorder=4)
    ax.text(-1.15, 1.15, z + 0.60, "grid", fontsize=BODY, color=GREY,
            zorder=9)


def stage2(fig):
    """One smoothed field per gene; reading up the vertical gives the fiber."""
    ax = vignette(fig, 1)
    base_plane(ax, faint=True, s=1.0)
    x0, y0 = 0.10, 0.10
    g = np.linspace(-1.25, 1.25, 40)
    X, Y = np.meshgrid(g, g)
    mask = inside(np.dstack([X, Y]))
    for k, (z0, cmap) in enumerate(zip([1.00, 1.95, 2.90], GENE_CMAP)):
        F = gene_field(X, Y, k)
        Z = np.where(mask, z0 + 0.48 * F, np.nan)
        s = ax.plot_surface(X, Y, Z, cmap=cmap, vmin=z0 - 0.30, vmax=z0 + 0.60,
                            linewidth=0, antialiased=True, alpha=0.93,
                            rstride=1, cstride=1, shade=True, zorder=4)
        s.set_rasterized(True)
        fk = gene_field(np.array([[x0]]), np.array([[y0]]), k)[0, 0]
        ax.scatter([x0], [y0], [z0 + 0.48 * fk], color=GENES[k], s=5,
                   depthshade=False, zorder=7)
    ax.plot([x0, x0], [y0, y0], [0, 3.55], color=INK, lw=0.6, ls=(0, (2, 1.6)),
            zorder=6)
    ax.text(x0 + 0.30, y0, 3.70, r"$\tilde f_v\in\mathbb{R}^{G}$",
            fontsize=BODY, color=INK, zorder=9)


def stage3(fig):
    """The graph, a stalk over every vertex, and one edge worked in full."""
    ax = vignette(fig, 2)
    base_plane(ax, faint=True, s=0.9)
    graph_layer(ax, GP, ED, z=1.05, lw=0.5, node_s=1.8)
    stalk_fibers(ax, GP, z0=1.05, height=0.80, every=3, lw=0.6)
    # One edge in full: both restriction maps carrying their stalk up into the
    # edge stalk, and D(e) as what is left between the two images inside it.
    # Its vertices are placed by hand at the near edge of the view so the two
    # maps read as distinct instead of projecting onto each other.
    u, v = np.array([0.20, -0.66]), np.array([0.78, -0.24])
    restriction_callout(ax, u, v, z0=1.05, height=0.80, rise=0.62, stalk=0.62,
                        labels=False)
    mid = (u + v) / 2
    # F(e) names the edge stalk, so it sits on top of it; D(e) names the red
    # span inside it and stays alongside
    ax.text(mid[0], mid[1], 1.05 + 0.80 + 0.62 + 0.62 + 0.22,
            r"$\mathcal{F}(e)$", fontsize=BODY, color=INK, zorder=11,
            ha="center", va="bottom")
    # name a stalk where one actually stands, rather than in the corner
    sel = GP[::3]
    q = sel[np.argmin(sel[:, 0] + 0.4 * sel[:, 1])]
    ax.text(q[0] - 0.42, q[1], 1.05 + 0.80 + 0.10, r"$\mathcal{F}(v)$",
            fontsize=BODY, color=DOMAIN[int(domain_of(q))], zorder=11)


def stage4(fig):
    """Every edge scored, then smoothed into a field."""
    ax = vignette(fig, 3)
    base_plane(ax, faint=True, s=0.9)
    graph_layer(ax, GP, ED, z=1.05, colours=edge_scores(GP, ED), lw=0.55,
                node_s=0.0)
    field_surface(ax, 2.25, 0.80, "YlOrRd", n=40)
    ax.text(-0.72, 1.02, 3.02, r"$\tilde B(v)$", fontsize=BODY, color=HOT,
            zorder=9)


def stage5(fig):
    """The partition, with the field's ridges over its borders."""
    ax = vignette(fig, 4)
    domain_surface(ax, 0.0)
    ax.plot(OUTLINE[:, 0], OUTLINE[:, 1], 0.01, color="white", lw=0.6,
            zorder=4)
    field_surface(ax, 2.20, 0.80, "YlOrRd", alpha=0.60, n=40)


STAGES = [
    (stage1, "Molecules",  "raw detections, no\nsegmentation; a grid  (1)"),
    (stage2, "Fibers",     "one smoothed field\nper gene  (1)"),
    (stage3, "Stalks",     "a vector space per vertex,\nmaps along edges  (2-4)"),
    (stage4, "Disagreement", "per edge, then\nsmoothed  (5, 6)"),
    (stage5, "Domains",    "boundary-attenuated\nclustering  (7, 8)"),
]


# --- the mechanism, opened out ---------------------------------------------

def subspace(ax, o, ang, color, w=0.052, h=0.30, alpha=0.30, lw=0.7):
    """A plane through the origin, seen edge-on: a stalk in gene space."""
    a = np.deg2rad(ang)
    e1 = np.array([np.cos(a) * w, np.sin(a) * h])
    e2 = np.array([-np.sin(a) * w * 0.55, np.cos(a) * h * 0.30])
    quad = np.array([o + e1 + e2, o + e1 - e2, o - e1 - e2, o - e1 + e2])
    ax.add_patch(Polygon(quad, closed=True, facecolor=color, alpha=alpha,
                         edgecolor=color, lw=lw, zorder=3))
    return o + e1


def mechanism(fig):
    """The zoom-up of stage 3: what happens on one edge.

    Two cells, not three. This block explains the sequence above; every extra
    element in it takes attention the sequence should be getting.
    """
    from matplotlib.patches import Arc

    ax = flat(mm_axes(fig, MECH_X, MECH_Y, MECH_W, MECH_H))
    ax.add_patch(FancyBboxPatch((0.002, 0.02), 0.996, 0.96,
                                boxstyle="round,pad=0.003,rounding_size=0.02",
                                transform=ax.transAxes, facecolor="none",
                                edgecolor=RULE, lw=0.5, zorder=0))
    ax.plot([0.50, 0.50], [0.10, 0.86], color=RULE, lw=0.5)
    ax.text(0.022, 0.90, r"comparing the stalks across one edge $e=(u,v)$",
            fontsize=BODY,
            color=ACCENT, ha="left", va="center")

    # --- left cell: the stalks are aligned before they are compared
    o = np.array([0.105, 0.50])
    t1 = subspace(ax, o, 86, DOMAIN[0], w=0.055, h=0.22)
    t2 = subspace(ax, o, 32, DOMAIN[1], w=0.055, h=0.22)
    for tip, col in ((t1, DOMAIN[0]), (t2, DOMAIN[1])):
        ax.add_patch(FancyArrowPatch(o, tip, arrowstyle="-|>", mutation_scale=4,
                                     lw=1.0, color=col, shrinkA=0, shrinkB=0,
                                     zorder=5))
    ax.add_patch(Arc(o, 0.085, 0.30, theta1=32, theta2=86, lw=0.7, color=INK,
                     zorder=7))
    ax.text(o[0] + 0.052, o[1] + 0.135, r"$\theta_1$", fontsize=BODY,
            color=INK, ha="center", va="center", zorder=8)

    ax.annotate("", (0.268, 0.52), (0.198, 0.52),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK))
    ax.text(0.233, 0.63, "align", fontsize=BODY, color=INK, ha="center")

    o2 = np.array([0.360, 0.50])
    subspace(ax, o2, 57, DOMAIN[0], alpha=0.30, w=0.060, h=0.24)
    tip = subspace(ax, o2, 57, DOMAIN[1], alpha=0.30, w=0.049, h=0.20)
    ax.add_patch(FancyArrowPatch(o2, tip, arrowstyle="-|>", mutation_scale=4,
                                 lw=1.2, color=INK, shrinkA=0, shrinkB=0,
                                 zorder=5))
    ax.text(0.245, 0.155,
            r"$M_{uv}=U_u^{\top}U_v=L\Sigma R^{\top}$,  "
            r"$\sigma_i=\cos\theta_i$",
            fontsize=BODY, color=INK, ha="center", va="center")

    # --- right cell: what is left once they are in one frame
    ax.text(0.755, 0.865, r"$D(e)=\|\mathcal{F}_{u\triangleleft e}\xi_u-"
                          r"\mathcal{F}_{v\triangleleft e}\xi_v\|^{2}$",
            fontsize=HEAD, color=HOT, ha="center", va="center")
    cmap = _edge_cmap()
    for cx, au, av, ttl, score, bar in (
            (0.635, 46, 68, "inside a domain", 0.12, 0.055),
            (0.875, 26, 96, "across a boundary", 0.90, 0.150)):
        oo = np.array([cx - 0.035, 0.42])
        for ang, col in ((au, DOMAIN[0]), (av, DOMAIN[1])):
            a = np.deg2rad(ang)
            tip = oo + np.array([0.065 * np.cos(a), 0.36 * np.sin(a)])
            ax.add_patch(FancyArrowPatch(oo, tip, arrowstyle="-|>",
                                         mutation_scale=6, lw=1.2, color=col,
                                         shrinkA=0, shrinkB=0, zorder=5))
        a1, a2 = np.deg2rad(au), np.deg2rad(av)
        p1 = oo + np.array([0.065 * np.cos(a1), 0.36 * np.sin(a1)])
        p2 = oo + np.array([0.065 * np.cos(a2), 0.36 * np.sin(a2)])
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="<|-|>",
                                     mutation_scale=5,
                                     lw=1.8 if score > 0.5 else 1.0,
                                     color=HOT, shrinkA=0, shrinkB=0, zorder=6))
        # The residual is what colours the edge in stage 4, so it is drawn on
        # that scale and in that colour rather than left as an abstraction.
        x0 = cx - 0.005 - bar / 2
        ax.add_patch(Polygon([[x0, 0.225], [x0 + bar, 0.225],
                              [x0 + bar, 0.268], [x0, 0.268]], closed=True,
                             facecolor=cmap(score), edgecolor=RULE, lw=0.3,
                             zorder=5))
        ax.text(cx - 0.005, 0.135, ttl, fontsize=BODY, color=GREY,
                ha="center", va="center")
        # Naming the consequence beside each case says what the bar is for,
        # without a floating caption crossing both of them.
        ax.text(cx - 0.005, 0.055,
                "small $D(e)$: grey edge" if score < 0.5
                else "large $D(e)$: red edge",
                fontsize=BODY, color=GREY if score < 0.5 else HOT,
                ha="center", va="center")


GP = grid_points()
ED = knn_edges(GP)


@dataclass
class Item:
    """One piece of the figure, and the box it occupies on the 180 mm canvas.

    ``draw`` writes in composed-figure coordinates whether it is being called
    to build the whole figure or to export the piece on its own; the shift is
    applied by ``canvas``. There is therefore one description of the layout,
    not two, and an exported piece is the same drawing as the composed one.
    """

    name: str
    x: float
    y: float
    w: float
    h: float
    draw: Callable[[object], None]


def _draw_storyline(fig):
    for i, (draw, title, sub) in enumerate(STAGES):
        draw(fig)
        stage_title(fig, i, title, sub)
    over = overlay(fig, zorder=20)
    for i in range(5):
        badge(over, STAGE_XS[i] + 3.5, TITLE_Y, i + 1)
        if i < 4:
            chevron(over, STAGE_XS[i] + STAGE_W + 2.25, BAND_Y)


def _draw_mechanism(fig):
    lens(overlay(fig, zorder=1))
    mechanism(fig)


ITEMS = [
    Item("fig1a_storyline", 0.0,   1.0, 180.0, 57.0, _draw_storyline),
    Item("fig1b_detail",    0.0,  58.0, 180.0, 38.0, _draw_mechanism),
]


# -------------------------------------------------------------------- build

def build():
    """The whole figure, one 180 x 142 mm canvas."""
    ss.apply()
    fig = ss.panel(W_MM, H_MM)[0]
    fig.clear()
    with canvas(W_MM, H_MM):
        for item in ITEMS:
            item.draw(fig)
        ftext(fig, 1.0, TITLE_Y, "a", fontsize=ss.TYPE_PT["normal"]["label"],
              color=INK, ha="left", va="center", fontweight="bold")
        ftext(fig, MECH_X - 4.5, MECH_Y - 1.0, "b",
              fontsize=ss.TYPE_PT["normal"]["label"], color=INK, ha="left",
              va="top", fontweight="bold")
    return fig


def build_item(item: Item, pad: float = PAD_MM):
    """One piece on its own canvas, drawn at exactly its composed size."""
    ss.apply()
    w, h = item.w + 2 * pad, item.h + 2 * pad
    fig = ss.panel(w, h)[0]
    fig.clear()
    with canvas(w, h, pad - item.x, pad - item.y):
        item.draw(fig)
    return fig


def export_panels(outdir: Path, pad: float = PAD_MM, png_dpi: int = 600):
    """Write every piece separately, plus a manifest of where each one goes.

    Each file is exactly the size it occupies in the composed figure, so
    placing it at 100% -- never scaled to fit -- puts the type back at 6, 8 and
    10 pt. The manifest gives the top-left corner of each file on a 180 mm
    canvas, so the composition can be rebuilt exactly.
    """
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in ITEMS:
        fig = build_item(item, pad)
        stem = outdir / item.name
        ss.save_panel(fig, stem.with_suffix(".pdf"), strict=False)
        ss.save_panel(fig, stem.with_suffix(".png"), strict=False, dpi=png_dpi)
        plt.close(fig)
        rows.append((item.name, item.x - pad, item.y - pad,
                     item.w + 2 * pad, item.h + 2 * pad))
        print(f"  {item.name:18s} {rows[-1][3]:6.1f} x {rows[-1][4]:5.1f} mm "
              f"at ({rows[-1][1]:6.1f}, {rows[-1][2]:5.1f})")

    manifest = outdir / "fig1_placement.csv"
    with manifest.open("w") as fh:
        fh.write("# place each file at 100%, top-left corner at (x_mm, y_mm)\n")
        fh.write(f"# canvas: {W_MM:g} x {H_MM:g} mm\n")
        fh.write("file,x_mm,y_mm,width_mm,height_mm\n")
        for name, x, y, w, h in rows:
            fh.write(f"{name}.pdf,{x:.1f},{y:.1f},{w:.1f},{h:.1f}\n")
    print("\nwrote", manifest)
    return rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--panels":
        outdir = Path(argv[1]) if len(argv) > 1 else HERE / "panels" / "fig1"
        print(f"fig1 pieces, each at its composed size, into {outdir}:")
        export_panels(outdir)
        return
    fig = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ss.save_panel(fig, OUT, strict=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
