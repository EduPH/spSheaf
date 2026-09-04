"""Extended Data — PECAb multiplexed protein imaging.

Drawn from the arrays ``pecab_signaling.py`` exports, not from its plots:

    data/results/pecab/output_pecab_signaling_Rep{1,2}/panels/pecab.npz

  a  five protein-state domains on the section
  b  the sheaf boundary field over the same grid
  c  the mesenchymal-minus-epithelial score, a continuous axis
  d  lineage-marker co-localization, replicate 1
  e  the same, replicate 2
  f  the 21 marker pairs, replicate 1 against replicate 2

The section is drawn with no frame and no tick labels: the coordinates are
stage micrometres and carry no meaning for the reader, so a scale bar says
what the axes would have said, with less ink.

    python ed_pecab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import sheafstyle as ss
from layout import Layout, Panel

HERE = Path(__file__).parent
REPO = HERE.parent
PANELS = HERE / "panels"
PANELS.mkdir(exist_ok=True)
RESULTS = REPO / "data" / "results" / "pecab"
OUT = HERE / "out" / "ed_pecab.pdf"

TICK = ss.TYPE_PT["normal"]["tick"]
TEXT = ss.TYPE_PT["normal"]["text"]

#: Five domains. Okabe-Ito, with the mesenchymal compartment carrying the
#: strong blue it has everywhere else in the manuscript.
DOMAIN_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#7A5CA8"]

#: Antibody names as the manuscript writes them.
PRETTY = {"aSMA": r"$\alpha$-SMA", "a-SMA": r"$\alpha$-SMA",
          "E-Cadherin": "E-cadherin", "N-Cadherin": "N-cadherin"}

SCALEBAR_UM = 2000.0


def load(rep: str):
    sys.path.insert(0, str(REPO))
    from experiments.common.panel_export import load_panels
    return load_panels(str(RESULTS / f"output_pecab_signaling_{rep}" / "panels" / "pecab.npz"))


def _to_image(grid, values, fill=np.nan):
    """Rasterize a regular grid onto a 2-D array, with ``extent``.

    The fibers sit on a lattice, so the section is an image, not a scatter.
    Drawing it as square markers leaves seams wherever the marker size and the
    lattice pitch disagree -- visible as stripes across the tissue -- and the
    seams change with the output size. An image has no such freedom.
    """
    x, y = grid[:, 0], grid[:, 1]
    ux, uy = np.unique(x), np.unique(y)
    dx = np.min(np.diff(ux)) if len(ux) > 1 else 1.0
    dy = np.min(np.diff(uy)) if len(uy) > 1 else 1.0
    ix = np.rint((x - x.min()) / dx).astype(int)
    iy = np.rint((y - y.min()) / dy).astype(int)

    values = np.asarray(values)
    shape = (iy.max() + 1, ix.max() + 1)
    if values.ndim == 2:                       # already RGBA
        img = np.zeros(shape + (4,), dtype=float)
        img[iy, ix] = values
    else:
        img = np.full(shape, fill, dtype=float)
        img[iy, ix] = values
    # imshow's extent is the outer edge of the border pixels, not their centres
    extent = (x.min() - dx / 2, x.max() + dx / 2,
              y.max() + dy / 2, y.min() - dy / 2)   # inverted: y grows downward
    return img, extent


def _section(ax):
    """Shared frame for the three maps: no frame, no ticks, image aspect."""
    ss.spatial(ax, equal=False)   # imshow sets equal aspect itself


def _scalebar(ax, grid, um=SCALEBAR_UM, label=None):
    """Bottom-left of the axes, in data units, clear of the artwork."""
    x0 = grid[:, 0].min() + 0.04 * np.ptp(grid[:, 0])
    y0 = grid[:, 1].max() + 0.10 * np.ptp(grid[:, 1])
    ax.plot([x0, x0 + um], [y0, y0], color="#242B33", lw=1.3,
            solid_capstyle="butt", clip_on=False)
    ax.text(x0 + um / 2, y0 + 0.045 * np.ptp(grid[:, 1]),
            label or f"{um/1000:g} mm", fontsize=TICK, color="#242B33",
            ha="center", va="top", clip_on=False)


def _cbar(fig, ax, mappable, label, ticks=None):
    """A colour bar pinned to the image, not to the figure edge."""
    cax = ax.inset_axes([1.03, 0.06, 0.035, 0.88])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=TICK, labelpad=2)
    cb.ax.tick_params(labelsize=TICK, length=1.6, width=0.4, pad=1.2)
    cb.outline.set_linewidth(0.4)
    if ticks is not None:
        cb.set_ticks(ticks)
    return cb


def panel_a(path, arr, meta):
    """The five protein-state domains, each named where it sits."""
    from matplotlib.colors import to_rgba

    grid, lab = arr["grid"], arr["labels"]
    mes = int(meta["mes_domain"])
    rgba = np.array([to_rgba(DOMAIN_COLORS[int(k) % len(DOMAIN_COLORS)])
                     for k in lab])
    img, extent = _to_image(grid, rgba)

    fig, ax = ss.panel(56, 48)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
    ax.imshow(img, extent=extent, interpolation="nearest", zorder=2)
    _section(ax)
    _scalebar(ax, grid)
    # Naming each domain on the tissue beats a legend: no key to cross-refer,
    # and no box competing with the section for space.
    for k in sorted({int(v) for v in lab}):
        m = lab == k
        pts = grid[m]
        # The centroid of a concave or split domain can fall outside it, which
        # puts the label on a neighbour. Snap to the domain's own point nearest
        # that centroid instead.
        c = pts.mean(0)
        cx, cy = pts[np.argmin(((pts - c) ** 2).sum(1))]
        txt = f"{k}\nmesenchymal" if k == mes else f"{k}"
        ax.text(cx, cy, txt, fontsize=TICK, color="white", ha="center",
                va="center", zorder=4, linespacing=1.2,
                path_effects=_halo())
    ss.save_panel(fig, path)


def _halo(lw=1.1):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw, foreground="#00000055")]


def _field(path, arr, key, cmap, label, diverging=False, pct=99.0):
    """One scalar field over the section.

    Colour limits are clipped to a percentile. The boundary field has a few
    very hot vertices; scaled to the maximum, everything else renders as one
    flat dark slab and the field the panel exists to show disappears.
    """
    grid, v = arr["grid"], np.asarray(arr[key], dtype=float)
    if diverging:
        lim = float(np.percentile(np.abs(v), pct))
        vmin, vmax, ticks = -lim, lim, [-lim, 0, lim]
    else:
        vmin, vmax = float(np.percentile(v, 100 - pct)), float(np.percentile(v, pct))
        ticks = None
    img, extent = _to_image(grid, v)

    fig, ax = ss.panel(56, 48)
    # A signed tick label ("+2.1") is wider than an unsigned one, and the bar's
    # own label sits outside it. Without the extra room the label is clipped by
    # the panel edge -- which save_panel cannot rescue, since the canvas is the
    # panel's final size.
    fig.subplots_adjust(left=0.02, right=0.74 if diverging else 0.82,
                        top=0.96, bottom=0.04)
    im = ax.imshow(img, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", zorder=2)
    _section(ax)
    _scalebar(ax, grid)
    cb = _cbar(fig, ax, im, label)
    if ticks is not None:
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:+.1f}".replace("+0.0", "0") if t else "0"
                           for t in ticks])
    ss.save_panel(fig, path)


def panel_b(path, arr, meta):
    _field(path, arr, "boundary_smooth", "inferno", "sheaf boundary score")


def panel_c(path, arr, meta):
    # "mesenchymal - epithelial" does not fit beside a 56 mm panel without
    # squeezing the section. The caption defines the score; the bar names it.
    _field(path, arr, "emt_score", "RdBu_r", "EMT score", diverging=True)


def _pretty(names):
    return [PRETTY.get(n, n) for n in names]


def _heatmap(path, C, names, title, cbar=False):
    fig, ax = ss.panel(50 if not cbar else 58, 46)
    fig.subplots_adjust(left=0.30, right=0.99 if not cbar else 0.82,
                        top=0.88, bottom=0.30)
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(_pretty(names), rotation=90)
    ax.set_yticklabels(_pretty(names))
    ax.tick_params(length=0, pad=1.5)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=TICK, pad=3)
    if cbar:
        _cbar(fig, ax, im, "spatial Pearson $r$", ticks=[-1, 0, 1])
    ss.save_panel(fig, path)


def panel_d(path, arr, meta):
    _heatmap(path, arr["marker_corr"], meta["markers"], "replicate 1")


def panel_e(path, arr, meta):
    _heatmap(path, arr["marker_corr"], meta["markers"], "replicate 2", cbar=True)


def panel_f(path, a1, m1, a2, m2):
    """The 21 off-diagonal marker pairs, one replicate against the other."""
    n1, n2 = list(m1["markers"]), list(m2["markers"])
    if n1 != n2:  # score only the markers both panels share, in one order
        common = [n for n in n1 if n in n2]
        i1 = [n1.index(n) for n in common]
        i2 = [n2.index(n) for n in common]
        C1 = arrsub(a1["marker_corr"], i1)
        C2 = arrsub(a2["marker_corr"], i2)
    else:
        C1, C2 = a1["marker_corr"], a2["marker_corr"]
    iu = np.triu_indices(C1.shape[0], k=1)
    x, y = C1[iu], C2[iu]
    r = float(np.corrcoef(x, y)[0, 1])

    fig, ax = ss.panel(56, 46)
    fig.subplots_adjust(left=0.20, right=0.97, top=0.90, bottom=0.18)
    ax.plot([-1, 1], [-1, 1], lw=0.6, ls=(0, (3, 2)), color="#9AA4AE", zorder=1)
    ax.scatter(x, y, s=6, color=ss.METHOD_COLORS["spSheaf"], linewidths=0,
               zorder=3)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_aspect("equal")
    ax.set_xlabel("replicate 1, marker-pair $r$")
    ax.set_ylabel("replicate 2, marker-pair $r$")
    ax.set_title(f"Pearson $r={r:.2f}$  ({len(x)} pairs)", fontsize=TICK, pad=3)
    ss.save_panel(fig, path)
    return r, len(x)


def arrsub(C, idx):
    return C[np.ix_(idx, idx)]


def main():
    ss.apply()
    a1, m1 = load("Rep1")
    a2, m2 = load("Rep2")

    panel_a(PANELS / "pec_a.pdf", a1, m1)
    panel_b(PANELS / "pec_b.pdf", a1, m1)
    panel_c(PANELS / "pec_c.pdf", a1, m1)
    panel_d(PANELS / "pec_d.pdf", a1, m1)
    panel_e(PANELS / "pec_e.pdf", a2, m2)
    r, npairs = panel_f(PANELS / "pec_f.pdf", a1, m1, a2, m2)

    lay = Layout(
        name="ed_pecab", width_mm=180.0, height_mm=110.0,
        panels=[
            Panel("a", PANELS / "pec_a.pdf", x=5,   y=6,  label_dx=-4, label_dy=-5),
            Panel("b", PANELS / "pec_b.pdf", x=63,  y=6,  label_dx=-4, label_dy=-5),
            Panel("c", PANELS / "pec_c.pdf", x=121, y=6,  label_dx=-4, label_dy=-5),
            Panel("d", PANELS / "pec_d.pdf", x=5,   y=60, label_dx=-4, label_dy=-4),
            Panel("e", PANELS / "pec_e.pdf", x=59,  y=60, label_dx=-4, label_dy=-4),
            Panel("f", PANELS / "pec_f.pdf", x=122, y=60, label_dx=-4, label_dy=-4),
        ],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(lay.summary())
    print(f"\ncross-replicate agreement: r = {r:.3f} over {npairs} marker pairs")
    print("wrote", lay.compose(OUT))


if __name__ == "__main__":
    main()
