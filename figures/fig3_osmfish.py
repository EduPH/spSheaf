"""Figure 3 — osmFISH mouse somatosensory cortex.

Drawn from the arrays ``run_osmFISH.py`` exports, not from its plots:

    data/results/osmfish/output_osmFISH_boundary/panels/osmfish.npz

  a  the eleven annotated cortical regions, per cell
  b  the predicted partition, Hungarian-matched to a
  c  where the two disagree
  d  the clusters on the grid the pipeline actually works on
  e  the raw sheaf boundary score
  f  the same after smoothing
  g  per-region recall, ordered as the cortex is layered

These are the quantities the legacy figure showed. The drawing is what
changed: one shared legend instead of an unreadable inset key, no axis frames
or pixel coordinates on the maps, a scale bar, and the boundary fields on
percentile-clipped colour limits so the field is visible rather than flattened
by its few hottest vertices.

    python fig3_osmfish.py
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
def _results(*parts):
    """Resolve a results path through ``spsheaf.paths``.

    Building it from the repository root instead ignores SPSHEAF_DATA and
    paths.toml, so a checkout whose data lives elsewhere -- which is the normal
    case, the tree is tens of gigabytes -- cannot find its own inputs.
    """
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from spsheaf import paths
    return paths.results().joinpath(*parts)


DUMP = _results("osmfish", "output_osmFISH_boundary", "panels", "osmfish.npz")
OUT = HERE / "out" / "fig3.pdf"

TICK = ss.TYPE_PT["normal"]["tick"]
TEXT = ss.TYPE_PT["normal"]["text"]

GREY = "#B6C0CA"
INK = "#242B33"
HOT = "#B03A18"

#: Cortex reads outside-in, so the palette runs light to dark through the
#: layers and gives the non-laminar structures their own hues.
REGION_COLORS = {
    "Pia Layer 1":                   "#F2E3C6",
    "Layer 2-3 lateral":             "#F5C77E",
    "Layer 2-3 medial":              "#E8A33D",
    "Layer 3-4":                     "#D98032",
    "Layer 4":                       "#C25E2A",
    "Layer 5":                       "#9E4A2E",
    "Layer 6":                       "#6E3A2E",
    "White matter":                  "#8C93A8",
    "Internal Capsule Caudoputamen": "#5B6478",
    "Hippocampus":                   "#3E7CB1",
    "Ventricle":                     "#7A5CA8",
}

#: Cortical order, pia to white matter, then the structures below it.
REGION_ORDER = ["Pia Layer 1", "Layer 2-3 lateral", "Layer 2-3 medial",
                "Layer 3-4", "Layer 4", "Layer 5", "Layer 6", "White matter",
                "Internal Capsule Caudoputamen", "Hippocampus", "Ventricle"]

#: Shorter forms for the bar chart, where the full name would eat the panel.
SHORT = {"Internal Capsule Caudoputamen": "Int. Capsule Caudoput."}

SCALEBAR_UM = 500.0
UM_PER_PX = 0.065  # osmFISH imaging frame: 65 nm per pixel


def load():
    sys.path.insert(0, str(REPO))
    from experiments.common.panel_export import load_panels
    if not DUMP.exists():
        raise SystemExit(
            f"no panel data at {DUMP}\n"
            "run `make panels-osmfish` first (needs ~8 GB)."
        )
    return load_panels(str(DUMP))


def _cells(ax, xy, keep=None):
    ss.spatial(ax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xy[:, 0].min() - 600, xy[:, 0].max() + 600)
    ax.set_ylim(xy[:, 1].min() - 600, xy[:, 1].max() + 600)


def _scalebar(ax, xy, um=SCALEBAR_UM):
    px = um / UM_PER_PX
    x0 = xy[:, 0].min()
    y0 = xy[:, 1].min() - 300
    ax.plot([x0, x0 + px], [y0, y0], color=INK, lw=1.3, solid_capstyle="butt",
            clip_on=False)
    ax.text(x0 + px / 2, y0 - 900, f"{um:g} $\\mu$m", fontsize=TICK, color=INK,
            ha="center", va="bottom", clip_on=False)


def _caption(ax, text):
    """A line under the map, in the margin rather than on the artwork."""
    ax.text(0.5, -0.015, text, transform=ax.transAxes, fontsize=TICK,
            color=INK, ha="center", va="top")


def panel_a(path, arr, meta):
    """The annotation."""
    xy, gt = arr["cell_coords"], arr["cell_gt_int"]
    names = meta["gt_names"]
    fig, ax = ss.panel(44, 62)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.10)
    for i, name in enumerate(names):
        m = gt == i
        ax.scatter(xy[m, 0], xy[m, 1], s=1.1, linewidths=0,
                   color=REGION_COLORS.get(name, GREY), rasterized=True)
    _cells(ax, xy)
    _scalebar(ax, xy)
    ss.save_panel(fig, path)


def panel_b(path, arr, meta):
    """The prediction, coloured by the region each cluster was matched to."""
    xy = arr["cell_coords"]
    matched = arr["cell_pred_matched"]
    names = meta["gt_names"]
    fig, ax = ss.panel(44, 62)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.10)
    for i, name in enumerate(names):
        m = matched == i
        ax.scatter(xy[m, 0], xy[m, 1], s=1.1, linewidths=0,
                   color=REGION_COLORS.get(name, GREY), rasterized=True)
    unmatched = matched < 0
    if unmatched.any():
        ax.scatter(xy[unmatched, 0], xy[unmatched, 1], s=1.1, linewidths=0,
                   color="#DCE2E8", rasterized=True)
    _cells(ax, xy)
    _caption(ax, f"ARI {meta['ari']:.3f}   NMI {meta['nmi']:.3f}")
    ss.save_panel(fig, path)


def panel_c(path, arr, meta):
    """Disagreement, which is where the boundary field should be."""
    xy = arr["cell_coords"]
    gt, matched = arr["cell_gt_int"], arr["cell_pred_matched"]
    valid = gt >= 0
    wrong = valid & (matched != gt)
    fig, ax = ss.panel(44, 62)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.10)
    ax.scatter(xy[valid & ~wrong, 0], xy[valid & ~wrong, 1], s=1.1,
               linewidths=0, color="#E3E8ED", rasterized=True)
    ax.scatter(xy[wrong, 0], xy[wrong, 1], s=1.1, linewidths=0, color=HOT,
               rasterized=True)
    _cells(ax, xy)
    pct = 100 * wrong.sum() / valid.sum()
    _caption(ax, f"{wrong.sum():,} of {valid.sum():,} cells ({pct:.0f}%)")
    ss.save_panel(fig, path)


def legend_strip(path, meta):
    """One key for a, b and c, laid out in the cortex's own order."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    names = [n for n in REGION_ORDER if n in meta["gt_names"]]
    fig = plt.figure(figsize=(ss.mm(140), ss.mm(9)))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    handles = [Patch(facecolor=REGION_COLORS[n], edgecolor="none", label=n)
               for n in names]
    ax.legend(handles=handles, loc="center", ncol=6, fontsize=TICK,
              frameon=False, handlelength=0.9, handleheight=0.9,
              handletextpad=0.4, columnspacing=1.1, labelspacing=0.45,
              borderpad=0.0)
    ss.save_panel(fig, path)
    plt.close(fig)


def panel_d(path, arr, meta):
    """Per-region recall, in cortical order."""
    gt, matched = arr["cell_gt_int"], arr["cell_pred_matched"]
    names = meta["gt_names"]
    rows = []
    for name in REGION_ORDER:
        if name not in names:
            continue
        i = names.index(name)
        m = gt == i
        rows.append((name, float((matched[m] == i).mean()) if m.any() else 0.0,
                     int(m.sum())))

    fig, ax = ss.panel(84, 46)
    fig.subplots_adjust(left=0.44, right=0.97, top=0.97, bottom=0.20)
    ys = range(len(rows))[::-1]
    for y, (name, acc, n) in zip(ys, rows):
        ax.barh(y, acc, height=0.74, color=REGION_COLORS[name],
                edgecolor="none")
        ax.text(acc + 0.012, y, f"{acc:.2f}", va="center", fontsize=TICK,
                color=INK)
    ax.set_yticks(list(ys))
    # The legend above carries the full names, so the longest one is
    # abbreviated here rather than eating a third of the panel's width.
    ax.set_yticklabels([f"{SHORT.get(n, n)}  ({c:,})" for n, _, c in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("recall")
    ss.save_panel(fig, path)


def panel_grid_clusters(path, arr, meta):
    """The partition on the grid the pipeline works on, before it reaches cells.

    Coloured through the same Hungarian matching as **b**, so the grid can be
    read against the annotation in **a** rather than against an arbitrary key.
    """
    g, lab = arr["grid_coords"], arr["grid_labels"]
    names = meta["gt_names"]
    lmap = {int(k): int(v) for k, v in meta["label_map"].items()}
    ks = sorted({int(v) for v in lab})
    fig, ax = ss.panel(44, 62)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.10)
    for k in ks:
        m = lab == k
        region = lmap.get(k, -1)
        colour = (REGION_COLORS.get(names[region], GREY)
                  if 0 <= region < len(names) else "#DCE2E8")
        ax.scatter(g[m, 0], g[m, 1], s=0.7, linewidths=0, color=colour,
                   rasterized=True)
    _cells(ax, g)
    _caption(ax, f"{len(ks)} clusters, {len(g):,} grid points")
    ss.save_panel(fig, path)


def _boundary(path, arr, key, title):
    """One boundary field.

    Colour limits are clipped to the 1st-99th percentile. Scaled to the
    maximum, the few hottest vertices flatten everything else into one dark
    slab -- which is what the legacy panel showed.
    """
    g, b = arr["grid_coords"], np.asarray(arr[key], dtype=float)
    lo, hi = np.percentile(b, [1, 99])
    fig, ax = ss.panel(44, 62)
    # The map stops at 0.70 rather than 0.78 so that the bar AND its rotated
    # label fit inside the 44 mm panel. save_panel keeps the declared size
    # rather than cropping to the drawn content, so anything past the right
    # edge is clipped in the figure: at 0.78 the label was cut off.
    fig.subplots_adjust(left=0.02, right=0.70, top=0.98, bottom=0.06)
    sc = ax.scatter(g[:, 0], g[:, 1], s=0.7, c=b, cmap="inferno",
                    vmin=lo, vmax=hi, linewidths=0, rasterized=True)
    _cells(ax, g)
    # centred on the panel, so the label -- which is longer than the bar it is
    # centred on -- has room above and below it
    cax = ax.inset_axes([1.05, 0.26, 0.06, 0.48])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(title, fontsize=TICK, labelpad=2)
    cb.ax.tick_params(labelsize=TICK, length=1.6, width=0.4, pad=1.2)
    cb.outline.set_linewidth(0.4)
    ss.save_panel(fig, path)


def panel_raw(path, arr, meta):
    _boundary(path, arr, "boundary_raw", "raw boundary score")


def panel_smooth(path, arr, meta):
    _boundary(path, arr, "boundary_smooth", "smoothed boundary score")


def main():
    ss.apply()
    arr, meta = load()

    panel_a(PANELS / "f3a.pdf", arr, meta)
    panel_b(PANELS / "f3b.pdf", arr, meta)
    panel_c(PANELS / "f3c.pdf", arr, meta)
    panel_grid_clusters(PANELS / "f3d.pdf", arr, meta)
    legend_strip(PANELS / "f3leg.pdf", meta)
    panel_raw(PANELS / "f3e.pdf", arr, meta)
    panel_smooth(PANELS / "f3f.pdf", arr, meta)
    panel_d(PANELS / "f3g.pdf", arr, meta)

    lay = Layout(
        name="fig3", width_mm=180.0, height_mm=146.0,
        panels=[
            Panel("a", PANELS / "f3a.pdf", x=4,   y=5,  label_dx=-3, label_dy=-4),
            Panel("b", PANELS / "f3b.pdf", x=48,  y=5,  label_dx=-3, label_dy=-4),
            Panel("c", PANELS / "f3c.pdf", x=92,  y=5,  label_dx=-3, label_dy=-4),
            Panel("d", PANELS / "f3d.pdf", x=136, y=5,  label_dx=-3, label_dy=-4),
            Panel(None, PANELS / "f3leg.pdf", x=20, y=68),
            Panel("e", PANELS / "f3e.pdf", x=4,   y=80, label_dx=-3, label_dy=-4),
            Panel("f", PANELS / "f3f.pdf", x=48,  y=80, label_dx=-3, label_dy=-4),
            Panel("g", PANELS / "f3g.pdf", x=96,  y=88, label_dx=-4, label_dy=-5),
        ],
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(lay.summary())
    print("\nwrote", lay.compose(OUT))


if __name__ == "__main__":
    main()
