"""Extended Data Fig. 3 -- whole-section data at scale.

a  Stereo-seq E16.5 embryo: the run output raster (no panel data were dumped
   for that run; it is placed as a raster, see ed_composites.py).
b  MERSCOPE hemisection: the 15 sheaf domains, drawn from
   ``domain_labels.csv``.
c  The smoothed boundary field over the same grid (``boundary_field.csv``).
d  Allen CCFv3 major divisions rigidly registered onto the same grid
   (``allen_reference_labels.csv``), the independent anatomical reference
   the domains are scored against (metrics_vs_allen.csv).

    python ed_scale.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sheafstyle as ss
from layout import Layout, Panel, raster_panel, row
from spsheaf import paths

HERE = Path(__file__).parent
PANELS = HERE / "panels"
PANELS.mkdir(exist_ok=True)
OUT = HERE / "out" / "ed_scale.pdf"
SRC = HERE / "assets"

RES = paths.results("brain_receptor") / "output_brain_receptor"

# 15 domains: Okabe-Ito plus tab20 fill, fixed order
DOMAIN_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
                 "#56B4E9", "#F0E442", "#8C8C8C", "#1F77B4", "#9467BD",
                 "#8C564B", "#7F7F7F", "#BCBD22", "#17BECF", "#2CA02C"]
ALLEN_ORDER = ["Isocortex", "OLF", "HPF", "CTXsp", "STR", "PAL", "TH", "HY",
               "fiber tracts", "VS"]
ALLEN_COLORS = ["#0072B2", "#56B4E9", "#009E73", "#2CA02C", "#E69F00",
                "#D55E00", "#CC79A7", "#9467BD", "#8C8C8C", "#F0E442"]

W_MM, H_MM = 56.0, 44.0


#: b, c and d share one axes rectangle and one data window, so the three
#: brains come out at the same size and height; the colour bar of c and the
#: legend of d sit in the band below the map
AX_RECT = [0.01, 0.17, 0.98, 0.75]


def _limits():
    d = pd.read_csv(RES / "domain_labels.csv")
    pad = 0.02 * max(np.ptp(d.x), np.ptp(d.y))
    return (d.x.min() - pad, d.x.max() + pad), (d.y.min() - pad, d.y.max() + pad)


def _map(ax, x, y, colors):
    ss.spatial(ax)
    sc = ax.scatter(x, y, s=1.6, c=colors, marker="s", linewidths=0)
    ss.rasterize(sc, dpi=600)
    xl, yl = _limits()
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    return sc


def panel_domains(path):
    d = pd.read_csv(RES / "domain_labels.csv")
    fig, ax = ss.panel(W_MM, H_MM); ax.remove(); ax = fig.add_axes(AX_RECT)
    _map(ax, d.x, d.y, np.array(DOMAIN_COLORS)[d.label.values % len(DOMAIN_COLORS)])
    ax.set_title(f"{d.label.nunique()} sheaf domains", fontsize=ss.TYPE_PT["normal"]["text"], pad=2)
    ss.save_panel(fig, path)


def panel_boundary(path):
    b = pd.read_csv(RES / "boundary_field.csv")
    fig, ax = ss.panel(W_MM, H_MM); ax.remove(); ax = fig.add_axes(AX_RECT)
    ss.spatial(ax)
    v = b.boundary_smooth.values
    sc = ax.scatter(b.x, b.y, s=1.6, c=v, cmap="inferno", marker="s", linewidths=0,
                    vmin=np.percentile(v, 1), vmax=np.percentile(v, 99))
    ss.rasterize(sc, dpi=600)
    xl, yl = _limits()
    ax.set_xlim(*xl); ax.set_ylim(*yl)
    cax = fig.add_axes([0.25, 0.075, 0.5, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cax.set_title("smoothed boundary score", fontsize=ss.TYPE_PT["normal"]["tick"], pad=2)
    cb.ax.tick_params(labelsize=ss.TYPE_PT["normal"]["tick"], length=1.5, pad=1.5)
    cb.outline.set_linewidth(0.4)
    ax.set_title("sheaf boundary field", fontsize=ss.TYPE_PT["normal"]["text"], pad=2)
    ss.save_panel(fig, path)


def panel_allen(path):
    a = pd.read_csv(RES / "allen_reference_labels.csv")
    m = a.allen_division != "outside"
    idx = a.loc[m, "allen_division"].map({k: i for i, k in enumerate(ALLEN_ORDER)}).values
    fig, ax = ss.panel(W_MM, H_MM); ax.remove(); ax = fig.add_axes(AX_RECT)
    _map(ax, a.x[m], a.y[m], np.array(ALLEN_COLORS)[idx])
    ax.set_title("Allen CCFv3 divisions (rigid registration)",
                 fontsize=ss.TYPE_PT["normal"]["text"], pad=2)
    for i, nm in enumerate(ALLEN_ORDER):
        ax.scatter([], [], s=6, c=ALLEN_COLORS[i], label=nm, linewidths=0, marker="s")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=5,
              fontsize=ss.TYPE_PT["small"]["tick"], frameon=False,
              handletextpad=0.3, columnspacing=0.8, labelspacing=0.2,
              borderaxespad=0.0)
    ss.save_panel(fig, path)


def main():
    ss.apply()
    pa, dpi = raster_panel(SRC / "ed_stereoseq.png", PANELS / "eds_a.pdf", 108.0)
    from pypdf import PdfReader
    ha = float(PdfReader(str(pa)).pages[0].mediabox.height) / (72.0 / 25.4)
    panel_domains(PANELS / "eds_b.pdf")
    panel_boundary(PANELS / "eds_c.pdf")
    panel_allen(PANELS / "eds_d.pdf")
    xs = row([W_MM] * 3, 180.0, margin=6.0)
    y2 = 6.0 + ha + 9.0
    lay = Layout(
        name="ed_scale", width_mm=180.0, height_mm=y2 + H_MM + 3.0,
        panels=[
            Panel("a", pa, (180.0 - 108.0) / 2, 6.0, label_dx=-4, label_dy=-5),
            Panel("b", PANELS / "eds_b.pdf", xs[0], y2, label_dx=-4, label_dy=-3),
            Panel("c", PANELS / "eds_c.pdf", xs[1], y2, label_dx=-4, label_dy=-3),
            Panel("d", PANELS / "eds_d.pdf", xs[2], y2, label_dx=-4, label_dy=-3),
        ],
    )
    print(lay.summary(), f"\nStereo-seq raster at {dpi:.0f} dpi")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("wrote", lay.compose(OUT))


if __name__ == "__main__":
    main()
