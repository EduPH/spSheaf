"""Worked example: a Fig. 2 skeleton built entirely from the pipeline.

Every number here is real except panel **d**, which draws synthetic points only
to exercise the spatial-panel rule and rasterization. Replace panel d's data
loader with the real slice and the figure is production output.

    python example_fig2.py           # writes panels/*.pdf and fig2.pdf
    python verify_figure.py fig2.pdf
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sheafstyle as ss
from layout import Layout, Panel

HERE = Path(__file__).parent
PANELS = HERE / "panels"
PANELS.mkdir(exist_ok=True)

# --- real numbers from the manuscript ---------------------------------------

SEGFREE = [  # Table 3, MERFISH mean over 5 slices
    ("spSheaf", 0.557),
    ("Points2Regions", 0.387),
    ("FICTURE", 0.274),
    ("SSAM", 0.219),
]

ABLATION = [  # Table 2, mean over 5 slices x 5 seeds
    ("sheaf", 0.571),
    ("sheaf_local", 0.557),
    ("identity", 0.544),
    ("neighbour", 0.541),
    ("shuffled", 0.526),
    ("procrustes", 0.519),
    ("grassmann", 0.518),
    ("projection", 0.518),
    ("random", 0.518),
    ("none", 0.496),
]

BOUNDARY_AP = [  # Section 'The disagreement field...', AP@15 um
    ("projection", 0.331),
    ("grassmann", 0.329),
    ("procrustes", 0.328),
    ("identity", 0.271),
    ("neighbour", 0.259),
    ("sheaf", 0.251),
]


def panel_a(path):
    """Segmentation-free head-to-head."""
    fig, ax = ss.panel(52, 42)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.94, bottom=0.32)
    names = [n for n, _ in SEGFREE]
    vals = [v for _, v in SEGFREE]
    ax.bar(
        names,
        vals,
        color=[ss.METHOD_COLORS[n] for n in names],
        width=0.68,
    )
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
                fontsize=ss.TYPE_PT["normal"]["tick"])
    ax.set_ylabel("ARI (MERFISH)")
    ax.set_ylim(0, 0.66)
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ss.save_panel(fig, path)


def panel_b(path):
    """Edge-score ablation, ten arms, controls hatched."""
    fig, ax = ss.panel(60, 42)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.95, bottom=0.20)
    names = [n for n, _ in ABLATION][::-1]
    vals = [v for _, v in ABLATION][::-1]
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.barh(i, v, height=0.7, **ss.arm_style(n))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlim(0.45, 0.60)
    ax.set_xlabel("ARI")
    ax.axvline(0.496, color="#666666", lw=0.5, ls=(0, (2, 2)))
    ax.text(0.4975, 0.15, "no weighting", fontsize=4, color="#666666",
            va="bottom", ha="left")
    ss.save_panel(fig, path)


def panel_c(path):
    """Direct boundary detection -- the inversion."""
    fig, ax = ss.panel(52, 42)
    fig.subplots_adjust(left=0.32, right=0.97, top=0.95, bottom=0.20)
    names = [n for n, _ in BOUNDARY_AP][::-1]
    vals = [v for _, v in BOUNDARY_AP][::-1]
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.barh(i, v, height=0.7, **ss.arm_style(n))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("boundary AP @ 15 $\\mu$m")
    ax.set_xlim(0, 0.38)
    ss.save_panel(fig, path)


def panel_d(path, n=40_000, seed=0):
    """Spatial domain map. SYNTHETIC DATA -- swap in the real slice."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1, 1, size=(n, 2))
    keep = (xy**2).sum(1) < 1.0
    xy = xy[keep]
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    r = np.linalg.norm(xy, axis=1)
    labels = ((theta + np.pi) / (2 * np.pi) * 5).astype(int) + (r > 0.55).astype(int) * 5
    labels %= 8

    fig, ax = ss.panel(60, 46)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ss.spatial(ax)
    cmap = ["#0072B2", "#56B4E9", "#009E73", "#E69F00",
            "#CC79A7", "#B266A0", "#8C8C8C", "#333333"]
    sc = ax.scatter(
        xy[:, 0], xy[:, 1],
        c=[cmap[l] for l in labels],
        s=0.35, linewidths=0, marker=".",
    )
    ss.rasterize(sc)
    ss.save_panel(fig, path)


FIG2 = Layout(
    name="fig2",
    width_mm=180.0,
    height_mm=104.0,
    # Panels sit below a 6 mm label gutter; labels are placed above each panel
    # (negative label_dy) so they never collide with an axis label.
    panels=[
        Panel("a", PANELS / "fig2a.pdf", x=5,   y=6,  label_dx=-3, label_dy=-5),
        Panel("b", PANELS / "fig2b.pdf", x=61,  y=6,  label_dx=-3, label_dy=-5),
        Panel("c", PANELS / "fig2c.pdf", x=126, y=6,  label_dx=-3, label_dy=-5),
        Panel("d", PANELS / "fig2d.pdf", x=5,   y=56, label_dx=-3, label_dy=-5),
    ],
)


def main():
    ss.apply()
    panel_a(PANELS / "fig2a.pdf")
    panel_b(PANELS / "fig2b.pdf")
    panel_c(PANELS / "fig2c.pdf")
    panel_d(PANELS / "fig2d.pdf")
    print(FIG2.summary())
    out = FIG2.compose(HERE / "fig2.pdf")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
