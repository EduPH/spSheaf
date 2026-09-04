"""Figure 2 — the segmentation-free comparison and what the sheaf term contributes.

All values are measured. Per-slice competitor scores come from
``benchmarks/segfree/matched_selection.py`` (leave-one-slice-out over the stored
sweeps); spSheaf per-slice values are the 20-seed means of Table 1.

Two spSheaf estimators appear, and they are not interchangeable. Panel **a**
plots the 20-seed means, which average to 0.559. Panel **b** compares selection
rules, so its reference is the single representative seed scored under the
matched rule -- 0.557, the value in ``matched_selection_ari.csv`` alongside the
competitor values on that same panel. Reconciling the two by moving either
number would break the panel's agreement with its own source table; both are
labelled instead.

    python fig2_manuscript.py
    python verify_figure.py out/fig2.pdf --width-mm 180
"""

from __future__ import annotations

from pathlib import Path

import sheafstyle as ss
from layout import Layout, Panel

HERE = Path(__file__).parent
PANELS = HERE / "panels"
PANELS.mkdir(exist_ok=True)
#: The manuscript includes this by the path below, so write it there directly.
OUT = HERE / "out" / "fig2.pdf"

SLICES = ["0.04", "0.09", "0.14", "0.19", "0.24"]

# per-slice ARI under one held-out selection rule for every method
PER_SLICE = {
    "spSheaf":        [0.511, 0.525, 0.451, 0.660, 0.652],   # Table 1, 20-seed means
    "FICTURE":        [0.253, 0.239, 0.283, 0.275, 0.282],
    "Points2Regions": [0.296, 0.236, 0.235, 0.245, 0.278],
    "SSAM":           [0.166, 0.239, 0.302, 0.178, 0.153],
}

# effect of the selection rule: per-slice in-sample argmax -> held-out
RULE = {"Points2Regions": (0.387, 0.258),
        "FICTURE":        (0.274, 0.266),
        "SSAM":           (0.219, 0.208)}

ABLATION = [("sheaf", 0.571), ("sheaf_local", 0.557), ("identity", 0.544),
            ("neighbour", 0.541), ("shuffled", 0.526), ("procrustes", 0.519),
            ("grassmann", 0.518), ("projection", 0.518), ("random", 0.518),
            ("none", 0.496)]

BOUNDARY = [("projection", 0.331), ("grassmann", 0.329), ("procrustes", 0.328),
            ("identity", 0.271), ("neighbour", 0.259), ("sheaf", 0.251)]

TICK = ss.TYPE_PT["normal"]["tick"]


def panel_a(path):
    fig, ax = ss.panel(86, 48)
    fig.subplots_adjust(left=0.10, right=0.755, top=0.96, bottom=0.17)
    x = range(len(SLICES))
    # labels are nudged apart where two methods end within 0.01 of each other
    dy = {"spSheaf": 0, "FICTURE": 4.5, "Points2Regions": -4.5, "SSAM": 0}
    for name in ("spSheaf", "FICTURE", "Points2Regions", "SSAM"):
        lead = name == "spSheaf"
        ax.plot(x, PER_SLICE[name], marker="o",
                color=ss.METHOD_COLORS[name],
                lw=1.4 if lead else 0.9,
                ms=3.2 if lead else 2.4,
                zorder=3 if lead else 2)
        ax.annotate(name, (len(SLICES) - 1, PER_SLICE[name][-1]),
                    xytext=(6, dy[name]), textcoords="offset points",
                    color=ss.METHOD_COLORS[name], fontsize=TICK, va="center")
    ax.set_xticks(list(x)); ax.set_xticklabels(SLICES)
    ax.set_xlim(-0.25, len(SLICES) - 0.75)
    ax.set_ylim(0.10, 0.72)
    ax.set_xlabel("MERFISH slice (Bregma)")
    ax.set_ylabel("ARI")
    ss.save_panel(fig, path)


def panel_b(path):
    fig, ax = ss.panel(72, 48)
    fig.subplots_adjust(left=0.16, right=0.62, top=0.96, bottom=0.17)
    dy = {"Points2Regions": -4.5, "FICTURE": 4.5, "SSAM": 0}
    for name, (a, b) in RULE.items():
        c = ss.METHOD_COLORS[name]
        ax.plot([0, 1], [a, b], marker="o", ms=2.8, lw=1.1, color=c)
        ax.annotate(f"{a:.3f}", (0, a), xytext=(3, 5),
                    textcoords="offset points", color=c, fontsize=TICK,
                    va="bottom", ha="left")
        ax.annotate(f"{name} {b:.3f}", (1, b), xytext=(6, dy[name]),
                    textcoords="offset points", color=c, fontsize=TICK, va="center")
    # The reference is the single representative seed, so that it comes from
    # the same table as the competitor values on this panel. Panel a's 20-seed
    # means average to 0.559; saying which is which here stops the two panels
    # reading as a contradiction.
    ax.axhline(0.557, color=ss.METHOD_COLORS["spSheaf"], lw=0.8, ls=(0, (3, 2)))
    ax.annotate("spSheaf 0.557\n(single seed)", (1, 0.557), xytext=(6, 0),
                textcoords="offset points", color=ss.METHOD_COLORS["spSheaf"],
                fontsize=TICK, va="center", linespacing=1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["per-slice\nargmax", "held-out\n(matched)"])
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(0.15, 0.62)
    ax.set_ylabel("mean ARI")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ss.save_panel(fig, path)


def _barh(ax, rows, xlim, xlabel, ref=None, ref_label=None):
    names = [n for n, _ in rows][::-1]
    vals = [v for _, v in rows][::-1]
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.barh(i, v, height=0.72, **ss.arm_style(n))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylim(-1.15 if ref is not None else -0.7, len(names) - 0.3)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    if ref is not None:
        ax.axvline(ref, color="#555555", lw=0.6, ls=(0, (2, 2)))
        ax.text(ref, -0.78, ref_label, fontsize=4, color="#555555",
                va="center", ha="center")


def panel_c(path):
    fig, ax = ss.panel(86, 46)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.97, bottom=0.20)
    _barh(ax, ABLATION, (0.45, 0.605), "ARI", ref=0.496, ref_label="no weighting")
    ax.set_xticks([0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60])
    ss.save_panel(fig, path)


def panel_d(path):
    fig, ax = ss.panel(72, 46)
    fig.subplots_adjust(left=0.26, right=0.97, top=0.97, bottom=0.20)
    _barh(ax, BOUNDARY, (0, 0.38), "boundary AP @ 15 $\\mu$m")
    ss.save_panel(fig, path)


FIG2 = Layout(
    name="fig2", width_mm=180.0, height_mm=112.0,
    panels=[
        Panel("a", PANELS / "f2a.pdf", x=8,   y=6,  label_dx=-4, label_dy=-5),
        Panel("b", PANELS / "f2b.pdf", x=101, y=6,  label_dx=-4, label_dy=-5),
        Panel("c", PANELS / "f2c.pdf", x=8,   y=62, label_dx=-4, label_dy=-5),
        Panel("d", PANELS / "f2d.pdf", x=101, y=62, label_dx=-4, label_dy=-5),
    ],
)


def main():
    ss.apply()
    panel_a(PANELS / "f2a.pdf")
    panel_b(PANELS / "f2b.pdf")
    panel_c(PANELS / "f2c.pdf")
    panel_d(PANELS / "f2d.pdf")
    print(FIG2.summary())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("\nwrote", FIG2.compose(OUT))


if __name__ == "__main__":
    main()
