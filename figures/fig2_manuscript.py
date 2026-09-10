"""Figure 2 — the segmentation-free comparison and what the sheaf term contributes.

All values are measured. Per-slice competitor scores come from
``benchmarks/segfree/matched_selection.py`` (leave-one-slice-out over the stored
sweeps); spSheaf per-slice values are the 20-seed means of Table 1. The domain
maps of panel **a** are read from the dump written by
``experiments.merfish.segfree_panels``; the per-slice ablation and boundary
values behind the dots of panels **d** and **e** are read from the
``sheaf_ablation_MERFISH_<slice>.csv`` and ``boundary_eval_MERFISH_<slice>.csv``
tables that ``make revision`` writes (the bar heights are the five-slice means
of those same tables).

spSheaf runs one fixed configuration on every slice, so its reference on
every panel is the same number: the 20-seed mean of Table 1 (also the
spSheaf row of ``matched_selection_ari.csv``). Panel **a**'s map is the cached
z = 4.5 um plane (ARI 0.62 on slice 0.19); Table 1's 0.660 is the z = 0 plane.

    python fig2_manuscript.py
    python verify_figure.py out/fig2.pdf --width-mm 180
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sheafstyle as ss
from layout import Layout, Panel
from spsheaf import paths

HERE = Path(__file__).parent
PANELS = HERE / "panels"
PANELS.mkdir(exist_ok=True)
#: The manuscript includes this by the path below, so write it there directly.
OUT = HERE / "out" / "fig2.pdf"

SLICES = ["0.04", "0.09", "0.14", "0.19", "0.24"]

# per-slice ARI: spSheaf at its one fixed configuration (20-seed means of
# Table 1); competitors under leave-one-slice-out selection, on the clean
# sections. Both read from matched_selection.py's output so the figure cannot
# drift from the table.
_MS = HERE.parent / "benchmarks" / "segfree" / "results"


def _per_slice():
    df = pd.read_csv(_MS / "matched_selection_per_slice_ari.csv")
    out = {}
    for name, g in df.groupby("method", sort=False):
        out[name.replace(" (de novo)", "")] = [float(v) for v in g.sort_values("slice").held_out]
    return out


def _rules():
    df = pd.read_csv(_MS / "matched_selection_ari.csv")
    df = df[df.method != "spSheaf"]
    return {r.method.replace(" (de novo)", ""): (float(r.per_slice_argmax), float(r.loo_cv))
            for r in df.itertuples()}


PER_SLICE = _per_slice()
RULE = _rules()

MAP_SLICE = "s0.19"
MAP_METHODS = ["spSheaf", "FICTURE", "Points2Regions", "SSAM"]


def _results(*parts):
    return paths.results().joinpath(*parts)


def _per_slice_table(prefix, col, key="arm"):
    """{arm: [value on each slice]} from the five per-slice revision tables.

    The tables spell the null controls ``CONTROL:shuffled`` etc.; the figure
    uses the short names of the manuscript (Section 4.9).
    """
    out = {}
    for sl in SLICES:
        df = pd.read_csv(_results("merfish", f"{prefix}_MERFISH_{sl}.csv"))
        for k, v in zip(df[key], df[col]):
            out.setdefault(str(k).replace("CONTROL:", ""), []).append(float(v))
    return out


#: Arm names as the manuscript spells them (Section 4.9). ``sheaf_local`` is the
#: method (local reference); ``sheaf`` is the same score with the tissue
#: reference, the optional term.
ARM_LABEL = {"sheaf_local": "sheaf (local ref.)", "sheaf": "sheaf (tissue ref.)",
             "identity": "identity", "neighbour": "neighbour", "grassmann": "grassmann",
             "projection": "projection", "procrustes": "procrustes",
             "shuffled": "shuffled", "random": "random", "none": "none"}


def _arm_means(col):
    """[(arm, five-slice mean)] sorted high to low, from the revision tables."""
    per = _per_slice_table("sheaf_ablation", col)
    return sorted(((k, float(np.mean(v))) for k, v in per.items()), key=lambda kv: -kv[1])


ABLATION = _arm_means("ARI")
BOUNDARY = [(k, v) for k, v in _arm_means("AP") if not np.isnan(v)]

TICK = ss.TYPE_PT["normal"]["tick"]


#: Domain colours, one per annotated MERFISH domain (Okabe-Ito + two).
DOMAIN_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                 "#D55E00", "#56B4E9", "#F0E442", "#8C8C8C"]


def panel_maps(path):
    """Ground truth and every method's domains on one slice, per cell."""
    npz = _results("merfish", "panels", f"segfree_maps_{MAP_SLICE}.npz")
    if not npz.exists():
        raise SystemExit(
            f"{npz} missing; run  python -m experiments.merfish.segfree_panels "
            f"--slice MERFISH_{MAP_SLICE[1:]}.h5ad")
    arr = np.load(npz)
    meta = json.loads(npz.with_suffix(".json").read_text())
    x, y, gt, valid = arr["x"], arr["y"], arr["gt"], arr["valid"]
    cols = [("annotation", gt, None)] + [
        (m, arr[f"pred_{m}"], meta["ari"][m]) for m in MAP_METHODS]
    w, h = 180 - 16, 36
    fig, axes = ss.panel_grid(w, h, 1, len(cols))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.02, wspace=0.04)
    cmap = np.array(DOMAIN_COLORS)
    for ax, (name, lab, ari) in zip(axes, cols):
        ss.spatial(ax)
        m = valid & (lab >= 0)
        sc = ax.scatter(x[m], y[m], s=0.9, c=cmap[lab[m] % len(cmap)],
                        linewidths=0, marker="o")
        ss.rasterize(sc, dpi=600)
        title = name if ari is None else f"{name}  ARI {ari:.2f}"
        color = ss.METHOD_COLORS.get(name, "#333333")
        ax.set_title(title, fontsize=ss.TYPE_PT["normal"]["text"], color=color, pad=2)
    ss.save_panel(fig, path)
    return meta["gt_names"]


def legend_strip(path, gt_names):
    """One-row legend for the annotated domains, placed under the maps."""
    fig, ax = ss.panel(180 - 16, 5)
    ax.set_axis_off()
    cmap = np.array(DOMAIN_COLORS)
    for i, nm in enumerate(gt_names):
        ax.scatter([], [], s=9, c=cmap[i], label=nm, linewidths=0)
    ax.legend(loc="center", ncol=len(gt_names), frameon=False,
              fontsize=ss.TYPE_PT["normal"]["tick"], handletextpad=0.3,
              columnspacing=1.4, borderaxespad=0)
    ss.save_panel(fig, path, strict=False)


def panel_a(path):
    fig, ax = ss.panel(86, 44)
    fig.subplots_adjust(left=0.10, right=0.755, top=0.96, bottom=0.17)
    x = range(len(SLICES))
    # labels are nudged apart where two methods end within 0.01 of each other
    dy = {"spSheaf": 7, "FICTURE": -6, "Points2Regions": 0, "SSAM": 0}
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
    fig, ax = ss.panel(72, 44)
    fig.subplots_adjust(left=0.16, right=0.62, top=0.96, bottom=0.17)
    dy = {"Points2Regions": 5, "FICTURE": -5, "SSAM": 0}
    # SSAM's argmax (0.464) now sits just under Points2Regions' (0.510), so its
    # left-hand label is pushed below the marker to keep the two apart.
    dyl = {"Points2Regions": (3, 4), "FICTURE": (3, -9), "SSAM": (3, -9)}
    for name, (a, b) in RULE.items():
        c = ss.METHOD_COLORS[name]
        ax.plot([0, 1], [a, b], marker="o", ms=2.8, lw=1.1, color=c)
        ax.annotate(f"{a:.3f}", (0, a), xytext=dyl[name],
                    textcoords="offset points", color=c, fontsize=TICK,
                    va="bottom" if dyl[name][1] > 0 else "top", ha="left")
        ax.annotate(f"{name} {b:.3f}", (1, b), xytext=(6, dy[name]),
                    textcoords="offset points", color=c, fontsize=TICK, va="center")
    # The reference is the single representative seed, so that it comes from
    # the same table as the competitor values on this panel. Panel a's 20-seed
    # means average to the same number as panel a's line; read from the CSV
    # reading as a contradiction.
    sp = float(pd.read_csv(_MS / "matched_selection_ari.csv").query("method == 'spSheaf'").loo_cv.iloc[0])
    ax.axhline(sp, color=ss.METHOD_COLORS["spSheaf"], lw=0.8, ls=(0, (3, 2)))
    ax.annotate(f"spSheaf {sp:.3f}\n(20-seed mean)", (1, sp), xytext=(6, 2),
                textcoords="offset points", color=ss.METHOD_COLORS["spSheaf"],
                fontsize=TICK, va="bottom", linespacing=1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["per-slice\nargmax", "held-out\n(matched)"])
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(0.08, 0.66)
    ax.set_ylabel("mean ARI")
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ss.save_panel(fig, path)


def _barh(ax, rows, per_slice, xlim, xlabel):
    """Bars are five-slice means; open dots are the five per-slice values."""
    names = [n for n, _ in rows][::-1]
    vals = [v for _, v in rows][::-1]
    rng = np.random.default_rng(0)
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.barh(i, v, height=0.72, **ss.arm_style(n))
        pts = per_slice.get(n)
        if pts is not None:
            jit = rng.uniform(-0.18, 0.18, size=len(pts))
            ax.scatter(pts, i + jit, s=5, facecolors="white",
                       edgecolors="#333333", linewidths=0.4, zorder=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([ARM_LABEL.get(n, n) for n in names])
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)


def panel_c(path):
    per_slice = _per_slice_table("sheaf_ablation", "ARI")
    means = {k: float(np.mean(v)) for k, v in per_slice.items()}
    rows = sorted(means.items(), key=lambda kv: -kv[1])
    fig, ax = ss.panel(86, 40)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.97, bottom=0.23)
    _barh(ax, rows, per_slice, (0.36, 0.68), "ARI (bar: mean of 5 slices; dots: slices)")
    ss.save_panel(fig, path)


def panel_d(path):
    # the ablation table carries the boundary AP of every scored arm, including
    # the local-reference score, on the same cache as panel c
    per_slice = _per_slice_table("sheaf_ablation", "AP")
    per_slice = {k: v for k, v in per_slice.items() if not np.isnan(v).any()}
    means = {k: float(np.mean(v)) for k, v in per_slice.items()}
    rows = sorted(means.items(), key=lambda kv: -kv[1])
    fig, ax = ss.panel(72, 40)
    fig.subplots_adjust(left=0.32, right=0.97, top=0.97, bottom=0.23)
    _barh(ax, rows, per_slice, (0, 0.36), "boundary AP @ 15 $\\mu$m")
    ss.save_panel(fig, path)


FIG2 = Layout(
    name="fig2", width_mm=180.0, height_mm=135.0,
    panels=[
        Panel("a", PANELS / "f2maps.pdf", x=8, y=4, label_dx=-4, label_dy=-2),
        Panel(None, PANELS / "f2leg.pdf", x=8, y=40),
        Panel("b", PANELS / "f2a.pdf", x=8,   y=47, label_dx=-4, label_dy=-5),
        Panel("c", PANELS / "f2b.pdf", x=101, y=47, label_dx=-4, label_dy=-5),
        Panel("d", PANELS / "f2c.pdf", x=8,   y=94, label_dx=-4, label_dy=-5),
        Panel("e", PANELS / "f2d.pdf", x=101, y=94, label_dx=-4, label_dy=-5),
    ],
)


def main():
    ss.apply()
    gt_names = panel_maps(PANELS / "f2maps.pdf")
    legend_strip(PANELS / "f2leg.pdf", gt_names)
    panel_a(PANELS / "f2a.pdf")
    panel_b(PANELS / "f2b.pdf")
    panel_c(PANELS / "f2c.pdf")
    panel_d(PANELS / "f2d.pdf")
    print(FIG2.summary())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("\nwrote", FIG2.compose(OUT))


if __name__ == "__main__":
    main()
