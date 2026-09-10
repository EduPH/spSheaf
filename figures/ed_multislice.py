"""Extended Data Fig. 7 -- multi-slice alignment across serial sections.

Drawn from ``data/results/merfish/panels/multislice.npz`` (written by
``python -m experiments.merfish.multislice_panels``).

a  The five slices after gene-centroid registration, in the common frame,
   coloured by Hungarian-matched domain; per-slice ARI in the title.
b  Cross-slice label agreement for the four adjacent pairs, with the
   self-alignment upper bound (a slice against a shifted copy of itself).

    python ed_multislice.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sheafstyle as ss
from layout import Layout, Panel
from spsheaf import paths

HERE = Path(__file__).parent
PANELS = HERE / "panels"
PANELS.mkdir(exist_ok=True)
OUT = HERE / "out" / "ed_multislice.pdf"

DOMAIN_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                 "#D55E00", "#56B4E9", "#F0E442", "#8C8C8C"]


def load():
    npz = paths.results("merfish") / "panels" / "multislice.npz"
    if not npz.exists():
        raise SystemExit(f"{npz} missing; run  python -m experiments.merfish.multislice_panels")
    return np.load(npz), json.loads(npz.with_suffix(".json").read_text())


def panel_a(arr, meta, path):
    K = len(meta["slices"])
    fig, axes = ss.panel_grid(168, 38, 1, K)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.84, bottom=0.02, wspace=0.05)
    # one frame for all five, so shifts are visible
    xs = np.concatenate([arr[f"grid_aligned_{k}"][:, 0] for k in range(K)])
    ys = np.concatenate([arr[f"grid_aligned_{k}"][:, 1] for k in range(K)])
    lim = (xs.min() - 30, xs.max() + 30, ys.min() - 30, ys.max() + 30)
    cmap = np.array(DOMAIN_COLORS)
    for k, ax in enumerate(axes):
        ss.spatial(ax)
        g, lab = arr[f"grid_aligned_{k}"], arr[f"labels_{k}"]
        sc = ax.scatter(g[:, 0], g[:, 1], s=0.7, c=cmap[lab % len(cmap)],
                        marker="s", linewidths=0)
        ss.rasterize(sc, dpi=600)
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
        tag = meta["slices"][k].replace("MERFISH_", "").replace(".h5ad", "")
        dx, dy = meta["shift_um"][k]
        ax.set_title(f"Bregma {tag}   ARI {meta['ari'][k]:.3f}\n"
                     f"shift ({dx:+.1f}, {dy:+.1f}) $\\mu$m",
                     fontsize=ss.TYPE_PT["normal"]["text"], pad=2, linespacing=1.2)
    ss.save_panel(fig, path)


def panel_b(meta, path):
    fig, ax = ss.panel(84, 44)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.92, bottom=0.22)
    tags = [s.replace("MERFISH_", "").replace(".h5ad", "") for s in meta["slices"]]
    names = [f"{tags[i]} → {tags[j]}" for i, j in meta["pairs"]]
    vals = meta["agreement"]
    ax.bar(range(len(vals)), vals, width=0.68, color=ss.METHOD_COLORS["spSheaf"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                fontsize=ss.TYPE_PT["normal"]["tick"])
    ub = meta["self_alignment"]["agreement"]
    ax.axhline(ub, color="#333333", lw=0.7, ls=(0, (3, 2)))
    ax.text(len(vals) - 0.6, ub + 0.015, f"self-alignment upper bound {ub:.3f}",
            ha="right", va="bottom", fontsize=ss.TYPE_PT["normal"]["tick"], color="#333333")
    ax.axhline(meta["mean_agreement"], color=ss.METHOD_COLORS["spSheaf"], lw=0.7, ls=(0, (1, 1.5)))
    ax.text(-0.35, 0.985, f"mean of the four pairs {meta['mean_agreement']:.3f} (dotted)",
            ha="left", va="top", fontsize=ss.TYPE_PT["normal"]["tick"],
            color=ss.METHOD_COLORS["spSheaf"])
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("adjacent slice pair")
    ax.set_ylabel("label agreement")
    ss.save_panel(fig, path)


def main():
    ss.apply()
    arr, meta = load()
    panel_a(arr, meta, PANELS / "edms_a.pdf")
    panel_b(meta, PANELS / "edms_b.pdf")
    lay = Layout(
        name="ed_multislice", width_mm=180.0, height_mm=102.0,
        panels=[
            Panel("a", PANELS / "edms_a.pdf", x=6, y=5, label_dx=-4, label_dy=-3),
            Panel("b", PANELS / "edms_b.pdf", x=48, y=52, label_dx=-4, label_dy=-5),
        ],
    )
    print(lay.summary())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("wrote", lay.compose(OUT))


if __name__ == "__main__":
    main()
