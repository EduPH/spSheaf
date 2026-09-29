"""The ten-arm edge-score ablation on the two SpaceTx sections, at each
section's selected configuration (tissue reference). Same arms and controls
as Fig. 2d; bars are means over five k-means seeds on one section, so there
are no per-slice dots.

    python sup_ablation_spacetx.py          # reads reported/tissue/spacetx/ when SPSHEAF_DATA is not the tissue root
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import sheafstyle as ss
from spsheaf import paths
from verify_figure import verify

HERE = Path(__file__).parent
REPORTED = HERE.parent / "reported" / "tissue" / "spacetx"


def _table(key):
    """The live results tree first (SPSHEAF_DATA at the tissue root), else the
    tracked copy, so a clone draws the figure without the data."""
    try:
        p = paths.results("spacetx") / f"sheaf_ablation_{key}.csv"
        if p.exists():
            return p
    except Exception:
        pass
    return REPORTED / f"sheaf_ablation_{key}.csv"
OUTDIR = HERE / "out"
W = 180.0
ARM_LABEL = {"sheaf": "sheaf (tissue ref., default)", "sheaf_local": "sheaf (local ref.)",
             "identity": "identity", "neighbour": "neighbour", "procrustes": "procrustes",
             "projection": "projection", "grassmann": "grassmann",
             "CONTROL:shuffled": "shuffled", "CONTROL:random": "random", "CONTROL:none": "none"}
ORDER = ["sheaf", "sheaf_local", "neighbour", "identity", "procrustes", "projection",
         "grassmann", "CONTROL:shuffled", "CONTROL:random", "CONTROL:none"]
SECTIONS = [("MERFISH_Allen_Prototype", "Allen prototype MERFISH, $k$ = 6"),
            ("BARISTASeq", "BaristaSeq, $k$ = 6")]


def main():
    ss.apply()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T, TICK = ss.TYPE_PT["normal"]["text"], ss.TYPE_PT["normal"]["tick"]
    height = 52.0
    fig, axes = ss.panel_grid(W, height, 1, 2)
    fig.subplots_adjust(left=36.0 / W, right=0.985, top=1 - 6.0 / height,
                        bottom=10.0 / height, wspace=0.55)
    for ax, (key, title), letter in zip(axes, SECTIONS, "ab"):
        d = pd.read_csv(_table(key)).set_index("arm")
        arms = [a for a in ORDER if a in d.index][::-1]
        for i, a in enumerate(arms):
            v, sd = d.loc[a, "ARI"], d.loc[a, "ARI_sd"]
            ax.barh(i, v, height=0.72, **ss.arm_style(a.replace("CONTROL:", "")))
            ax.errorbar(v, i, xerr=sd, fmt="none", ecolor="#333333", elinewidth=0.5, capsize=1.2)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([ARM_LABEL[a] for a in arms], fontsize=TICK)
        ax.set_ylim(-0.7, len(arms) - 0.3)
        lo = np.floor((d.ARI.min() - 0.03) * 20) / 20
        ax.set_xlim(lo, min(1.0, d.ARI.max() + 0.03))
        ax.set_xlabel("ARI (mean $\\pm$ s.d. over 5 seeds, grid level)", fontsize=T)
        ax.set_title(title, fontsize=T, pad=3, color="#333333")
        ax.tick_params(labelsize=TICK, length=1.6, pad=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text(-0.55, 1.04, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")
    out = OUTDIR / "sup_ablation_spacetx.pdf"
    ss.save_panel(fig, out, strict=False)
    print(f"wrote {out}")
    for p in verify(out, width_mm=W)[0]:
        print("  !", p)


if __name__ == "__main__":
    main()
