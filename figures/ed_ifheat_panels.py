"""The chromatin-mark archetype heatmap of the IF Extended Data figure.

``experiments/takei/make_zone_heatmaps.py --panel if`` clusters the pooled
zones and writes ``zone_mark_heatmap.csv`` (archetype x mark composition) and
``zone_archetype_sizes.csv`` (zones and nuclei per archetype). This draws them
at the type specification, replacing the placed raster
``if_zone_mark_heatmap.png``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sheafstyle as ss

#: Marks are ordered heterochromatin -> active -> structural upstream; DAPI is
#: total DNA rather than a chromatin mark and is drawn last, after a gap.
DNA_STAIN = "DAPI"


def zone_mark_heatmap(prof_csv, sizes_csv, path, width_mm=168.0):
    """Draw the archetype x mark heatmap and return its height in mm."""
    prof = pd.read_csv(prof_csv, index_col=0)
    sizes = pd.read_csv(sizes_csv)
    marks = [m for m in prof.columns if m != DNA_STAIN]
    if DNA_STAIN in prof.columns:
        marks = marks + [DNA_STAIN]
    prof = prof[marks]

    n_row, n_col = prof.shape
    row_mm, cbar_mm = 4.6, 2.6
    left_mm, right_mm = 46.0, 34.0
    top_mm, bottom_mm = 1.5, 17.0          # bottom carries the rotated marks
    height = top_mm + n_row * row_mm + bottom_mm
    fig, ax = ss.panel(width_mm, height)
    fig.subplots_adjust(left=left_mm / width_mm,
                        right=1 - right_mm / width_mm,
                        top=1 - top_mm / height,
                        bottom=bottom_mm / height)

    v = np.abs(prof.values).max()
    im = ax.imshow(prof.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ss.rasterize(im, dpi=600)

    T = ss.TYPE_PT["normal"]["tick"]
    ax.set_xticks(range(n_col))
    ax.set_xticklabels(prof.columns, rotation=90, fontsize=T)
    ax.set_yticks(range(n_row))
    ax.set_yticklabels(prof.index, fontsize=T)
    ax.set_ylim(n_row - 0.5, -0.5)
    ax.tick_params(length=1.5, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    # separate the DNA stain from the chromatin marks
    if DNA_STAIN in prof.columns:
        ax.axvline(n_col - 1.5, color="white", lw=1.2)

    order = {r.archetype: (r.n_zones, r.n_nuclei) for _, r in sizes.iterrows()}
    for i, name in enumerate(prof.index):
        nz, nn = order[name]
        ax.text(n_col - 0.3, i, f"{nz} zones  {nn}/46 nuclei",
                ha="left", va="center", fontsize=T, color="0.35")

    cax = fig.add_axes([1 - (right_mm - 27.0) / width_mm, bottom_mm / height,
                        cbar_mm / width_mm, (n_row * row_mm) / height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("enrichment (z)", fontsize=T, labelpad=2)
    cb.ax.tick_params(labelsize=T, length=1.5, pad=1.5)
    cb.outline.set_visible(False)

    ss.save_panel(fig, path, strict=False)
    return height
