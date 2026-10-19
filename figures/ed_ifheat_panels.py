"""The vector panels of the IF Extended Data figure.

``radial_certification`` draws panel a from the per-zone certification table,
replacing the placed raster ``if_certification.png``; ``zone_mark_heatmap``
draws panel b.

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
    left_mm, right_mm = 34.0, 34.0          # just the row labels on the left; counts and the bar on the right
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

def radial_certification(zone_csv, summary_csv, path, width_mm=120.0):
    """Panel a of the IF figure, drawn from the zone table instead of placed.

    Left: every zone of every nucleus as a point, peripheral-mark score against
    radial position, with one faint least-squares line per nucleus. Right: the
    46 per-nucleus correlations, which is the claim the panel is making -- not
    that the pooled cloud trends upward, but that it trends upward *within*
    every nucleus separately. A pooled scatter alone cannot distinguish those
    two, so both halves are needed.

    Returns the panel height in mm.
    """
    d = pd.read_csv(zone_csv)
    s = pd.read_csv(summary_csv).iloc[0]

    T = ss.TYPE_PT["normal"]["text"]
    TICK = ss.TYPE_PT["normal"]["tick"]
    BLUE = ss.METHOD_COLORS.get("spSheaf", "#0072B2")

    # per-nucleus correlation, recomputed here so the panel and the summary
    # table cannot disagree
    rs = {}
    for cell, g in d.groupby("cell"):
        if len(g) >= 3 and g.radius.std() > 0 and g.periph_score.std() > 0:
            rs[cell] = float(np.corrcoef(g.radius, g.periph_score)[0, 1])
    r = np.array(sorted(rs.values()))

    height = 40.0
    fig, axes = ss.panel_grid(width_mm, height, 1, 2)
    fig.subplots_adjust(left=11.0 / width_mm, right=0.975,
                        top=1 - 2.5 / height, bottom=7.5 / height, wspace=0.42)
    ax, bx = axes

    # --- left: the zones themselves -----------------------------------
    for cell, g in d.groupby("cell"):
        g = g.sort_values("radius")
        ax.plot(g.radius, g.periph_score, "-", color=BLUE, lw=0.3, alpha=0.30,
                zorder=2, solid_capstyle="round")
    sc = ax.scatter(d.radius, d.periph_score, s=2.2, facecolors="none",
                    edgecolors=BLUE, linewidths=0.35, alpha=0.75, zorder=3)
    ss.rasterize(sc, dpi=600)
    b, a = np.polyfit(d.radius, d.periph_score, 1)
    xs = np.linspace(d.radius.min(), d.radius.max(), 50)
    ax.plot(xs, a + b * xs, "-", color="#333333", lw=1.0, zorder=4)
    ax.set_xlabel("radial position of the zone", fontsize=T)
    ax.set_ylabel("peripheral-mark score", fontsize=T, labelpad=2)
    ax.set_title(f"{len(d)} zones, {d.cell.nunique()} nuclei",
                 fontsize=TICK, color="#333333", pad=2)

    # --- right: one correlation per nucleus ----------------------------
    y = np.arange(len(r))
    bx.scatter(r, y, s=3.0, facecolors="none", edgecolors=BLUE, linewidths=0.4,
               zorder=3)
    bx.axvline(0.0, color="#999999", lw=0.6, zorder=2)
    bx.axvline(r.mean(), color="#333333", lw=0.9, zorder=4)
    bx.fill_betweenx([-1, len(r)], r.mean() - r.std(), r.mean() + r.std(),
                     color="#333333", alpha=0.08, lw=0, zorder=1)
    bx.set_ylim(-1, len(r))
    bx.set_xlim(min(-0.08, r.min() - 0.05), 1.0)
    bx.set_xticks([0.0, 0.5, 1.0])
    bx.set_yticks([])
    bx.set_xlabel("within-nucleus correlation $r$", fontsize=T)
    bx.set_title(f"$r = {r.mean():+.2f} \\pm {r.std():.2f}$, "
                 f"positive in {int((r > 0).sum())}/{len(r)}",
                 fontsize=TICK, color="#333333", pad=2)

    for a_ in (ax, bx):
        a_.tick_params(labelsize=TICK, length=1.6, pad=1.5)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)

    # the second claim of the caption, which is a count rather than a fit;
    # set inside the right panel's empty upper-left corner (the points climb
    # to the lower right), where it cannot collide with the axis labels
    if "frac_periphmost_is_hetero" in s.index:
        n = int(round(float(s.frac_periphmost_is_hetero) * len(r)))
        bx.text(0.03, 0.97,
                f"peripheral-most zone is a\nlamina/heterochromatin zone\n"
                f"in {n}/{len(r)} nuclei",
                transform=bx.transAxes, fontsize=TICK, va="top", ha="left",
                color="#333333", linespacing=1.3)

    ss.save_panel(fig, path, strict=False)
    print(f"  radial certification: {len(d)} zones, {len(r)} nuclei, "
          f"r = {r.mean():+.3f} +- {r.std():.3f}, "
          f"{int((r > 0).sum())}/{len(r)} positive "
          f"(summary table: {float(s.mean_r_radius_periph):+.3f} "
          f"+- {float(s.sd_r_radius_periph):.3f}, "
          f"{int(s.n_positive_periph)}/{len(r)})")
    return height
