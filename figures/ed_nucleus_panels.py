"""A row of small multiples for one 3D nucleus, drawn from the zone table.

The Takei runs write ``nucleus_<id>_zones.csv`` with one row per grid voxel
(x, y, z in nm, zone). This draws, at the type specification, six optical
sections at evenly spaced z and two orthogonal slabs (xz at the median y, yz at
the median x), as one panel PDF of the requested width. Replaces the placed
rasters ``ed_dna_nucleus.png`` / ``if_nucleus.png`` in the Extended Data
composites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sheafstyle as ss

ZONE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
               "#56B4E9", "#F0E442", "#8C8C8C"]


def _grid_image(u, v, zone, ustep, vstep, ulim, vlim, ncol, palette=None):
    """Voxel labels on a regular (u, v) grid as an RGBA image; empty = transparent."""
    from matplotlib.colors import to_rgba
    ui = np.round((u - ulim[0]) / ustep).astype(int)
    vi = np.round((v - vlim[0]) / vstep).astype(int)
    nu = int(round((ulim[1] - ulim[0]) / ustep)) + 1
    nv = int(round((vlim[1] - vlim[0]) / vstep)) + 1
    img = np.zeros((nv, nu, 4))
    rgba = np.array([to_rgba(c) for c in (palette or ZONE_COLORS)])
    img[vi, ui] = rgba[zone % ncol]
    return img


def nucleus_row(csv, path, width_mm=168.0, n_sections=6, label_prefix="zone",
                colors=None, scale_um=None, names=None, slabs=("xz", "yz"),
                legend_w=0.10):
    """Draw the row and return its height in mm."""
    d = pd.read_csv(csv)
    zs = np.sort(d.z.unique())
    if n_sections == 1:
        pick = zs[[len(zs) // 2]]          # a mid-depth section, not the top cap
    else:
        pick = zs[np.linspace(0, len(zs) - 1, n_sections).round().astype(int)]
    n = n_sections + len(slabs)
    cell = min((width_mm * (1 - legend_w) - 2.0) / n, 30.0)
    used = (n * cell + 2.0) / width_mm         # fraction of the canvas the maps need
    height = cell + 6.0                       # square cells plus a title line
    fig, axes = ss.panel_grid(width_mm, height, 1, n)
    fig.subplots_adjust(left=0.006, right=min(used, 1 - legend_w), top=0.80,
                        bottom=0.03, wspace=0.06)
    xs, ys = np.sort(d.x.unique()), np.sort(d.y.unique())
    xstep, ystep = np.diff(xs).min(), np.diff(ys).min()
    zstep = np.diff(zs).min() if len(zs) > 1 else xstep
    xr, yr, zr = (xs.min(), xs.max()), (ys.min(), ys.max()), (zs.min(), zs.max())
    T = ss.TYPE_PT["normal"]["tick"]
    palette = list(colors) if colors is not None else ZONE_COLORS
    ncol = len(palette)

    def draw(ax, img, title, ulim, vlim, ustep, vstep):
        ss.spatial(ax)
        im = ax.imshow(img, origin="lower", interpolation="none",
                       extent=(ulim[0] - ustep / 2, ulim[1] + ustep / 2,
                               vlim[0] - vstep / 2, vlim[1] + vstep / 2))
        ss.rasterize(im, dpi=600)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=T, pad=1.5)

    for ax, z in zip(axes[:n_sections], pick):
        s_ = d[d.z == z]
        draw(ax, _grid_image(s_.x.values, s_.y.values, s_.zone.values, xstep, ystep, xr, yr, ncol, palette),
             f"z = {z/1000:.1f} $\\mu$m", xr, yr, xstep, ystep)
    islab = n_sections
    if "xz" in slabs:
        s_ = d[d.y == ys[len(ys) // 2]]
        draw(axes[islab], _grid_image(s_.x.values, s_.z.values, s_.zone.values, xstep, zstep, xr, zr, ncol, palette),
             "xz slab", xr, zr, xstep, zstep)
        islab += 1
    if "yz" in slabs:
        s_ = d[d.x == xs[len(xs) // 2]]
        draw(axes[islab], _grid_image(s_.y.values, s_.z.values, s_.zone.values, ystep, zstep, yr, zr, ncol, palette),
             "yz slab", yr, zr, ystep, zstep)
    if scale_um is not None:
        ax0 = axes[0]
        x0, x1 = ax0.get_xlim(); y0, y1 = ax0.get_ylim()
        bar = scale_um * 1000.0                      # zone tables are in nm
        xa = x1 - 0.06 * (x1 - x0) - bar
        ya = y0 + 0.07 * (y1 - y0)
        ax0.plot([xa, xa + bar], [ya, ya], color="black", lw=1.2,
                 solid_capstyle="butt", clip_on=False)
        ax0.text(xa + bar / 2, ya + 0.03 * (y1 - y0), f"{scale_um:g} $\\mu$m",
                 ha="center", va="bottom", fontsize=T)

    lax = fig.add_axes([min(used, 1 - legend_w) + 0.01, 0.05, legend_w - 0.01, 0.75])
    lax.set_axis_off()
    for zk in sorted(d.zone.unique()):
        lab = (names.get(int(zk)) if isinstance(names, dict)
               else None) or f"{label_prefix} {int(zk)}"
        lax.scatter([], [], s=9, c=palette[zk % ncol], marker="s", linewidths=0,
                    label=lab)
    lax.legend(loc="center left", frameon=False, fontsize=T, handletextpad=0.3,
               labelspacing=0.3, borderaxespad=0)
    ss.save_panel(fig, path, strict=False)
    return height
