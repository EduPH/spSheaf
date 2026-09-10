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


def _grid_image(u, v, zone, ustep, vstep, ulim, vlim, ncol):
    """Voxel labels on a regular (u, v) grid as an RGBA image; empty = transparent."""
    from matplotlib.colors import to_rgba
    ui = np.round((u - ulim[0]) / ustep).astype(int)
    vi = np.round((v - vlim[0]) / vstep).astype(int)
    nu = int(round((ulim[1] - ulim[0]) / ustep)) + 1
    nv = int(round((vlim[1] - vlim[0]) / vstep)) + 1
    img = np.zeros((nv, nu, 4))
    rgba = np.array([to_rgba(c) for c in ZONE_COLORS])
    img[vi, ui] = rgba[zone % ncol]
    return img


def nucleus_row(csv, path, width_mm=168.0, n_sections=6, label_prefix="zone"):
    """Draw the row and return its height in mm."""
    d = pd.read_csv(csv)
    zs = np.sort(d.z.unique())
    pick = zs[np.linspace(0, len(zs) - 1, n_sections).round().astype(int)]
    n = n_sections + 2
    legend_w = 0.10
    cell = (width_mm * (1 - legend_w) - 2.0) / n
    height = cell + 6.0                       # square cells plus a title line
    fig, axes = ss.panel_grid(width_mm, height, 1, n)
    fig.subplots_adjust(left=0.006, right=1 - legend_w, top=0.80, bottom=0.03, wspace=0.06)
    xs, ys = np.sort(d.x.unique()), np.sort(d.y.unique())
    xstep, ystep = np.diff(xs).min(), np.diff(ys).min()
    zstep = np.diff(zs).min() if len(zs) > 1 else xstep
    xr, yr, zr = (xs.min(), xs.max()), (ys.min(), ys.max()), (zs.min(), zs.max())
    T = ss.TYPE_PT["normal"]["tick"]
    ncol = len(ZONE_COLORS)

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
        draw(ax, _grid_image(s_.x.values, s_.y.values, s_.zone.values, xstep, ystep, xr, yr, ncol),
             f"z = {z/1000:.1f} $\\mu$m", xr, yr, xstep, ystep)
    ym = ys[len(ys) // 2]
    s_ = d[d.y == ym]
    draw(axes[n_sections], _grid_image(s_.x.values, s_.z.values, s_.zone.values, xstep, zstep, xr, zr, ncol),
         "xz slab", xr, zr, xstep, zstep)
    xm = xs[len(xs) // 2]
    s_ = d[d.x == xm]
    draw(axes[n_sections + 1], _grid_image(s_.y.values, s_.z.values, s_.zone.values, ystep, zstep, yr, zr, ncol),
         "yz slab", yr, zr, ystep, zstep)
    lax = fig.add_axes([1 - legend_w + 0.005, 0.05, legend_w - 0.01, 0.75])
    lax.set_axis_off()
    for zk in sorted(d.zone.unique()):
        lax.scatter([], [], s=9, c=ZONE_COLORS[zk % ncol], marker="s", linewidths=0,
                    label=f"{label_prefix} {int(zk)}")
    lax.legend(loc="center left", frameon=False, fontsize=T, handletextpad=0.3,
               labelspacing=0.3, borderaxespad=0)
    ss.save_panel(fig, path, strict=False)
    return height
