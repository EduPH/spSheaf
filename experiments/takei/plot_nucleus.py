#!/usr/bin/env python
"""
Readable views of a nucleus's inferred zones.

A single 3D scatter of a filled nucleus occludes its own interior, so this
renders the zones the way microscopy is read — as slices — plus a rotatable
interactive view:

  <name>_slices.png  : top row = x-y optical sections at increasing z;
                       bottom row = three orthogonal central slabs (xy / xz / yz)
  <name>.html        : rotatable 3D (plotly)

Input is a zones CSV (columns x, y, z, zone) from run_takei_nucleus.py.

Usage
-----
    python plot_nucleus.py --labels output_takei/nucleus_4_12_zones.csv
    python plot_nucleus.py --labels "output_takei/*_zones.csv"   # batch
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True,
                   help="A zones CSV or a glob (quote it).")
    p.add_argument("--out-dir", default=None,
                   help="Default: alongside each input file.")
    p.add_argument("--zone-col", default="zone")
    p.add_argument("--n-zslices", type=int, default=6)
    p.add_argument("--no-interactive", action="store_true")
    return p


def _cmap(zones):
    uniq = sorted(set(zones))
    cm = plt.cm.get_cmap("tab10", max(len(uniq), 2))
    return uniq, {z: cm(i) for i, z in enumerate(uniq)}


def slices_png(df, zcol, out_path, n_z=6):
    x, y, z = df.x.values, df.y.values, df.z.values
    zone = df[zcol].values
    uniq, lut = _cmap(zone)
    cols = np.array([lut[v] for v in zone])

    zlv = np.unique(z)
    pick = zlv[np.linspace(0, len(zlv) - 1, min(n_z, len(zlv))).astype(int)]
    ncol = max(len(pick), 3)
    fig, axes = plt.subplots(2, ncol, figsize=(3.2 * ncol, 6.6))

    # Row 1: optical sections (x-y at each chosen z)
    for i in range(ncol):
        ax = axes[0, i]
        if i < len(pick):
            m = z == pick[i]
            ax.scatter(x[m], y[m], c=cols[m], s=10)
            ax.set_title(f"z = {pick[i]:.0f} nm", fontsize=9)
            ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        if i >= len(pick):
            ax.set_visible(False)

    # Row 2: three orthogonal central slabs
    xm, ym, zm = np.median(x), np.median(y), np.median(z)
    sx = (np.percentile(x, 75) - np.percentile(x, 25)) * 0.35
    sy = (np.percentile(y, 75) - np.percentile(y, 25)) * 0.35
    sz = (np.percentile(z, 75) - np.percentile(z, 25)) * 0.5
    panels = [
        ("x-y @ mid-z", np.abs(z - zm) <= sz, x, y, "x (nm)", "y (nm)", True),
        ("x-z @ mid-y", np.abs(y - ym) <= sy, x, z, "x (nm)", "z (nm)", False),
        ("y-z @ mid-x", np.abs(x - xm) <= sx, y, z, "y (nm)", "z (nm)", False),
    ]
    for i in range(ncol):
        ax = axes[1, i]
        if i < 3:
            title, m, a, b, xl, yl, inv = panels[i]
            ax.scatter(a[m], b[m], c=cols[m], s=10)
            ax.set_title(title, fontsize=9)
            ax.set_aspect("equal")
            if inv:
                ax.invert_yaxis()
            ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
            ax.tick_params(labelsize=7)
        else:
            ax.set_visible(False)

    handles = [plt.Line2D([0], [0], marker="o", ls="", color=lut[z_],
               label=f"zone {z_}") for z_ in uniq]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(uniq), 6),
               fontsize=8, bbox_to_anchor=(0.5, -0.03))
    name = os.path.basename(out_path)
    fig.suptitle(f"{name}  —  optical sections (top) & orthogonal slabs (bottom)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def interactive_html(df, zcol, out_path):
    import plotly.graph_objects as go
    uniq = sorted(set(df[zcol]))
    pal = plt.cm.get_cmap("tab10", max(len(uniq), 2))
    traces = []
    for i, z_ in enumerate(uniq):
        d = df[df[zcol] == z_]
        rgb = tuple(int(255 * v) for v in pal(i)[:3])
        traces.append(go.Scatter3d(
            x=d.x, y=d.y, z=d.z, mode="markers", name=f"zone {z_}",
            marker=dict(size=2.6, color=f"rgb{rgb}", opacity=0.75)))
    fig = go.Figure(traces)
    fig.update_layout(
        title=f"{os.path.basename(out_path)} — nuclear zones (drag to rotate)",
        scene=dict(xaxis_title="x (nm)", yaxis_title="y (nm)",
                   zaxis_title="z (nm)", aspectmode="data"),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(out_path, include_plotlyjs="cdn")


def main():
    a = build_parser().parse_args()
    files = sorted(glob.glob(a.labels))
    if not files:
        raise SystemExit(f"No files match {a.labels}")
    for f in files:
        df = pd.read_csv(f)
        out_dir = a.out_dir or os.path.dirname(f) or "."
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(f))[0]
        slices_png(df, a.zone_col, os.path.join(out_dir, base + "_slices.png"),
                   n_z=a.n_zslices)
        print("  ->", os.path.join(out_dir, base + "_slices.png"))
        if not a.no_interactive:
            try:
                interactive_html(df, a.zone_col,
                                 os.path.join(out_dir, base + ".html"))
                print("  ->", os.path.join(out_dir, base + ".html"))
            except Exception as e:
                print("  (interactive skipped:", e, ")")


if __name__ == "__main__":
    main()
