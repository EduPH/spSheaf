#!/usr/bin/env python
"""
3D visualisation of sheaf-inferred domains from a 3D run.

Reads a ``(x, y, z, label)`` CSV (as written by ``run_brain_receptor_3d.py``)
and produces:
  * domain_map_3d_scatter.png : static multi-angle 3D scatter (matplotlib)
  * domain_map_3d_interactive.html : rotatable 3D scatter (plotly)

The z axis is visually exaggerated (``--z-exaggerate``) because MERSCOPE slabs
are only ~10 um thick against a mm-scale field; tick labels remain true um.

Usage
-----
    python plot_3d.py --labels output_brain_receptor_3d_crop/domain_labels_3d.csv \
        --out-dir output_brain_receptor_3d_crop
"""
import argparse
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
                   help="CSV with columns x, y, z, label.")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--z-exaggerate", type=float, default=120.0,
                   help="Visual stretch of the z axis (default 120x).")
    p.add_argument("--max-points", type=int, default=40000,
                   help="Subsample for a responsive plot (default 40k).")
    p.add_argument("--point-size", type=float, default=6.0)
    return p


def _colors(labels):
    uniq = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 2))
    lut = {c: cmap(i) for i, c in enumerate(uniq)}
    return uniq, lut


def static_png(df, out_dir, zx, ps):
    x, y, z, lab = df.x.values, df.y.values, df.z.values, df.label.values
    uniq, lut = _colors(lab)
    cols = np.array([lut[l] for l in lab])
    views = [(22, -60), (12, 20), (75, -90)]
    titles = ["oblique", "near-side", "top-down"]
    fig = plt.figure(figsize=(6 * len(views), 6))
    for k, ((elev, azim), t) in enumerate(zip(views, titles)):
        ax = fig.add_subplot(1, len(views), k + 1, projection="3d")
        ax.scatter(x, y, z * zx, c=cols, s=ps, depthshade=True,
                   edgecolors="none")
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        ax.set_zlabel(f"z (um, x{zx:g})")
        # true z tick labels
        zt = ax.get_zticks()
        ax.set_zticklabels([f"{v/zx:.1f}" for v in zt])
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 0.6))
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=lut[c],
               label=f"domain {c}") for c in uniq]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(uniq), 8),
               fontsize=8, markerscale=1.4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sheaf-inferred domains in 3D - Brain Receptor Showcase "
                 "(z-stack)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = os.path.join(out_dir, "domain_map_3d_scatter.png")
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return path


def interactive_html(df, out_dir, zx):
    import plotly.graph_objects as go
    uniq = sorted(set(df.label))
    palette = (plt.cm.get_cmap("tab20", max(len(uniq), 2)))
    traces = []
    for i, c in enumerate(uniq):
        d = df[df.label == c]
        rgb = tuple(int(255 * v) for v in palette(i)[:3])
        traces.append(go.Scatter3d(
            x=d.x, y=d.y, z=d.z * zx, mode="markers",
            name=f"domain {c}",
            marker=dict(size=2.2, color=f"rgb{rgb}"),
            hovertemplate="x=%{x:.0f} y=%{y:.0f} "
                          f"z=%{{customdata:.1f}}um<extra>domain {c}</extra>",
            customdata=d.z))
    fig = go.Figure(traces)
    fig.update_layout(
        title="Sheaf-inferred domains in 3D (drag to rotate) - "
              f"Brain Receptor Showcase, z exaggerated {zx:g}x",
        scene=dict(xaxis_title="x (um)", yaxis_title="y (um)",
                   zaxis_title=f"z (um, x{zx:g})",
                   aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.5)),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
    path = os.path.join(out_dir, "domain_map_3d_interactive.html")
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def main():
    a = build_parser().parse_args()
    df = pd.read_csv(a.labels)
    if len(df) > a.max_points:
        df = df.sample(a.max_points, random_state=0).reset_index(drop=True)
    os.makedirs(a.out_dir, exist_ok=True)
    p1 = static_png(df, a.out_dir, a.z_exaggerate, a.point_size)
    print("  ->", p1)
    try:
        p2 = interactive_html(df, a.out_dir, a.z_exaggerate)
        print("  ->", p2)
    except Exception as e:
        print("  (skipped interactive HTML:", e, ")")


if __name__ == "__main__":
    main()
