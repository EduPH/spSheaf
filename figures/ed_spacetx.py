"""Extended Data — the two independent SpaceTx cortical sections, per cell.

One row per section. Each row shows the layer annotation, the smoothed sheaf
disagreement field on the spSheaf grid, the spSheaf partition at the
parameters carried over from MERFISH, and every competitor at its best
in-sample setting, all Hungarian-matched to the annotation. Each row then adds
spSheaf at its own best in-sample configuration -- the argmax of the
162-configuration grid, which is the budget the competitors were given -- so the
two spSheaf columns are the transferred and in-sample rows of Table 2 side by
side. Every ARI printed above a map is read from the results tables that Table 2
is built from; nothing is typed in here.

The in-sample column appears only once its panel dump exists. ``main`` prints the
exact command for any that is missing, and the configuration in it is read from
the sweep table, never hard-coded.

Inputs (all under the results root of ``spsheaf.paths``):
    spacetx/<ds>/panels/spsheaf.npz|json             experiments.spacetx.run_spacetx
    spacetx/<ds>/panels/spsheaf_insample.npz|json    (same, at the sweep argmax)
    spacetx/<ds>/sweep.csv                           experiments.spacetx.sweep
    segfree/spacetx_<ds>/labels_{p2r_best,ssam_best_sweep,ficture_best}.csv
    segfree/spacetx_<ds>/metrics_{p2r,ssam}.json, segfree/ficture/metrics_spacetx_<ds>.csv

    python ed_spacetx.py
    python verify_figure.py out/ed_spacetx.pdf --width-mm 180
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sheafstyle as ss
from layout import Layout, Panel
from spsheaf import paths
from spsheaf.evaluation import hungarian_match_labels

HERE = Path(__file__).parent
PANELS = HERE / "panels"
PANELS.mkdir(exist_ok=True)
OUT = HERE / "out" / "ed_spacetx.pdf"

DOMAIN_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                 "#D55E00", "#56B4E9", "#F0E442", "#8C8C8C"]
EXTRA = "#C8C8C8"          # clusters beyond the annotated ones (unmatched)
TEXT = ss.TYPE_PT["normal"]["text"]
TICK = ss.TYPE_PT["normal"]["tick"]

DATASETS = [("MERFISH_Allen_Prototype", "Allen prototype MERFISH"),
            ("BARISTASeq", "BaristaSeq")]


def _results(*parts):
    return paths.results().joinpath(*parts)


def _load_spsheaf(ds, name="spsheaf"):
    d = np.load(_results("spacetx", ds, "panels", f"{name}.npz"))
    meta = json.loads(_results("spacetx", ds, "panels", f"{name}.json").read_text())
    return d, meta


INSAMPLE = "spsheaf_insample"
#: sweep column -> run_spacetx flag, for the reproduction command printed below
FLAGS = {"bandwidth": "--bandwidth", "smooth_radius": "--smooth-radius",
         "uniform_dim": "--uniform-dim", "spatial_weight": "--spatial-weight",
         "boundary_alpha": "--boundary-alpha", "reference": "--reference"}


def insample_config(ds):
    """The argmax row of the in-sample grid, as a plain dict, or None."""
    p = _results("spacetx", ds, "sweep.csv")
    if not p.exists():
        return None
    d = pd.read_csv(p).dropna(subset=["ari"])
    return None if d.empty else d.loc[d.ari.idxmax()].to_dict()


def insample_command(ds, cfg):
    """The command that produces the missing panel dump, at exactly that config."""
    flags = " ".join(f"{FLAGS[k]} {cfg[k]:g}" if isinstance(cfg[k], float) else f"{FLAGS[k]} {cfg[k]}"
                     for k in FLAGS if k in cfg and pd.notna(cfg[k]))
    return (f"python -m experiments.spacetx.run_spacetx {ds} $SPACETX_COLS "
            f"--grid-spacing 10 {flags} --panel-name {INSAMPLE}")


def _matched(pred, gti, valid):
    """Contiguous relabel, then Hungarian match onto the annotation indices."""
    _, contig = np.unique(pred, return_inverse=True)
    return hungarian_match_labels(contig.astype(int), gti, valid)


def _competitors(ds, gti, valid):
    seg = _results("segfree", f"spacetx_{ds}")
    p2r = json.loads((seg / "metrics_p2r.json").read_text())["best"]
    # SSAM's grid spans the built-in normalisation at the published resolution
    # (metrics_ssam.json) and sctransform at resolutions 1, 4 and 8
    # (metrics_ssam_r*.json); the panel shows whichever is best on this section.
    ssam_runs = []
    for path in [seg / "metrics_ssam.json", *sorted(seg.glob("metrics_ssam_r*.json"))]:
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        tag = path.stem.replace("metrics_ssam", "")
        ssam_runs.append((max(d["sweep"], key=lambda r: r["ARI"])["ARI"], tag))
    ssam_ari, ssam_tag = max(ssam_runs)
    fic = pd.read_csv(_results("segfree", "ficture", f"metrics_spacetx_{ds}.csv"))
    fic_best = fic.loc[fic.ARI.idxmax()]
    out = []
    for name, csv, ari in [("Points2Regions", "labels_p2r_best.csv", p2r["ARI"]),
                           ("SSAM", f"labels_ssam_best_sweep{ssam_tag}.csv", ssam_ari),
                           ("FICTURE", "labels_ficture_best.csv", fic_best.ARI)]:
        lab = pd.read_csv(seg / csv)["pred"].values
        out.append((name, _matched(lab, gti, valid), float(ari)))
    return out


def _draw_cells(ax, xy, lab, s):
    cmap = np.array(DOMAIN_COLORS)
    col = np.where(lab < len(cmap), cmap[np.clip(lab, 0, len(cmap) - 1)], EXTRA)
    col[lab < 0] = EXTRA
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=col, linewidths=0, marker="o")
    ss.rasterize(sc, dpi=600)


def row_panel(ds, path, *, height_mm, cell_s=1.2):
    d, meta = _load_spsheaf(ds)
    names = meta["gt_names"]
    xy, gti = d["cell_coords"], d["cell_gt_int"]
    valid = gti >= 0
    grid, field = d["grid_coords"], d["boundary_smooth"]
    cols = [("annotation", "cells", gti, None),
            ("sheaf field $\\tilde B$ (rank)", "field", field, None),
            ("spSheaf", "cells", d["cell_pred_matched"], meta["metrics"]["ari"])]
    cfg = insample_config(ds)
    if cfg is not None:
        if _results("spacetx", ds, "panels", f"{INSAMPLE}.npz").exists():
            de, me = _load_spsheaf(ds, INSAMPLE)
            cols.append(("spSheaf, in-sample", "cells",
                         de["cell_pred_matched"], me["metrics"]["ari"]))
            if abs(me["metrics"]["ari"] - cfg["ari"]) > 0.01:
                print(f"  ! {ds}: panel ARI {me['metrics']['ari']:.3f} but the grid argmax "
                      f"is {cfg['ari']:.3f} -- the dump is stale, re-run:\n    "
                      + insample_command(ds, cfg))
        else:
            print(f"  ! {ds}: no in-sample column (grid argmax ARI {cfg['ari']:.3f}). "
                  f"To draw it:\n    " + insample_command(ds, cfg))
    for name, lab, ari in _competitors(ds, gti, valid):
        cols.append((name, "cells", lab, ari))
    w = 180 - 16
    fig, axes = ss.panel_grid(w, height_mm, 1, len(cols))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.80, bottom=0.02, wspace=0.05)
    step = float(np.min(np.diff(np.unique(grid[:, 0]))))          # grid spacing
    ij = np.round((grid - grid.min(0)) / step).astype(int)
    # show the field only under tissue: grid points within reach of an
    # annotated cell (4 spacings). Outside, the empty-fiber border dominates the colour scale
    # and hides the interior structure the panel is meant to show.
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(xy[valid]).query(grid, k=1)
    under = dist <= 4 * step
    img = np.full(ij.max(0)[::-1] + 1, np.nan)
    # percentile rank of the field among under-tissue grid points: the tissue
    # edge is orders of magnitude above the laminar transitions and would
    # otherwise saturate the scale (the title says "rank").
    from scipy.stats import rankdata
    img[ij[under, 1], ij[under, 0]] = rankdata(field[under]) / under.sum()
    lo, hi = 0.0, 1.0
    extent = (grid[:, 0].min() - step / 2, grid[:, 0].max() + step / 2,
              grid[:, 1].min() - step / 2, grid[:, 1].max() + step / 2)
    for ax, (name, kind, arr, ari) in zip(axes, cols):
        ss.spatial(ax)
        if kind == "field":
            im = ax.imshow(img, origin="lower", extent=extent, cmap="magma",
                           vmin=lo, vmax=hi, interpolation="nearest")
            ss.rasterize(im, dpi=600)
        else:
            _draw_cells(ax, xy, arr, cell_s)
        title = name if ari is None else f"{name}\nARI {ari:.2f}"
        key = "spSheaf" if name.startswith("spSheaf") else name
        ax.set_title(title, fontsize=TEXT, color=ss.METHOD_COLORS.get(key, "#333333"), pad=2)
        ax.set_xlim(grid[:, 0].min(), grid[:, 0].max())
        ax.set_ylim(grid[:, 1].min(), grid[:, 1].max())
    ss.save_panel(fig, path)
    return names


def legend_strip(path, names):
    fig, ax = ss.panel(180 - 16, 4.5)
    ax.set_axis_off()
    cmap = np.array(DOMAIN_COLORS)
    for i, nm in enumerate(names):
        ax.scatter([], [], s=9, c=cmap[i], label=nm, linewidths=0)
    ax.scatter([], [], s=9, c=EXTRA, label="unmatched cluster", linewidths=0)
    ax.legend(loc="center", ncol=len(names) + 1, frameon=False, fontsize=TICK,
              handletextpad=0.3, columnspacing=1.4, borderaxespad=0)
    ss.save_panel(fig, path, strict=False)


H_A, H_B = 30.0, 42.0
FIG = Layout(
    name="ed_spacetx", width_mm=180.0, height_mm=4 + H_A + 5 + 6 + H_B + 5 + 2,
    panels=[
        Panel("a", PANELS / "edsx_a.pdf", x=8, y=4, label_dx=-4, label_dy=-2),
        Panel(None, PANELS / "edsx_la.pdf", x=8, y=4 + H_A),
        Panel("b", PANELS / "edsx_b.pdf", x=8, y=4 + H_A + 5 + 6, label_dx=-4, label_dy=-2),
        Panel(None, PANELS / "edsx_lb.pdf", x=8, y=4 + H_A + 5 + 6 + H_B),
    ],
)


def main():
    ss.apply()
    names_a = row_panel("MERFISH_Allen_Prototype", PANELS / "edsx_a.pdf",
                        height_mm=H_A, cell_s=1.6)
    legend_strip(PANELS / "edsx_la.pdf", names_a)
    names_b = row_panel("BARISTASeq", PANELS / "edsx_b.pdf", height_mm=H_B, cell_s=1.0)
    legend_strip(PANELS / "edsx_lb.pdf", names_b)
    print(FIG.summary())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("\nwrote", FIG.compose(OUT))


if __name__ == "__main__":
    main()
