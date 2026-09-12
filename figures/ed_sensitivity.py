"""Extended Data: which hyperparameters the accuracy actually depends on.

Reads the three in-sample grids -- the 108+24 configuration osmFISH sweep and
the 162-configuration SpaceTx sweeps, all scored against the section's own
annotation -- and shows ARI against each swept parameter in turn, marginalising
over the others. Each table is taken from the results root when it is there and
from the versioned copy under ``reported/`` otherwise, so the panel builds from
a bare clone.

The point of the panel is a contrast. The kernel bandwidth and the
boundary-smoothing radius move ARI by a wide margin and their optima differ
between datasets, which is why those two settings do not transfer. The stalk
dimension does not: over 4 <= d <= 20, up to 61% of the 33-gene osmFISH
feature space, the mean moves by at most 0.02 and the maximum by 0.001. That
is the answer to whether the subspace is short of capacity, and it is a null
result worth showing rather than asserting.

    python ed_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sheafstyle as ss
from spsheaf import paths
from verify_figure import verify

HERE = Path(__file__).parent
OUTDIR = HERE / "out"
#: versioned copies of the same tables, so the panel builds from a bare clone
REPORTED = HERE.parent / "reported"
PX_TO_UM = 1.0 / 15.3846        # osmFISH imaging scale; SpaceTx grids are in um
W = 180.0

#: column key -> (axis label, whether the osmFISH values need px -> um)
PARAMS = [
    ("reference", "coordinate reference", False),
    ("bandwidth", "kernel $h$ ($\\mu$m)", True),
    ("smooth_radius", "smoothing $\\sigma_s$ ($\\mu$m)", True),
    ("spatial_weight", "spatial weight $\\alpha_s$", False),
    ("uniform_dim", "stalk dimension $d$", False),
]


#: every column that indexes a configuration, so the marginals can be balanced
INDEX = ["reference", "grid_spacing", "bandwidth", "smooth_radius",
         "uniform_dim", "spatial_weight", "boundary_alpha"]


def balanced(d, key):
    """Rows of `d` forming a complete block over `key`.

    A marginal is only interpretable if every level of `key` was run on the
    same set of other settings. On osmFISH it was not -- d in {12,16,20} were
    added on a subgrid of the bandwidths and radii that perform, while d in
    {6,9} span the whole 108-configuration grid -- so an unbalanced marginal
    reports the other parameters' effect as if it were d's, threefold here.
    """
    if key not in d:
        return d.iloc[0:0]
    others = [c for c in INDEX if c in d and c != key]
    if not others:
        return d
    levels = set(d[key].unique())
    grp = d.groupby(others, dropna=False)
    keep = [idx for idx, g in grp if set(g[key].unique()) == levels]
    if not keep:
        return d.iloc[0:0]
    mask = d.set_index(others).index.isin(keep)
    return d[mask]


def _table(live: Path, archived: Path) -> Path | None:
    """The live results table if it is there, else the copy under reported/."""
    if live.exists():
        return live
    if archived.exists():
        print(f"  (using reported/{archived.relative_to(REPORTED)}; "
              f"{live.name} not in the results root)")
        return archived
    return None


def load():
    """The three grids, in micrometres, with a common set of columns."""
    out = []
    p = _table(paths.results("osmfish") / "output_osmFISH_boundary" / "osmfish_sweep.csv",
               REPORTED / "osmfish" / "osmfish_sweep.csv")
    if p is not None:
        d = pd.read_csv(p).dropna(subset=["ari"])
        for c in ("bandwidth", "smooth_radius"):
            d[c] = (d[c] * PX_TO_UM).round(1)   # keep tick labels short
        out.append(("osmFISH", "33 genes", d))
    for ds, label in [("MERFISH_Allen_Prototype", "Allen prototype"),
                      ("BARISTASeq", "BaristaSeq")]:
        p = _table(paths.results("spacetx") / ds / "sweep.csv",
                   REPORTED / "spacetx" / f"{ds}_sweep.csv")
        if p is not None:
            d = pd.read_csv(p).dropna(subset=["ari"])
            genes = {"MERFISH_Allen_Prototype": "268 genes", "BARISTASeq": "81 genes"}[ds]
            out.append((label, genes, d))
    if not out:
        raise SystemExit("no sweep tables found, in the results root or in "
                         "reported/; run 'make osmfish-insample' and "
                         "'make spacetx-insample-grid' first")
    return out


def main():
    ss.apply()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load()
    rng = np.random.default_rng(0)

    T = ss.TYPE_PT["normal"]["text"]
    TICK = ss.TYPE_PT["normal"]["tick"]
    cell_w = (W - 26.0) / len(PARAMS)
    row_h = 27.0
    height = len(data) * row_h + 6.0
    fig, axes = ss.panel_grid(W, height, len(data), len(PARAMS), sharey="row")
    fig.subplots_adjust(left=22.0 / W, right=0.995, top=1 - 3.0 / height,
                        bottom=9.0 / height, wspace=0.10, hspace=0.42)
    axes = np.atleast_2d(axes)

    blue = ss.METHOD_COLORS.get("spSheaf", "#0072B2")
    for r, (name, genes, d) in enumerate(data):
        for c, (key, label, _) in enumerate(PARAMS):
            ax = axes[r, c]
            b = balanced(d, key)
            if key not in d or not len(b) or b[key].nunique() < 2:
                # keep the cell so the row keeps its label and the grid stays
                # square, and say why it is empty
                only = (f"{d[key].iloc[0]} only" if key in d and len(d)
                        else "not swept")
                ax.text(0.5, 0.5, only, transform=ax.transAxes, ha="center",
                        va="center", fontsize=TICK, color="#9AA5B1")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if c == 0:
                    ax.set_ylabel(f"{name}\n({genes})", fontsize=T, labelpad=3)
                if r == len(data) - 1:
                    ax.set_xlabel(label, fontsize=T)
                continue
            cat = b[key].dtype == object
            if cat:                      # categorical: order local, tissue
                order = [v for v in ("local", "tissue") if v in set(b[key])]
                b = b.assign(**{key: b[key].map({v: i for i, v in enumerate(order)})})
                vals = np.arange(len(order))
            else:
                vals = np.sort(b[key].unique())
            # every configuration of the balanced block as a faint dot,
            # jittered so ties are visible
            x = b[key].to_numpy(float)
            span = (vals.max() - vals.min()) or 1.0
            ax.scatter(x + rng.uniform(-0.012, 0.012, len(x)) * span, b.ari,
                       s=3.0, facecolors="none", edgecolors="#9AA5B1",
                       linewidths=0.35, zorder=2)
            m = b.groupby(key).ari.agg(["mean", "max"])
            ax.plot(m.index, m["mean"], "-o", color=blue, lw=1.0, ms=2.6, zorder=3)
            ax.plot(m.index, m["max"], ":", color=blue, lw=0.8, zorder=3)
            ax.set_xticks(vals)
            ax.set_xticklabels(order if cat else [f"{v:g}" for v in vals],
                               fontsize=TICK)
            if cat:
                ax.set_xlim(-0.4, len(vals) - 0.6)
            ax.tick_params(labelsize=TICK, length=1.6, pad=1.5)
            # the spread each parameter induces, as one number per cell
            drop = m["mean"].max() - m["mean"].min()
            ax.set_title(f"spread {drop:.3f}  ($n$ {len(b)})", fontsize=TICK,
                         color="#333333" if drop >= 0.05 else "#9AA5B1", pad=1.5)
            if r == len(data) - 1:
                ax.set_xlabel(label, fontsize=T)
            if c == 0:
                ax.set_ylabel(f"{name}\n({genes})", fontsize=T, labelpad=3)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)

    out = OUTDIR / "ed_sensitivity.pdf"
    ss.save_panel(fig, out, strict=False)
    print(f"wrote {out}")
    for name, _, d in data:
        s = {}
        for k, _, _ in PARAMS:
            if k not in d:
                continue
            b = balanced(d, k)
            if len(b):
                mm = b.groupby(k).ari.mean()
                s[k] = mm.max() - mm.min()
        print(f"  {name:16s} n={len(d):4d}  spread in mean ARI: "
              + ", ".join(f"{k} {v:.3f}" for k, v in s.items()))
    problems, _ = verify(out, width_mm=W)
    for m_ in problems:
        print("  !", m_)


if __name__ == "__main__":
    main()
