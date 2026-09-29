"""Can k be chosen without an annotation? The criteria against k, on known truth.

Draws the sweeps written by ``experiments/synthetic/k_selection.py``: two
synthetic designs, two data seeds each, k from 2 to 12.

Layout is two rows because the quantities are on two scales, and a single panel
with two y-axes would invite exactly the comparison that is not meaningful.
Row 1 holds the three agreement measures, all of which are adjusted Rand
indices on [0, 1] and so directly comparable: grid-phase stability, the
k-means-seed stability that serves as its control, and -- available only
because the tissue is synthetic -- the ARI against the generating domains.
Row 2 holds the silhouette, which lives on its own scale.

What the panel is for is the contrast between the two columns. On the flat
design k is a fact and both criteria find it. On the nested design two pairs of
domains differ only subtly, the ARI against the truth is flat over several k,
and the criteria settle on the number of well-separated programs rather than
the generating count -- which is the honest behaviour, and the reason the
output of the procedure is a bracket rather than a number.

    python ed_kselection.py
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
REPORTED = HERE.parent / "reported"
W = 180.0

#: (key, column heading, second line, label for the orange reference curve)
COLUMNS = [
    ("flat", "synthetic, well separated", "$k$ is a fact"),
    ("nested", "synthetic, two sibling pairs", "$k$ is a convention"),
    ("merfish", "MERFISH, five real sections", "$k$ annotated as 8"),
]
#: true of every column; what the reference partition IS differs, and the
#: header says so, but one legend is better than three
ARI_LABEL = "ARI vs reference partition"

#: Okabe-Ito, the palette the rest of the figures use; blue is the criterion
#: being proposed, orange the label-dependent reference, grey the control.
C_STAB = ss.METHOD_COLORS.get("spSheaf", "#0072B2")
C_TRUE = "#E69F00"
C_CTRL = "#9AA5B1"
C_SIL = "#009E73"
INK = "#333333"


def _find(*candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


def load(key, k_true=6):
    """One column's sweeps, with ``ari`` as the common reference column.

    The synthetic designs have several data seeds; the real section has one
    (a second would be a second animal, not a second draw). The dashed line is
    the generating count for the synthetic designs and the annotated count for
    the real one -- a fact in the first case and a convention in the second,
    which is why they are shown side by side.
    """
    if key == "merfish":
        out = []
        for i, s in enumerate(("0.04", "0.09", "0.14", "0.19", "0.24")):
            name = f"k_selection_MERFISH_{s}.csv"
            p = _find(paths.results("merfish") / name, REPORTED / "merfish" / name)
            if p is not None:
                out.append(pd.read_csv(p).assign(seed=i, k_true=8))
        if not out:
            raise SystemExit("no real-section sweeps; run "
                             "'make merfish-k-selection' first")
        # the band over the five slices is a band over sections of one tissue,
        # which is the quantity that matters: a criterion that cannot bound k
        # reproducibly across them cannot bound it on one unannotated section
        return pd.concat(out, ignore_index=True).rename(
            columns={"ari_vs_annotation": "ari"})
    out = []
    for seed in range(8):
        name = f"k_selection_{key}_k{k_true}_s{seed}.csv"
        p = _find(paths.results("synthetic") / name, REPORTED / "synthetic" / name)
        if p is not None:
            out.append(pd.read_csv(p).assign(seed=seed))
    if not out:
        raise SystemExit(f"no sweeps for the {key} design; run "
                         f"'make k-selection' first")
    return pd.concat(out, ignore_index=True).rename(
        columns={"ari_vs_truth": "ari"})


def band(ax, d, col, color, label, *, lw=1.0, ms=2.6, ls="-"):
    """Mean over seeds as the line, the seed-to-seed range as the band."""
    g = d.groupby("k")[col]
    k = g.mean().index.to_numpy()
    ax.fill_between(k, g.min(), g.max(), color=color, alpha=0.16, lw=0,
                    zorder=2)
    ax.plot(k, g.mean(), ls, color=color, lw=lw, ms=ms, marker="o",
            label=label, zorder=3, clip_on=False)
    return g.mean()


def bracket(d):
    """The procedure: silhouette's peak, and where stability leaves its plateau."""
    m = d.groupby("k")[["stability", "silhouette"]].mean()
    plateau = m.loc[m.index <= 3, "stability"].mean()
    ok = m.index[m.stability >= 0.95 * plateau]
    upper = int(ok.max()) if len(ok) else int(m.index.min())
    return int(m.silhouette.idxmax()), upper


def main():
    ss.apply()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T = ss.TYPE_PT["normal"]["text"]
    TICK = ss.TYPE_PT["normal"]["tick"]

    height = 99.0
    fig, axes = ss.panel_grid(W, height, 2, len(COLUMNS), sharex="col")
    fig.subplots_adjust(left=13.0 / W, right=0.995, top=1 - 13.5 / height,
                        bottom=10.0 / height, wspace=0.17, hspace=0.16)
    axes = np.atleast_2d(axes)

    for c, (key, what, verdict) in enumerate(COLUMNS):
        d = load(key)
        k_true = int(d.k_true.iloc[0])
        lo, hi = bracket(d)
        top, bot = axes[0, c], axes[1, c]

        for ax in (top, bot):
            # the recommendation, drawn behind everything; a bracket of one
            # value would be invisible as a span, so give it a k's width
            a, b = (lo - 0.35, hi + 0.35) if lo == hi else (lo, hi)
            ax.axvspan(a, b, color=C_STAB, alpha=0.10, lw=0, zorder=1)
            ax.axvline(k_true, color=INK, lw=0.8, ls=(0, (3, 2)), zorder=4)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=TICK, length=1.6, pad=1.5)
            ks = sorted(d.k.unique())
            ax.set_xticks([x for x in ks if x % 2 == 0])
            ax.set_xlim(min(ks) - 0.5, max(ks) + 0.5)
            ax.grid(axis="y", color="#E6E6E6", lw=0.4, zorder=0)
            ax.set_axisbelow(True)

        band(top, d, "seed_stability", C_CTRL, "$k$-means seed (control)",
             lw=0.8, ms=2.0)
        band(top, d, "ari", C_TRUE, ARI_LABEL)
        band(top, d, "stability", C_STAB, "grid-phase stability")
        top.set_ylim(0.25, 1.02)
        top.set_title(f"{what}\n{verdict}", fontsize=T, color=INK, pad=3)

        band(bot, d, "silhouette", C_SIL, "silhouette on the fibers")
        v = d.silhouette.dropna()
        pad = 0.12 * max(v.max() - v.min(), 1e-3)
        bot.set_ylim(v.min() - pad, v.max() + 2.4 * pad)
        bot.set_xlabel("number of domains $k$", fontsize=T)

        # name the bracket once per column, inside the shaded span
        bot.annotate(f"$k$ = {lo}–{hi}" if lo != hi else f"$k$ = {lo}",
                     xy=((lo + hi) / 2, 0.04), xycoords=("data", "axes fraction"),
                     ha="center", va="bottom", fontsize=TICK, color=C_STAB,
                     bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

        if c == 0:
            top.set_ylabel("adjusted Rand index", fontsize=T, labelpad=2)
            bot.set_ylabel("silhouette", fontsize=T, labelpad=2)
            top.legend(loc="lower right", fontsize=TICK, frameon=False,
                       handlelength=1.4, borderpad=0.1, labelspacing=0.22)

    fig.text(13.0 / W, 1 - 1.2 / height,
             "Annotation-free criteria against $k$. The reference partition is the "
             "generating domains (synthetic) or the annotation (MERFISH); the band is "
             "two data seeds\nfor the synthetic designs and the five sections for "
             "MERFISH. Dashed, that reference's $k$; shaded, the interval the two "
             "criteria return on the pooled curves.",
             fontsize=TICK, va="top", ha="left", color=INK, linespacing=1.5)

    out = OUTDIR / "ed_kselection.pdf"
    ss.save_panel(fig, out, strict=False)
    print(f"wrote {out}")
    for key, _, _ in COLUMNS:
        d = load(key)
        m = d.groupby("k")[["stability", "silhouette", "ari",
                            "seed_stability"]].mean()
        lo, hi = bracket(d)
        plateau = m.index[m.ari >= 0.97 * m.ari.max()].tolist()
        print(f"  {key:7s} seeds={d.seed.nunique()}  bracket {lo}-{hi}  "
              f"argmax stability {int(m.stability.idxmax())}  "
              f"argmax silhouette {int(m.silhouette.idxmax())}  "
              f"ARI plateau (within 3%) {plateau}")
    problems, _ = verify(out, width_mm=W)
    for p in problems:
        print("  !", p)


if __name__ == "__main__":
    main()
