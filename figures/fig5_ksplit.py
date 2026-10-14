"""What the extra domain is: divisions of annotated domains at k = 9.

Raising k one above the annotated count divides an annotated domain. On the
five MERFISH sections two domains divide the same way on two adjacent slices
each -- BST on 0.19 / 0.24 and MPN on 0.04 / 0.09 -- and this draws both
(``experiments/merfish/k_split.py``, ``k_split_analysis.py``): the annotated
domain, its two halves at k = 9, the two fibers that lead the difference, and
on the right the per-gene difference between the halves on one slice against
the other. The complete record of every division, including those that do not
repeat, is the division table of the manuscript.

    python fig5_ksplit.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgba

import sheafstyle as ss
from spsheaf import paths
from verify_figure import verify
from ed_merfishslices import _image, under_tissue

HERE = Path(__file__).parent
OUTDIR = HERE / "out"
W = 180.0
K = 9
#: (domain, the two slices it divides on, the two fibers drawn, gene labels' side)
CASES = [("BST", ("0.19", "0.24"), ("Isl1", "Esr1")),
         ("MPN", ("0.04", "0.09"), ("Calcr", "Esr1"))]
C_A, C_B = "#0072B2", "#E69F00"
C_DOM, C_REST, C_EMPTY = "#56B4E9", "#E4E4E4", "#F7F7F7"
INK = "#333333"


def halves(z, domain):
    gt, valid = z["gt_int"], z["gt_int"] >= 0
    d = list(z["gt_names"]).index(domain)
    lab = z[f"labels_k{K}"]
    kids = []
    for c in np.unique(lab):
        m = (lab == c) & valid
        dom, cnt = np.unique(gt[m], return_counts=True)
        if dom[cnt.argmax()] == d:
            kids.append((int(cnt.max()), int(c)))
    kids.sort(reverse=True)
    assert len(kids) >= 2, f"{domain} is not divided at k={K}"
    in_dom = (gt == d) & valid
    return in_dom, (lab == kids[0][1]) & in_dom, (lab == kids[1][1]) & in_dom


def rgba_image(grid, colours):
    step = float(np.min(np.diff(np.unique(grid[:, 0]))))
    ij = np.round((grid - grid.min(0)) / step).astype(int)
    img = np.zeros(tuple(ij.max(0)[::-1] + 1) + (4,))
    img[ij[:, 1], ij[:, 0]] = colours
    ext = (grid[:, 0].min() - step / 2, grid[:, 0].max() + step / 2,
           grid[:, 1].min() - step / 2, grid[:, 1].max() + step / 2)
    return img, ext


def scatter(ax, x, y, genes, slices, T, TICK):
    r_ = np.corrcoef(x, y)[0, 1]
    ax.axhline(0, color="#CCCCCC", lw=0.5, zorder=1)
    ax.axvline(0, color="#CCCCCC", lw=0.5, zorder=1)
    ax.scatter(x, y, s=4.5, facecolors="none", edgecolors=INK, linewidths=0.4, zorder=3)
    lim = 1.08 * max(np.abs(x).max(), np.abs(y).max())
    ax.plot([-lim, lim], [-lim, lim], "--", color="#999999", lw=0.5, zorder=2)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    score = x + y
    # gene names in two stacks clear of the points -- the "blue" genes down
    # the right edge from the top, the "orange" genes up the left edge from
    # the bottom -- each joined to its point by a hairline, so no two labels
    # overlap whatever the points do; gene symbols in italics
    up = sorted(np.argsort(-score)[:4], key=lambda j: -y[j])
    down = sorted(np.argsort(score)[:4], key=lambda j: x[j])
    # blue: a column down the right edge; orange: a row along the bottom edge
    # to the right of their points, where the lower-right quadrant is empty
    places = [(j, 0.985, 0.985 - i * 0.085, "right") for i, j in enumerate(up)]
    places += [(j, 0.24 + i * 0.19, 0.04, "left") for i, j in enumerate(down)]
    for j, xa, ya, ha in places:
        ax.annotate(genes[j], (x[j], y[j]), xytext=(xa, ya),
                    textcoords="axes fraction", fontsize=TICK, fontstyle="italic",
                    color=C_A if score[j] > 0 else C_B, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.35,
                                    shrinkA=0, shrinkB=1.5), zorder=4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TICK, length=1.6, pad=1.5)
    ax.set_xlabel(f"blue $-$ orange half, Bregma {slices[0]}", fontsize=T)
    ax.set_ylabel(f"blue $-$ orange half, Bregma {slices[1]}", fontsize=T, labelpad=2)
    ax.set_title(f"mean fiber difference per gene, $r$ = {r_:.2f}", fontsize=T,
                 color=INK, pad=3)
    return r_


def composition(ax, T, TICK):
    """Panel e: the neuronal clusters of Moffitt et al. in the two BST halves.
    For every cluster with >= 20 cells, the fraction of its cells that fall in
    the orange half, on each of the two posterior sections; the dashed line
    is the fraction of all BST neurons in that half, so a cluster on the line
    is split like the domain and one far from it is concentrated in one
    half (experiments/merfish/k_split_celltypes.py)."""
    import pandas as pd
    src = HERE.parent / "reported" / "merfish" / "k_split_celltypes.csv"
    if not src.exists():
        src = paths.results("merfish") / "k_split_celltypes.csv"
    ct = pd.read_csv(src)
    ct = ct[~ct.cluster.str.startswith("class:") & ct.slice.isin([0.19, 0.24])]
    piv = ct.pivot(index="cluster", columns="slice", values="frac_in_smaller")
    q = ct.pivot(index="cluster", columns="slice", values="q")
    n = ct.pivot(index="cluster", columns="slice", values="n")
    both = piv.dropna().index
    order = piv.loc[both].mean(1).sort_values().index
    base = {}
    for sl in (0.19, 0.24):
        d = ct[ct.slice == sl]
        base[sl] = d.n_smaller.sum() / d.n.sum()
    x = np.arange(len(order))
    style = {0.19: dict(marker="o", facecolors=C_B, edgecolors=C_B),
             0.24: dict(marker="s", facecolors="white", edgecolors=C_B)}
    for sl, dx in ((0.19, -0.16), (0.24, 0.16)):
        ax.axhline(base[sl], color=C_B if sl == 0.19 else "#999999", lw=0.5,
                   ls=(0, (3, 2)), zorder=1)
        ax.scatter(x + dx, piv.loc[order, sl], s=11, linewidths=0.6, zorder=3, **style[sl])
        for i, cl in enumerate(order):
            if q.loc[cl, sl] < 0.01:
                ax.text(x[i] + dx, piv.loc[cl, sl] + 0.05, "*", ha="center", va="bottom",
                        fontsize=TICK, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{cl}\n{int(n.loc[cl].min())}-{int(n.loc[cl].max())}" for cl in order],
                       fontsize=TICK)
    ax.set_ylim(-0.03, 1.08); ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(-0.6, len(order) - 0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TICK, length=1.6, pad=1.5)
    ax.set_ylabel("fraction of the cluster's\ncells in the orange half", fontsize=T, labelpad=2)
    ax.set_xlabel("neuronal cluster of Moffitt et al. (cells per section)", fontsize=T)
    h = [ax.scatter([], [], s=11, linewidths=0.6, **style[0.19]),
         ax.scatter([], [], s=11, linewidths=0.6, **style[0.24])]
    ax.legend(h, ["Bregma 0.19", "Bregma 0.24"], loc="lower right", frameon=False,
              fontsize=TICK, handletextpad=0.3, borderaxespad=0.2, ncol=2, columnspacing=1.0)
    ax.text(len(order) - 0.4, max(base.values()) + 0.03, "all BST neurons", ha="right",
            va="bottom", fontsize=TICK, color="#666666")
    ax.set_title("neuronal clusters in the two BST halves (* $q < 0.01$)", fontsize=T, color=INK, pad=3)


def main():
    ss.apply()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T, TICK = ss.TYPE_PT["normal"]["text"], ss.TYPE_PT["normal"]["tick"]
    load = lambda s: np.load(paths.results("merfish") / f"k_split_MERFISH_{s}.npz",
                             allow_pickle=False)
    z0 = load(CASES[0][1][0])
    genes = list(z0["gene_names"])
    aspect = np.ptp(z0["grid_coords"][:, 1]) / np.ptp(z0["grid_coords"][:, 0])

    map_w = 26.0
    row_h = map_w * aspect
    gap_case = 5.0
    n_rows = sum(len(c[1]) for c in CASES)
    comp_h = 40.0                     # panel e, the cluster composition of the BST halves
    height = 8.0 + n_rows * row_h + (n_rows - 1) * 1.0 + (len(CASES) - 1) * gap_case + 4.0 + comp_h
    fig, ax0 = ss.panel(W, height); ax0.remove()
    left = 9.0
    sx = left + 4 * (map_w + 1.0) + 12.0
    sw = W - sx - 4.0

    y = 8.0
    letters = iter("abcdef")
    rs = []
    for domain, slices, gene_pair in CASES:
        gi = [genes.index(g) for g in gene_pair]
        y_top = y
        diff = {}
        for s in slices:
            z = load(s); grid = z["grid_coords"]; F = z["fibers_norm"]
            under = under_tissue(grid, z["gt_int"] >= 0)
            in_dom, a, b = halves(z, domain)
            diff[s] = F[a].mean(0) - F[b].mean(0)
            col = np.tile(to_rgba(C_EMPTY), (len(grid), 1))
            col[under] = to_rgba(C_REST); col[in_dom] = to_rgba(C_DOM)
            col2 = np.tile(to_rgba(C_EMPTY), (len(grid), 1))
            col2[under] = to_rgba(C_REST); col2[a] = to_rgba(C_A); col2[b] = to_rgba(C_B)
            panels = [("rgba", col, f"annotated {domain}"),
                      ("rgba", col2, f"{domain} at $k$ = {K}")]
            for g, j in zip(gene_pair, gi):
                panels.append(("field", np.where(under, F[:, j], np.nan), (g, " fiber")))
            for c, (kind, arr, title) in enumerate(panels):
                ax = fig.add_axes([(left + c * (map_w + 1.0)) / W,
                                   1 - (y + row_h) / height, map_w / W, row_h / height])
                ss.spatial(ax)
                if kind == "rgba":
                    img, ext = rgba_image(grid, arr)
                    im = ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
                else:
                    img, ext = _image(arr, grid, np.isfinite(arr))
                    lo, hi = np.nanpercentile(arr, [2, 98])
                    im = ax.imshow(img, origin="lower", extent=ext, cmap="viridis",
                                   vmin=lo, vmax=hi, interpolation="nearest")
                ss.rasterize(im, dpi=600)
                ax.set_xlim(grid[:, 0].min(), grid[:, 0].max())
                ax.set_ylim(grid[:, 1].min(), grid[:, 1].max())
                if s == slices[0]:
                    if isinstance(title, tuple):
                        # gene symbol in italics, the word after it upright:
                        # two text objects, the second anchored to the first
                        t1 = ax.text(0.42, 1.03, title[0], transform=ax.transAxes, ha="right",
                                     va="bottom", fontsize=T, color=INK, fontstyle="italic")
                        ax.annotate(title[1], (1, 0), xycoords=t1, ha="left", va="bottom",
                                    fontsize=T, color=INK)
                    else:
                        ax.set_title(title, fontsize=T, pad=2, color=INK)
                if c == 0:
                    ax.text(-0.04, 0.5, f"Bregma {s}", transform=ax.transAxes,
                            rotation=90, va="center", ha="right", fontsize=T, color=INK)
                if c == 1:
                    n_d = in_dom.sum()
                    ax.text(0.02, 0.02, f"{a.sum() / n_d:.0%} / {b.sum() / n_d:.0%} of {domain}",
                            transform=ax.transAxes, fontsize=TICK, color=INK, va="bottom")
            y += row_h + 1.0
        fig.text((left - 7.5) / W, 1 - (y_top - 1.0) / height, next(letters),
                 fontsize=8, fontweight="bold", va="bottom", ha="left")
        # the scatter for this case, spanning its two map rows
        sax = fig.add_axes([sx / W, 1 - (y - 1.0 - 2.0) / height, sw / W,
                            (2 * row_h - 6.0) / height])
        r_ = scatter(sax, diff[slices[0]], diff[slices[1]], genes, slices, T, TICK)
        fig.text((sx - 7.5) / W, 1 - (y_top - 1.0) / height, next(letters),
                 fontsize=8, fontweight="bold", va="bottom", ha="left")
        rs.append((domain, r_))
        y += gap_case - 1.0

    # panel e under the maps, full width
    y += 2.0
    eax = fig.add_axes([(left + 10.0) / W, 1 - (y + comp_h - 11.0) / height,
                        (W - left - 10.0 - 4.0) / W, (comp_h - 15.0) / height])
    composition(eax, T, TICK)
    fig.text((left - 7.5) / W, 1 - (y - 1.0) / height, next(letters),
             fontsize=8, fontweight="bold", va="bottom", ha="left")

    out = OUTDIR / "fig5.pdf"
    ss.save_panel(fig, out, strict=False)
    print(f"wrote {out}  " + "  ".join(f"{d}: r = {r:.3f}" for d, r in rs))
    for p in verify(out, width_mm=W)[0]:
        print("  !", p)


if __name__ == "__main__":
    main()
