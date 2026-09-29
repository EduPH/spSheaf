"""The BST split at k = 9 on the two posterior MERFISH sections.

Raising k one above the annotated count divides the annotated BST into two
halves on slices 0.19 and 0.24 (``experiments/merfish/k_split.py``). This
draws the two halves beside the annotation and the two fibers that
differ most between them, and, on the right, the per-gene difference between
the halves on one slice against the other -- the number that says the split
is the same split on both sections.

    python sup_ksplit.py
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
SLICES = ("0.19", "0.24")
DOMAIN, K = "BST", 9
GENES = ("Isl1", "Esr1")
C_A, C_B = "#0072B2", "#E69F00"
C_DOM, C_REST, C_EMPTY = "#56B4E9", "#E4E4E4", "#F7F7F7"
INK = "#333333"


def halves(z):
    """The two largest k-clusters whose majority annotation is DOMAIN."""
    gt, valid = z["gt_int"], z["gt_int"] >= 0
    names = list(z["gt_names"]); d = names.index(DOMAIN)
    lab = z[f"labels_k{K}"]
    kids = []
    for c in np.unique(lab):
        m = (lab == c) & valid
        dom, cnt = np.unique(gt[m], return_counts=True)
        if dom[cnt.argmax()] == d:
            kids.append((int(cnt.max()), int(c)))
    kids.sort(reverse=True)
    assert len(kids) >= 2, f"{DOMAIN} is not split at k={K}"
    in_dom = (gt == d) & valid
    a = (lab == kids[0][1]) & in_dom
    b = (lab == kids[1][1]) & in_dom
    return in_dom, a, b


def rgba_image(grid, colours):
    step = float(np.min(np.diff(np.unique(grid[:, 0]))))
    ij = np.round((grid - grid.min(0)) / step).astype(int)
    img = np.zeros(tuple(ij.max(0)[::-1] + 1) + (4,))
    img[ij[:, 1], ij[:, 0]] = colours
    extent = (grid[:, 0].min() - step / 2, grid[:, 0].max() + step / 2,
              grid[:, 1].min() - step / 2, grid[:, 1].max() + step / 2)
    return img, extent


def main():
    ss.apply()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    T, TICK = ss.TYPE_PT["normal"]["text"], ss.TYPE_PT["normal"]["tick"]

    Z = {s: np.load(paths.results("merfish") / f"k_split_MERFISH_{s}.npz",
                    allow_pickle=False) for s in SLICES}
    genes = list(Z[SLICES[0]]["gene_names"])
    gi = [genes.index(g) for g in GENES]

    grid0 = Z[SLICES[0]]["grid_coords"]
    aspect = np.ptp(grid0[:, 1]) / np.ptp(grid0[:, 0])
    map_w = 27.0
    row_h = map_w * aspect
    height = 2 * row_h + 12.0
    fig, ax0 = ss.panel(W, height)
    ax0.remove()
    left = 8.0
    # four map columns per row, then the scatter on the right
    axes = [[fig.add_axes([(left + c * (map_w + 1.0)) / W,
                           1 - (8.0 + (r + 1) * row_h + r * 1.0) / height,
                           map_w / W, row_h / height])
             for c in range(4)] for r in range(2)]
    sx = left + 4 * (map_w + 1.0) + 12.0
    sax = fig.add_axes([sx / W, 9.0 / height, (W - sx - 4.0) / W,
                        (height - 17.0) / height])

    diff = {}
    for r, s in enumerate(SLICES):
        z = Z[s]; grid = z["grid_coords"]; F = z["fibers_norm"]
        under = under_tissue(grid, z["gt_int"] >= 0)
        in_dom, a, b = halves(z)
        diff[s] = F[a].mean(0) - F[b].mean(0)

        # (1) annotation with the domain picked out
        col = np.tile(to_rgba(C_EMPTY), (len(grid), 1))
        col[under] = to_rgba(C_REST); col[in_dom] = to_rgba(C_DOM)
        # (2) the two halves
        col2 = np.tile(to_rgba(C_EMPTY), (len(grid), 1))
        col2[under] = to_rgba(C_REST); col2[a] = to_rgba(C_A); col2[b] = to_rgba(C_B)
        panels = [("rgba", col, f"annotated {DOMAIN}"),
                  ("rgba", col2, f"{DOMAIN} at $k$ = {K}")]
        for g, j in zip(GENES, gi):
            v = np.where(under, F[:, j], np.nan)
            panels.append(("field", v, f"{g} fiber"))
        for c, (kind, arr, title) in enumerate(panels):
            ax = axes[r][c]; ss.spatial(ax)
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
            if r == 0:
                ax.set_title(title, fontsize=T, pad=2, color=INK)
            if c == 0:
                ax.text(-0.04, 0.5, f"Bregma {s}", transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=T, color=INK)
        n_a, n_b, n_d = a.sum(), b.sum(), in_dom.sum()
        axes[r][1].text(0.02, 0.02, f"{n_a / n_d:.0%} / {n_b / n_d:.0%} of {DOMAIN}",
                        transform=axes[r][1].transAxes, fontsize=TICK, color=INK,
                        va="bottom")

    # (5) the split on one slice against the other, gene by gene
    x, y = diff[SLICES[0]], diff[SLICES[1]]
    r_ = np.corrcoef(x, y)[0, 1]
    sax.axhline(0, color="#CCCCCC", lw=0.5, zorder=1)
    sax.axvline(0, color="#CCCCCC", lw=0.5, zorder=1)
    sax.scatter(x, y, s=5, facecolors="none", edgecolors=INK, linewidths=0.4, zorder=3)
    lim = 1.05 * max(np.abs(x).max(), np.abs(y).max())
    sax.plot([-lim, lim], [-lim, lim], "--", color="#999999", lw=0.5, zorder=2)
    sax.set_xlim(-lim, lim); sax.set_ylim(-lim, lim)
    # label the genes that lead the split, on either side
    score = x + y
    lead = list(np.argsort(score)[:6]) + list(np.argsort(-score)[:6])
    for j in lead:
        sax.annotate(genes[j], (x[j], y[j]), xytext=(2.5, 1.5), textcoords="offset points",
                     fontsize=TICK, color=C_A if score[j] > 0 else C_B)
    for sp in ("top", "right"):
        sax.spines[sp].set_visible(False)
    sax.tick_params(labelsize=TICK, length=1.6, pad=1.5)
    sax.set_xlabel(f"blue $-$ orange half, Bregma {SLICES[0]}", fontsize=T)
    sax.set_ylabel(f"blue $-$ orange half, Bregma {SLICES[1]}", fontsize=T, labelpad=2)
    sax.set_title(f"mean fiber difference per gene, $r$ = {r_:.2f}",
                  fontsize=T, color=INK, pad=3)

    fig.text(left / W, 1 - 1.2 / height,
             f"Raising $k$ from the annotated 8 to 9 divides the annotated {DOMAIN} into "
             "the same two halves on both posterior sections.",
             fontsize=TICK, va="top", ha="left", color=INK)

    out = OUTDIR / "sup_ksplit.pdf"
    ss.save_panel(fig, out, strict=False)
    print(f"wrote {out}   r = {r_:.3f}")
    for p in verify(out, width_mm=W)[0]:
        print("  !", p)


if __name__ == "__main__":
    main()
