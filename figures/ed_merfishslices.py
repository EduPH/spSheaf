"""Supplementary Fig. 1: the five MERFISH slices, one column each, drawn from the cache.

Replaces the composite of five pre-rendered rasters. Every panel comes from
``data/cache/merfish/revision_s<slice>.npz``, the cache the manuscript's
numbers are computed on, so the maps cannot drift from Table 1: the domains
drawn here are re-clustered from that cache with the published parameters and
the ARI printed on each row is recomputed, not copied.

Four columns per slice: the annotation, the smoothed expression the sheaf is
built on (first principal component of the fibers), the smoothed disagreement
field as a percentile rank, and the recovered domains.

The ARI on each row is Table 1's value for that slice, read from the tracked
seed-sensitivity table rather than recomputed here, so the figure cannot show
a number the table does not. Table 1 scores at cell level over 20 seeds; the
map drawn is one seed at grid level, which is why the grid-level ARI printed
to the console differs by up to 0.02.

    python ed_merfishslices.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import sheafstyle as ss
from spsheaf import clustering, evaluation, paths
from verify_figure import verify

HERE = Path(__file__).parent
OUTDIR = HERE / "out"
SLICES = ["0.04", "0.09", "0.14", "0.19", "0.24"]
N_CLUSTERS, SEED, ALPHA_S, ALPHA_B = 8, 42, 0.5, 0.5

DOMAIN_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7",
                 "#D55E00", "#56B4E9", "#F0E442", "#8C8C8C"]
EMPTY = "#EDEDED"
W = 180.0


def table1_ari():
    """Per-slice ARI of Table 1: the 20-seed cell-level means."""
    import csv
    p = (paths.results("benchmark_sdmbench") / "output_seed_sensitivity"
         / "seed_sensitivity_per_slice.csv")
    with p.open() as fh:
        return {r["slice"].replace("MERFISH_", "").replace(".h5ad", ""):
                float(r["ari_mean"]) for r in csv.DictReader(fh)}


def _domain_names():
    """Annotation classes in the order of the integer codes (alphabetical,
    as evaluation.map_ground_truth codes them); from a dump that carries
    them, else the published list."""
    import json
    p = paths.results("merfish") / "panels" / "segfree_maps_s0.19.json"
    if p.exists():
        return json.loads(p.read_text())["gt_names"]
    return ["BST", "MPA", "MPN", "PV", "PVH", "PVT", "V3", "fx"]


def _cache(slice_id):
    p = paths.data_root() / "cache" / "merfish" / f"revision_s{slice_id}.npz"
    if not p.exists():
        raise SystemExit(f"{p} missing; run  make cache SLICE=MERFISH_{slice_id}.h5ad")
    d = np.load(p, allow_pickle=True)
    return (d["grid"], d["fib"].astype(float), d["boundary_smooth"],
            d["gt_int"], d["gt_valid"], [tuple(e) for e in d["edges"]])


def _domains(fib, grid, field, edges):
    lab = clustering.boundary_aware_clustering(
        fib, grid, field, edges, n_clusters=N_CLUSTERS, random_state=SEED,
        spatial_weight=ALPHA_S, boundary_alpha=ALPHA_B, verbose=False)
    lab, _ = clustering.spatial_cleanup(lab, edges, grid_coords=grid, verbose=False)
    return lab


def under_tissue(grid, valid, erode=6.0, frac=0.75):
    """Grid points in the interior, away from the section edge.

    The grid is built over the section, so its own outer ring *is* the tissue
    edge: there the fiber neighbourhood is one-sided and the fibers beyond it
    are empty, and the disagreement on that ring is far above the internal
    transitions. Left in, it owns the colour scale and the panel shows the
    outline of the section instead of the structure it is meant to show. A
    point counts as interior when it has at least `frac` of the largest
    neighbour count within `erode` spacings -- the smoothing radius, six
    spacings at the published sigma_s -- which drops the ring without
    depending on the annotation, whose invalid points are speckled throughout
    the interior.
    """
    from scipy.spatial import cKDTree
    step = float(np.min(np.diff(np.unique(grid[:, 0]))))
    n = cKDTree(grid).query_ball_point(grid, r=erode * step, return_length=True)
    return n >= frac * n.max()


def _image(values, grid, mask=None):
    """Grid-valued field as an image on the regular lattice; outside mask = nan."""
    step = float(np.min(np.diff(np.unique(grid[:, 0]))))
    ij = np.round((grid - grid.min(0)) / step).astype(int)
    img = np.full(tuple(ij.max(0)[::-1] + 1), np.nan)
    sel = np.ones(len(grid), bool) if mask is None else mask
    img[ij[sel, 1], ij[sel, 0]] = values[sel]
    extent = (grid[:, 0].min() - step / 2, grid[:, 0].max() + step / 2,
              grid[:, 1].min() - step / 2, grid[:, 1].max() + step / 2)
    return img, extent


def _labels_image(lab, grid, valid):
    from matplotlib.colors import to_rgba
    step = float(np.min(np.diff(np.unique(grid[:, 0]))))
    ij = np.round((grid - grid.min(0)) / step).astype(int)
    img = np.zeros(tuple(ij.max(0)[::-1] + 1) + (4,))
    rgba = np.array([to_rgba(c) for c in DOMAIN_COLORS])
    col = np.where(valid[:, None], rgba[np.clip(lab, 0, len(rgba) - 1) % len(rgba)],
                   np.array(to_rgba(EMPTY)))
    col[lab < 0] = to_rgba(EMPTY)
    img[ij[:, 1], ij[:, 0]] = col
    extent = (grid[:, 0].min() - step / 2, grid[:, 0].max() + step / 2,
              grid[:, 1].min() - step / 2, grid[:, 1].max() + step / 2)
    return img, extent


def main():
    ss.apply()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    from scipy.stats import rankdata

    t1 = table1_ari()
    rows = []
    for s in SLICES:
        grid, fib, field, gti, valid, edges = _cache(s)
        lab = _domains(fib, grid, field, edges)
        matched = evaluation.hungarian_match_labels(lab, gti, valid)
        pc1 = fib @ np.linalg.svd(fib - fib.mean(0), full_matrices=False)[2][0]
        rows.append((s, grid, pc1, field, gti, valid, matched, t1[s]))
        print(f"  slice {s}: Table 1 ARI {t1[s]:.3f} "
              f"(this seed, at grid level: {evaluation.compute_ari(lab, gti, valid):.3f})")

    # slices across, views down: five columns fit the 180 mm canvas at a
    # height that leaves room for the caption, where five rows did not
    aspect = np.ptp(rows[0][1][:, 1]) / np.ptp(rows[0][1][:, 0])
    left_mm = 21.0
    cell_w = (W - left_mm - 1.0) / len(SLICES)
    row_h = cell_w * aspect
    key_mm = 10.0                     # a key row under the maps: domain colours and the two scales
    height = 4 * row_h + 5.0 + key_mm
    fig, axes = ss.panel_grid(W, height, 4, len(SLICES))
    fig.subplots_adjust(left=left_mm / W, right=0.997, top=1 - 4.0 / height,
                        bottom=key_mm / height, wspace=0.035, hspace=0.05)

    views = ["annotation", "smoothed expression\n(fiber PC1)",
             "sheaf field $\\tilde B$\n(rank)", "spSheaf domains"]
    T = ss.TYPE_PT["normal"]["text"]
    for c, (s_, grid, pc1, field, gti, valid, matched, ari) in enumerate(rows):
        under = under_tissue(grid, valid)
        rank = np.full(len(field), np.nan)
        rank[under] = rankdata(field[under]) / under.sum()
        panels = [
            ("labels", gti, valid, None),
            ("field", np.where(under, pc1, np.nan), under, "viridis"),
            ("field", rank, under, "magma"),
            ("labels", matched, np.ones(len(matched), bool), ari),
        ]
        for r, (kind, arr, mask, extra) in enumerate(panels):
            ax = axes[r, c]
            ss.spatial(ax)
            if kind == "field":
                img, extent = _image(arr, grid, mask)
                finite = arr[np.isfinite(arr)]
                lo, hi = np.percentile(finite, [2, 98])
                im = ax.imshow(img, origin="lower", extent=extent, cmap=extra,
                               vmin=lo, vmax=hi, interpolation="nearest")
            else:
                img, extent = _labels_image(arr, grid, mask)
                im = ax.imshow(img, origin="lower", extent=extent,
                               interpolation="nearest")
            ss.rasterize(im, dpi=600)
            ax.set_xlim(grid[:, 0].min(), grid[:, 0].max())
            ax.set_ylim(grid[:, 1].min(), grid[:, 1].max())
            if r == 0:
                ax.set_title(f"Bregma {s_}", fontsize=T, pad=2)
            if r == 3:
                # inside the panel, on the empty corner above the section's
                # shoulder, rather than as a title squeezed between two rows
                ax.text(0.02, 0.98, f"ARI {extra:.2f}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=T,
                        color=ss.METHOD_COLORS.get("spSheaf", "#333333"),
                        bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))
    for r, name in enumerate(views):
        axes[r, 0].set_ylabel(name, fontsize=T, labelpad=3)
        axes[r, 0].yaxis.set_visible(True)
        axes[r, 0].set_yticks([])

    # key row: the eight annotation classes (the colours of rows 1 and 4) and
    # the two continuous scales, drawn once for the whole figure
    from matplotlib import cm
    from matplotlib.colors import Normalize
    TICK = ss.TYPE_PT["normal"]["tick"]
    names = _domain_names()
    kx0 = left_mm / W
    lax = fig.add_axes([kx0, 0.0, 0.50, key_mm / height])
    lax.set_axis_off()
    for i, nm in enumerate(names):
        lax.scatter([], [], s=12, c=DOMAIN_COLORS[i], label=nm, marker="s", linewidths=0)
    lax.scatter([], [], s=12, c=EMPTY, label="unannotated", marker="s", linewidths=0)
    lax.legend(loc="center left", ncol=5, frameon=False, fontsize=TICK,
               handletextpad=0.3, columnspacing=1.0, labelspacing=0.4, borderaxespad=0)
    # the two scales, bar above and its name below, in the right half of the key row
    bar_h = 1.6 / height
    for j, (cmap, label) in enumerate([("viridis", "expression PC1 (2nd-98th percentile)"),
                                       ("magma", "field rank (0-1)")]):
        x0 = 0.60 + 0.21 * j
        cax = fig.add_axes([x0, 5.2 / height, 0.16, bar_h])
        cb = fig.colorbar(cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap),
                          cax=cax, orientation="horizontal", ticks=[])
        cb.outline.set_linewidth(0.4)
        cax.text(0.5, -0.9, label, transform=cax.transAxes, ha="center", va="top", fontsize=TICK)

    out = OUTDIR / "ed_merfishslices.pdf"
    ss.save_panel(fig, out, strict=False)
    print(f"\nwrote {out}")
    problems, info = verify(out, width_mm=W)
    for m in problems:
        print("  !", m)


if __name__ == "__main__":
    main()
