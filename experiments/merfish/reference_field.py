"""What the coordinate reference does to the boundary field, on MERFISH.

Which reference wins is not predicted by anything we can measure without the
annotation, and this script is what establishes that rather than asserting it.
It compares the two references on the five MERFISH slices, on three quantities:

  1. Boundary alignment -- the smoothed field at vertices adjacent to an
     annotated interface, over its value in domain interiors. Above 1 means the
     field is elevated at interfaces.
  2. The width of the field AFTER the normalisation the weighting applies. The
     weight is w(v) = exp(-alpha_b * Btilde(v) / median Btilde), so a uniform
     rescaling of the field cancels and only its shape relative to its own
     median can matter. Reported as the 90th and 99th percentiles over the
     median, and the coefficient of variation.
  3. The spread of w itself, which is what the clustering actually sees.

The point is the dissociation. On MERFISH the tissue reference aligns with the
annotated interfaces slightly BETTER than the local reference and scores lower
(Table 2), so the reference's effect on the partition does not run through
boundary localisation -- the same dissociation reported for the edge scores in
Section "Why targeting discontinuities outperforms spectral methods". What it
does do is compress the normalised field, so the weighting attenuates less.

Whether the converse holds on the three sections where the tissue reference
wins is NOT tested here: those runs do not cache their fibers and stalks. Until
they do, this is a measurement on one dataset and not a mechanism.

    make merfish-reference-field

Reads the shared MERFISH cache (``make cache``), writes
``merfish/reference_field.csv`` under the results root.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spsheaf import paths

SLICES = ["0.04", "0.09", "0.14", "0.19", "0.24"]


def disagreement(stalks, edges, feat):
    """D(e) = || L^T xi_u - R^T xi_v ||^2, batched over edges."""
    Uu, Uv = stalks[edges[:, 0]], stalks[edges[:, 1]]
    L, _, Rt = np.linalg.svd(np.einsum("eig,eih->egh", Uu, Uv))
    xu = np.einsum("eig,ei->eg", Uu, feat[edges[:, 0]])
    xv = np.einsum("eih,ei->eh", Uv, feat[edges[:, 1]])
    d = np.einsum("egh,eg->eh", L, xu) - np.einsum("ehg,eg->eh", Rt, xv)
    return np.sum(d ** 2, axis=1)


def vertex_field(D, edges, n):
    s = np.zeros(n)
    c = np.zeros(n)
    for col in (0, 1):
        np.add.at(s, edges[:, col], D)
        np.add.at(c, edges[:, col], 1)
    return s / np.maximum(c, 1)


def smooth(B, grid, sigma):
    tree = cKDTree(grid)
    out = np.empty_like(B)
    for i, p in enumerate(grid):
        j = tree.query_ball_point(p, 2.5 * sigma)
        w = np.exp(-0.5 * (np.linalg.norm(grid[j] - p, axis=1) / sigma) ** 2)
        out[i] = (w * B[j]).sum() / w.sum()
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--alpha-b", type=float, default=0.5,
                   help="the weighting strength the field is read through")
    a = a.parse_args()

    rows = []
    for s in SLICES:
        f = paths.cache("merfish") / f"revision_s{s}.npz"
        if not f.exists():
            print(f"  missing {f.name}; run 'make cache SLICE=MERFISH_{s}.h5ad'")
            continue
        z = np.load(f, allow_pickle=True)
        stalks, edges, grid = z["stalks"], z["edges"], z["grid"]
        gt, ok = z["gt_int"], z["gt_valid"]
        sigma = float(z["params"][5])

        e0, e1 = edges[:, 0], edges[:, 1]
        scored = ok[e0] & ok[e1]
        crosses = (gt[e0] != gt[e1]) & scored
        interior = (gt[e0] == gt[e1]) & scored
        at_iface = np.zeros(len(grid), bool)
        at_iface[e0[crosses]] = True
        at_iface[e1[crosses]] = True

        for name, feat in (("local", z["fib_local"]), ("tissue", z["fib_global"])):
            D = disagreement(stalks, edges, feat)
            B = smooth(vertex_field(D, edges, len(grid)), grid, sigma)
            Bn = B / np.median(B)
            w = np.exp(-a.alpha_b * Bn)
            sel, oth = ok & at_iface, ok & ~at_iface
            rows.append(dict(
                slice=s, reference=name,
                iface_over_interior=Bn[sel].mean() / Bn[oth].mean(),
                edge_iface_over_interior=D[crosses].mean() / D[interior].mean(),
                D_interior_mean=D[interior].mean(),
                p90_over_median=np.percentile(Bn, 90),
                p99_over_median=np.percentile(Bn, 99),
                field_cv=Bn.std() / Bn.mean(),
                w_sd=w.std(), w_range_5_95=np.percentile(w, 95) - np.percentile(w, 5),
                n_iface_edges=int(crosses.sum()), n_interior_edges=int(interior.sum()),
            ))
            r = rows[-1]
            print(f"  {s}  {name:7s} interface/interior {r['iface_over_interior']:.3f}"
                  f"  p99/median {r['p99_over_median']:.2f}"
                  f"  CV {r['field_cv']:.2f}  w sd {r['w_sd']:.3f}")

    if not rows:
        raise SystemExit("no cached slices found")
    d = pd.DataFrame(rows)
    out = paths.results("merfish") / "reference_field.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)

    print("\nmean over the five slices:")
    g = d.groupby("reference")[["iface_over_interior", "p90_over_median",
                                "p99_over_median", "field_cv", "w_sd",
                                "w_range_5_95", "D_interior_mean"]].mean()
    print(g.round(3).to_string())
    n = d.groupby("slice").apply(
        lambda x: x.set_index("reference").iface_over_interior.idxmax(), include_groups=False)
    print(f"\ntissue aligns better on {int((n == 'tissue').sum())} of {len(n)} slices; "
          f"it scores lower on the partition (Table 2), which is the dissociation.")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
