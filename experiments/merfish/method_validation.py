"""Does the sheaf term see a boundary that no kernel statistic on fibers can?

The edge-score ablation (``sheaf_ablation.py``) answers this on real data by
replacing the sheaf score with kernel and subspace controls. This is the
complementary check: a construction in which the answer is known in advance
because the two regions are built to be indistinguishable to every kernel
statistic on the fibers themselves.

Two regions share

  * the same mean expression vector, and
  * the same distribution of neighbour fiber distances ``||f_u - f_v||``,

but their fibers are drawn in orthogonal linear subspaces of gene space. A
Gaussian-kernel view therefore has nothing to detect: mean expression is flat
across the interface by construction, and so is neighbour distance.

What the construction separates is the *subspace* comparison. Stalks are
aligned within a region and orthogonal across the interface, so the
principal-angle disagreement between neighbouring stalks is an order of
magnitude larger there. The coordinate disagreement the pipeline uses is not:
its contrast is at or slightly below 1, because a stalk built from a
neighbourhood straddling the interface spans a blend of the two subspaces and
captures less of each fiber's local fluctuation, which shrinks the coordinates
being differenced. That is the same ordering the direct boundary evaluation
finds on real MERFISH data, where the subspace scores localise the annotated
interfaces best and the coordinate score localises them worst; the coordinate
score responds over the whole transition rather than at the interface, which
is the width the exponential weight acts on.

Reported, and asserted, are the interface contrast of each field --- the mean
over edges crossing the interface divided by the mean over edges within a
region, so 1.0 means the field sees nothing --- and the principal-angle cosine
within versus across regions.

    python -m experiments.merfish.method_validation
    make method-validation
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from spsheaf import graph, sheaf


def build(n_side=40, n_genes=40, dim=6, noise=0.35, seed=0):
    """Two regions, matched in mean and neighbour distance, orthogonal in subspace."""
    rng = np.random.default_rng(seed)
    xs = np.arange(n_side, dtype=float)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    region = (coords[:, 0] >= n_side / 2).astype(int)       # left | right

    # two orthogonal subspaces of gene space, from one orthonormal basis
    Q = np.linalg.qr(rng.normal(size=(n_genes, 2 * dim)))[0]
    bases = [Q[:, :dim], Q[:, dim:2 * dim]]

    mean_expression = rng.normal(size=n_genes)              # shared, so it is flat
    fibers = np.empty((len(coords), n_genes))
    for r in (0, 1):
        m = region == r
        # identical coefficient distribution in each region, so the fiber
        # distance distribution matches; only the subspace differs
        coef = rng.normal(scale=noise, size=(m.sum(), dim))
        fibers[m] = mean_expression + coef @ bases[r].T
    return coords, fibers, region


def contrast(values, edges, region):
    """Mean over interface-crossing edges / mean over within-region edges."""
    crossing = region[edges[:, 0]] != region[edges[:, 1]]
    return float(values[crossing].mean() / values[~crossing].mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-side", type=int, default=40)
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write the measured values here")
    a = ap.parse_args(argv)

    coords, fibers, region = build(n_side=a.n_side, dim=a.dim, seed=a.seed)
    edges = np.asarray(graph.build_spatial_graph(coords, k=8, verbose=False))
    stalks = sheaf.compute_stalks(fibers, coords, uniform_dim=a.dim,
                                  k_nbrs=50, verbose=False)
    restrictions = sheaf.compute_restriction_maps(stalks, edges, a.dim, verbose=False)
    raw, _ = sheaf.compute_boundary_scores(fibers, coords, stalks, edges, restrictions,
                                           smooth_radius=0.0, verbose=False, k_nbrs=50)

    # the two kernel views, on the same edges
    mean_view = np.abs(fibers.mean(1)[edges[:, 0]] - fibers.mean(1)[edges[:, 1]])
    dist_view = np.linalg.norm(fibers[edges[:, 0]] - fibers[edges[:, 1]], axis=1)
    # the coordinate disagreement the pipeline uses is per vertex; take each
    # edge's mean of its endpoints
    coord_view = 0.5 * (raw[edges[:, 0]] + raw[edges[:, 1]])
    # the subspace comparison: mean principal-angle cosine, as a disagreement
    cos = np.array([np.linalg.svd(stalks[u].T @ stalks[v], compute_uv=False).mean()
                    for u, v in edges])
    subspace_view = 1.0 - cos
    crossing = region[edges[:, 0]] != region[edges[:, 1]]

    out = {
        "n_vertices": int(len(coords)), "n_edges": int(len(edges)),
        "contrast_mean_expression": contrast(mean_view + 1e-12, edges, region),
        "contrast_neighbour_distance": contrast(dist_view, edges, region),
        "contrast_subspace": contrast(subspace_view, edges, region),
        "contrast_coordinate": contrast(coord_view, edges, region),
        "cos_within": float(cos[~crossing].mean()),
        "cos_across": float(cos[crossing].mean()),
    }
    print(f"vertices {out['n_vertices']}, edges {out['n_edges']}")
    print("interface contrast (crossing / within, 1.0 = the field sees nothing)")
    print(f"  mean expression        {out['contrast_mean_expression']:.2f}")
    print(f"  neighbour distance     {out['contrast_neighbour_distance']:.2f}")
    print(f"  subspace disagreement  {out['contrast_subspace']:.1f}")
    print(f"  coordinate disagreement {out['contrast_coordinate']:.2f}")
    print(f"principal-angle cosine: within {out['cos_within']:.3f}, "
          f"across {out['cos_across']:.3f}")

    # the construction is only valid if both kernel views are genuinely flat
    assert abs(out["contrast_mean_expression"] - 1.0) < 0.25, out
    assert abs(out["contrast_neighbour_distance"] - 1.0) < 0.25, out
    # the subspace comparison is what separates the two regions
    assert out["contrast_subspace"] > 5.0, out
    assert out["cos_within"] > 0.9 and out["cos_across"] < 0.85, out
    # and the coordinate score does not mark the interface, as on real data
    assert out["contrast_coordinate"] < out["contrast_subspace"], out
    print("OK: both kernel views are flat; the subspace comparison separates the "
          "regions; the coordinate score does not mark the interface")

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
