"""Can k be chosen without an annotation? Established where the answer is known.

The manuscript takes the number of domains k as an input and sets it to the
annotated count, which is defensible for a benchmark and useless to a reader
with an unannotated section. This script is the attempt at practical guidance,
run first on synthetic tissue because a benchmark annotation is not the
uniquely correct k either: it is one expert's granularity, and an annotation
that merges two programs into one label makes the "true" k a convention rather
than a fact. On a construction we generate, k IS a fact.

Two criteria, both annotation-free:

  * grid-phase stability. The lattice the fibers sit on has an arbitrary
    phase. Shifting it by a fraction of the spacing re-samples the same
    molecules into different fibers, so two runs that differ only in that
    phase are two views of one tissue. Their agreement (ARI, compared on a
    common reference lattice) measures whether the partition is a property of
    the tissue or of the discretisation. Boundaries the data does not support
    have to be placed somewhere arbitrary, and should move.
  * silhouette on the fibers, the conventional criterion, which
    ``experiments/merfish/k_sensitivity.py`` already shows is a poor selector
    on its own -- its argmax over the five MERFISH slices is 10, 8, 9, 5, 9
    against an annotated 8.

Why the grid phase and not the k-means seed: on the five real MERFISH slices
the seeds at a fixed k agree to within 0.001 ARI, so seed resampling measures
nothing there. On the synthetic sweeps it is mostly but not always inert
(0.63--1.00 across k), so it is reported alongside as a control rather than
dismissed. The grid phase is the perturbation that moves the data.

The expected failure mode is named in advance, because stability-based
selection is known to favour coarse partitions -- a two-way split is trivially
reproducible -- so a stability curve alone will tend to point low. Reading it
together with the silhouette, which points high, is the point: two criteria
with opposite biases bracket k instead of pretending to identify it.

Designs:
  flat    K well-separated domains, so k is unambiguous and a criterion that
          cannot find it here will not find it anywhere.
  nested  K domains of which two pairs are siblings a small distance apart in
          expression, so K, K-1 and K-2 are all defensible readings. This is
          the synthetic analogue of an annotation's granularity being a
          choice, and what a criterion should report as a range.

    python -m experiments.synthetic.k_selection --design flat --k-true 6
    python -m experiments.synthetic.k_selection --design nested --k-true 6
    make k-selection
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, silhouette_score

from spsheaf import paths, pipeline

SLICE = "synthetic"


def build(design, k_true, n_genes=60, n_mol=220_000, extent=900.0,
          sep=1.0, sibling_sep=0.22, seed=0):
    """A molecule table whose domain structure is known by construction.

    Domains are Voronoi cells of k_true seed points -- irregular and
    contiguous, like tissue and unlike bands, so the spatial term cannot
    recover them for free. Each domain draws its genes from its own
    multinomial. In the nested design two pairs of domains share a program up
    to ``sibling_sep``, so the honest answer for k is an interval.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0.08, 0.92, size=(k_true, 2)) * extent

    # gene programs: one direction per domain, siblings sharing most of theirs
    base = rng.normal(size=(k_true, n_genes))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    if design == "nested":
        for a, b in ((0, 1), (2, 3)):
            if b < k_true:
                base[b] = base[a] + sibling_sep * base[b]
                base[b] /= np.linalg.norm(base[b])
    logits = sep * base
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)

    xy = rng.uniform(0, extent, size=(n_mol, 2))
    dom = np.argmin(((xy[:, None, :] - seeds[None]) ** 2).sum(-1), axis=1)
    gene = np.empty(n_mol, dtype=int)
    for d in range(k_true):
        m = dom == d
        gene[m] = rng.choice(n_genes, size=int(m.sum()), p=probs[d])

    mol = pd.DataFrame({"gene": [f"g{i:03d}" for i in gene],
                        "x": xy[:, 0], "y": xy[:, 1],
                        "matched_h5ad_file": SLICE})
    return mol, seeds, dom


def truth_on(grid, seeds):
    """The generating domain of each grid point: its nearest Voronoi seed."""
    return np.argmin(((grid[:, None, :] - seeds[None]) ** 2).sum(-1), axis=1)


def run(mol, k, offset, *, grid_spacing, bandwidth, smooth_radius,
        spatial_weight, uniform_dim, seed):
    res = pipeline.run_single_slice(
        mol=mol, slice_name=SLICE, n_clusters=k,
        grid_spacing=grid_spacing, bandwidth=bandwidth,
        smooth_radius=smooth_radius, spatial_weight=spatial_weight,
        uniform_dim=uniform_dim, boundary_alpha=0.5,
        k_graph=6, k_nbrs=50, cleanup=True, eval_at_cell_level=False,
        grid_offset=offset, random_state=seed, verbose=False)
    return np.asarray(res["grid_coords"]), np.asarray(res["labels"]), res


def transfer(src_grid, src_lab, ref_grid):
    """Labels carried to a common lattice, so two phases can be compared."""
    _, idx = cKDTree(src_grid).query(ref_grid, k=1)
    return src_lab[idx]


def bracket(d, *, plateau_k=3, tol=0.95):
    """The procedure's output: an interval, not a number.

    The silhouette's peak is the point estimate. Grid-phase stability supplies
    the upper bound, and only the upper bound: a coarse partition is trivially
    reproducible, so the stability curve is high and flat BELOW the supported k
    and its maximum sits at the coarsest split. What carries information is
    where it leaves that plateau, taken here as the largest k still within
    ``tol`` of the mean stability over k <= ``plateau_k``.
    """
    m = d.groupby("k")[["stability", "silhouette"]].mean()
    plateau = float(m.loc[m.index <= plateau_k, "stability"].mean())
    ok = m.index[m.stability >= tol * plateau]
    upper = int(ok.max()) if len(ok) else int(m.index.min())
    return int(m.silhouette.idxmax()), upper


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--design", choices=("flat", "nested"), default="flat")
    a.add_argument("--k-true", type=int, default=6)
    a.add_argument("--k", type=int, nargs=2, default=[2, 12], metavar=("MIN", "MAX"))
    a.add_argument("--grid-spacing", type=float, default=12.0)
    a.add_argument("--bandwidth", type=float, default=22.0)
    a.add_argument("--smooth-radius", type=float, default=50.0)
    a.add_argument("--spatial-weight", type=float, default=0.5)
    a.add_argument("--uniform-dim", type=int, default=6)
    a.add_argument("--n-mol", type=int, default=220_000)
    a.add_argument("--sep", type=float, default=1.0)
    a.add_argument("--shifts", type=float, nargs="+", default=[0.5],
                   help="grid-phase offsets as fractions of the spacing")
    a.add_argument("--seed", type=int, default=0)
    a = a.parse_args()

    mol, seeds, _ = build(a.design, a.k_true, n_mol=a.n_mol, sep=a.sep,
                          seed=a.seed)
    print(f"{a.design} design, k_true = {a.k_true}, {len(mol):,} molecules, "
          f"{mol.gene.nunique()} genes")
    cfg = dict(grid_spacing=a.grid_spacing, bandwidth=a.bandwidth,
               smooth_radius=a.smooth_radius, spatial_weight=a.spatial_weight,
               uniform_dim=a.uniform_dim)
    offsets = [(f * a.grid_spacing, f * a.grid_spacing) for f in a.shifts]

    rng_sil = np.random.default_rng(a.seed)
    print(f"\n{'k':>3s} {'stability':>10s} {'silhouette':>11s} "
          f"{'ARI vs truth':>13s} {'seed stab.':>11s}")
    rows = []
    for k in range(a.k[0], a.k[1] + 1):
        g0, l0, res0 = run(mol, k, None, seed=a.seed, **cfg)
        truth = truth_on(g0, seeds)
        fib0 = np.asarray(res0["fibers_norm"])
        sub = rng_sil.choice(len(l0), min(4000, len(l0)), replace=False)
        sil = (float(silhouette_score(fib0[sub], l0[sub]))
               if len(np.unique(l0[sub])) > 1 else float("nan"))
        # (a) grid-phase stability: the data moves
        stab = [adjusted_rand_score(l0, transfer(*run(mol, k, off, seed=a.seed,
                                                      **cfg)[:2], g0))
                for off in offsets]
        # (b) seed stability, as the control that shows why (a) is needed
        _, l_seed, _ = run(mol, k, None, seed=a.seed + 1, **cfg)
        seed_stab = adjusted_rand_score(l0, l_seed)
        rows.append(dict(design=a.design, k_true=a.k_true, k=k,
                         stability=float(np.mean(stab)),
                         stability_sd=float(np.std(stab)),
                         seed_stability=seed_stab,
                         silhouette=sil,
                         ari_vs_truth=adjusted_rand_score(truth, l0)))
        r = rows[-1]
        print(f"{k:3d} {r['stability']:10.3f} {r['silhouette']:11.3f} "
              f"{r['ari_vs_truth']:13.3f} {r['seed_stability']:11.3f}")

    d = pd.DataFrame(rows)
    out = (paths.results("synthetic")
           / f"k_selection_{a.design}_k{a.k_true}_s{a.seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\n-> {out}")

    print(f"\n  k_true                        = {a.k_true}")
    print(f"  argmax ARI against the truth  = {int(d.loc[d.ari_vs_truth.idxmax(), 'k'])}")
    print(f"  argmax grid-phase stability   = {int(d.loc[d.stability.idxmax(), 'k'])}")
    k_sil, k_upper = bracket(d)
    plateau = float(d[d.k <= 3].stability.mean())
    print(f"  coarse plateau (k<=3)         = {plateau:.3f}")
    print(f"  stability upper bound         = {k_upper}"
          f"   (largest k with stability >= 95% of the plateau)")
    print(f"  argmax silhouette             = {int(d.loc[d.silhouette.idxmax(), 'k'])}")
    print(f"  seed stability, min over k    = {d.seed_stability.min():.3f}"
          "   <- near 1 means the k-means seed is not a usable perturbation")
    lo, hi = min(k_sil, k_upper), max(k_sil, k_upper)
    inside = lo <= a.k_true <= hi
    print(f"\n  bracket [silhouette peak, stability bound] = {lo}-{hi}"
          f"   (k_true {a.k_true} {'inside' if inside else 'OUTSIDE'})")
    plateau_ari = d[d.ari_vs_truth >= 0.97 * d.ari_vs_truth.max()].k.tolist()
    print(f"  k within 3% of the best ARI against the truth: {plateau_ari}"
          "\n    -- where this spans several k, the generating count is not the"
          "\n       only defensible answer and a bracket is the honest output")


if __name__ == "__main__":
    main()
