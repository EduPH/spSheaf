"""Hyperparameter selection for osmFISH, matched to the MERFISH protocol.

    python -m experiments.osmfish.tune --stage diagnose   # run this first
    python -m experiments.osmfish.tune --stage rule
    python -m experiments.osmfish.tune --stage cv

**diagnose** — before tuning anything, ask whether the two domains the method
misses (Layer 2-3 lateral, and the 131-cell Layer 3-4 zone) are separable at
all from a 33-gene panel. A supervised classifier is given the ground-truth
labels and asked to tell the confusable pairs apart. If it cannot, no
unsupervised parameter setting will either, and that is the honest finding
rather than a tuning failure.

**rule** — apply the ratios fixed on MERFISH without consulting the osmFISH
annotation. MERFISH uses delta=15, h=25, sigma_s=60 um: h/delta = 1.67 and
sigma_s/delta = 4. The published osmFISH settings are delta=9.8, h=32.7,
sigma_s=19.6 um -- ratios 3.33 and 2.0 -- so they are not on the MERFISH rule;
they were inherited and nudged. The stalk dimension is re-derived by explained
variance instead of being fixed at 9 (d=9 of 33 genes is 27% of the feature
space, against 4.4% for MERFISH). Nothing here uses the labels, so this number
carries the strongest claim.

**cv** — the leave-one-out protocol the manuscript claims for MERFISH. osmFISH
is a single section, so the folds are contiguous spatial blocks rather than
slices; what matters is preserved, namely that a configuration is never scored
on the data used to select it. Contrast `run_osmFISH.py --sweep`, which reports
argmax ARI over 216 configurations scored on the annotation -- that number
cannot go in the paper.

Each config is run once over the whole section and then scored per block.
Running the pipeline on an isolated block would introduce edge effects at the
block boundary that have nothing to do with the hyperparameters.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from spsheaf import pipeline, fibers, io, paths
from experiments.common.cv import (spatial_blocks, cross_validate,
                                   summarise, score_labels)
from experiments.osmfish.run_osmFISH import (
    load_osmfish_molecules, load_cell_ground_truth,
    evaluate_at_cell_level, SLICE_NAME, PX_TO_UM,
)

# evaluate_at_cell_level returns:
#   (metrics, cell_predicted, cell_coords_aligned, cell_gt_int, gt_names)

MERFISH_H_OVER_DELTA = 25.0 / 15.0
MERFISH_SIGMA_OVER_DELTA = 60.0 / 15.0
GRID_PX = 150                       # published osmFISH grid spacing (9.8 um)

CONFUSABLE = [
    ("Layer 2-3 lateral", "Layer 2-3 medial"),
    ("Layer 3-4", "Layer 4"),
]


def _paths():
    root = paths.raw("osmfish")
    return (root / "mRNA_coords_raw_counting.hdf5",
            root / "osmfish_20260512084934.h5ad")


# ------------------------------------------------------------------ rule
def transferred_config(verbose=True):
    cfg = {
        "grid_spacing": GRID_PX,
        "bandwidth": int(round(GRID_PX * MERFISH_H_OVER_DELTA)),
        "smooth_radius": int(round(GRID_PX * MERFISH_SIGMA_OVER_DELTA)),
    }
    if verbose:
        print("  MERFISH ratios: h/delta = %.2f, sigma_s/delta = %.2f"
              % (MERFISH_H_OVER_DELTA, MERFISH_SIGMA_OVER_DELTA))
        for k, v in cfg.items():
            print(f"    {k:16s} = {v:4d} px ({v*PX_TO_UM:5.1f} um)")
        print("    published:        grid 150 (9.8), bandwidth 500 (32.7), "
              "smooth 300 (19.6)")
    return cfg


def stalk_dim_by_variance(mol, cfg, target=0.90, cand=range(3, 16), verbose=True):
    """Smallest d capturing `target` of local variance. Uses no labels."""
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    # io.select_slice is what adds the x_local / y_local columns
    # build_kernel_fibers defaults to. run_single_slice calls it internally;
    # anything calling build_kernel_fibers directly must do so too.
    mol_single = io.select_slice(mol, SLICE_NAME, verbose=False)
    grid, raw, _ = fibers.build_kernel_fibers(
        mol_single, grid_spacing=cfg["grid_spacing"],
        bandwidth=cfg["bandwidth"], verbose=False)
    fib = fibers.normalize_fibers(raw)
    nn = NearestNeighbors(n_neighbors=50).fit(grid)
    _, idx = nn.kneighbors(grid)

    rng = np.random.default_rng(0)
    sample = rng.choice(len(grid), size=min(400, len(grid)), replace=False)
    kmax = max(cand)
    cum = np.mean([np.cumsum(PCA(n_components=kmax).fit(
        fib[idx[i]]).explained_variance_ratio_) for i in sample], axis=0)
    d = next((k for k in cand if cum[k - 1] >= target), kmax)
    if verbose:
        for k in cand:
            print(f"    d={k:2d}  {cum[k-1]*100:5.1f}%" + (" <-" if k == d else ""))
        print(f"  -> d = {d} at {target:.0%} (published: 9)")
    return int(d)


def stage_rule(a):
    mol_p, h5_p = _paths()
    mol = load_osmfish_molecules(str(mol_p))
    print("\n[1] transferred spatial parameters")
    cfg = transferred_config()
    print("\n[2] stalk dimension by explained variance")
    d = stalk_dim_by_variance(mol, cfg, target=a.var_target)
    cfg["uniform_dim"] = d

    print("\n[3] running the transferred configuration")
    cell_coords, cell_gt = load_cell_ground_truth(str(h5_p))
    res = pipeline.run_single_slice(
        mol=mol, slice_name=SLICE_NAME, n_clusters=a.n_clusters,
        k_graph=6, k_nbrs=50, spatial_weight=0.5, boundary_alpha=0.5,
        cleanup=True, verbose=False, **cfg)
    m, *_ = evaluate_at_cell_level(res["grid_coords"], res["labels"],
                                   cell_coords, cell_gt, verbose=False)
    # (metrics, cell_predicted, cell_coords_aligned, cell_gt_int, gt_names)
    print(f"\n  transferred-rule config: ARI={m['ari']:.4f}  NMI={m['nmi']:.4f}")
    print(f"  published config:        ARI=0.5380  NMI=0.6390")
    print(f"  SSAM (best sweep):       ARI=0.5660  NMI=0.7210")

    out = paths.results("osmfish") / "tune_rule.json"
    out.write_text(json.dumps({"config": cfg, "metrics": m}, indent=2, default=float))
    print(f"\n-> {out}")


# ------------------------------------------------------------------ cv
def stage_cv(a):
    mol_p, h5_p = _paths()
    mol = load_osmfish_molecules(str(mol_p))
    cell_coords, cell_gt = load_cell_ground_truth(str(h5_p))
    # Base = the PUBLISHED spatial settings, not the transferred ones.
    # The manuscript's MERFISH protocol sets delta, h, d and k per dataset by
    # scale and density, and cross-validates only (sigma_s, alpha_b, alpha_s).
    # Matching that protocol therefore means holding delta/h/d at the
    # per-dataset values and CVing the three. --stage rule showed the
    # transferred h and d are worse here (ARI 0.428 vs 0.538), so building the
    # grid around them would search a basin we already know is poor.
    base = {"grid_spacing": GRID_PX, "bandwidth": 500, "uniform_dim": 9}
    if a.base == "transferred":
        base = transferred_config(verbose=False)
        base["uniform_dim"] = stalk_dim_by_variance(mol, base, a.var_target,
                                                    verbose=False)
    print(f"  base (held fixed): {base}")

    # The manuscript tunes exactly these three on MERFISH.
    configs = [dict(base, smooth_radius=sr, boundary_alpha=ab, spatial_weight=sw)
               for sr in (300, 450, 600)
               for ab in (0.25, 0.5, 1.0)
               for sw in (0.3, 0.5, 0.7)]
    print(f"  {len(configs)} configurations x {a.n_folds} spatial folds\n")

    # one pipeline run per config; score per fold afterwards
    per_config, fold_of_cell = [], None
    for i, cfg in enumerate(configs):
        res = pipeline.run_single_slice(
            mol=mol, slice_name=SLICE_NAME, n_clusters=a.n_clusters,
            k_graph=6, k_nbrs=50, cleanup=True, verbose=False, **cfg)
        _, pred_cells, coords_aligned, gt_int, _ = evaluate_at_cell_level(
            res["grid_coords"], res["labels"], cell_coords, cell_gt,
            verbose=False)
        per_config.append((np.asarray(gt_int), np.asarray(pred_cells)))
        if fold_of_cell is None:
            # blocks are defined on the ALIGNED frame, and identically for
            # every config, so folds are comparable across the grid
            fold_of_cell = spatial_blocks(coords_aligned, n_folds=a.n_folds,
                                          labels=gt_int, verbose=True)
            sizes = np.bincount(fold_of_cell)
            print(f"    spatial folds: {sizes.tolist()} cells\n")
        print(f"    [{i+1}/{len(configs)}] sr={cfg['smooth_radius']:4d} "
              f"ab={cfg['boundary_alpha']:.2f} sw={cfg['spatial_weight']:.1f}")

    def run_fn(cfg, fold):
        gt, pred = per_config[configs.index(cfg)]
        m = fold_of_cell == fold
        return score_labels(gt, pred, valid=m)

    print()
    cv = cross_validate(run_fn, configs, sorted(set(fold_of_cell)))
    s = summarise(cv)

    # A per-block ARI is NOT the same quantity as a whole-section ARI: a
    # spatial block contains fewer domains, so there is less to confuse and
    # the score runs high. Comparing cv_ari_mean against the published 0.538
    # or SSAM's 0.566 -- both whole-section -- would be apples to oranges.
    # So score the PUBLISHED config on the same blocks, giving a baseline in
    # the same units, and also report the CV-selected config on the whole
    # section, in the units the paper uses.
    pub_cfg = {"grid_spacing": GRID_PX, "bandwidth": 500, "uniform_dim": 9,
               "smooth_radius": 300, "boundary_alpha": 0.5,
               "spatial_weight": 0.5}
    pub_idx = next((i for i, c in enumerate(configs) if c == pub_cfg), None)
    if pub_idx is not None:
        gt_p, pr_p = per_config[pub_idx]
        per_block = [score_labels(gt_p, pr_p, valid=fold_of_cell == f)["ari"]
                     for f in sorted(set(fold_of_cell))]
        s["published_per_block_ari_mean"] = float(np.mean(per_block))

    res = pipeline.run_single_slice(
        mol=mol, slice_name=SLICE_NAME, n_clusters=a.n_clusters,
        k_graph=6, k_nbrs=50, cleanup=True, verbose=False, **s["final_config"])
    m_whole, *_ = evaluate_at_cell_level(
        res["grid_coords"], res["labels"], cell_coords, cell_gt, verbose=False)
    s["final_config_whole_section_ari"] = float(m_whole["ari"])
    s["final_config_whole_section_nmi"] = float(m_whole["nmi"])

    print(f"\n  PER-BLOCK (not comparable to whole-section numbers)")
    print(f"    cross-validated ARI  {s['cv_ari_mean']:.4f} +- {s['cv_ari_sd']:.4f}"
          f"   (s.e. {s['cv_ari_sd']/np.sqrt(len(cv['folds'])):.4f})")
    print(f"    cross-validated NMI  {s['cv_nmi_mean']:.4f} +- {s['cv_nmi_sd']:.4f}")
    if "published_per_block_ari_mean" in s:
        d = s["cv_ari_mean"] - s["published_per_block_ari_mean"]
        print(f"    published config, same blocks: "
              f"{s['published_per_block_ari_mean']:.4f}  (CV gains {d:+.4f})")

    print(f"\n  WHOLE SECTION (comparable to 0.538 published / 0.566 SSAM)")
    print(f"    CV-selected config   ARI {s['final_config_whole_section_ari']:.4f}"
          f"  NMI {s['final_config_whole_section_nmi']:.4f}")
    print(f"    config: {s['final_config']}")

    print(f"\n  selection stable across folds: {s['selection_stable_across_folds']}")
    if not s["selection_stable_across_folds"]:
        print("    A different config wins on each split, so the grid is flat and")
        print("    the 'best' setting is largely noise. Report the cross-validated")
        print("    mean with its spread, not the winning config's own score.")

    out = paths.results("osmfish") / "tune_cv.json"
    out.write_text(json.dumps(s, indent=2, default=float))
    print(f"\n-> {out}")


# ------------------------------------------------------------------ diagnose
def stage_diagnose(a):
    import anndata as ad
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    _, h5_p = _paths()
    adata = ad.read_h5ad(str(h5_p))
    X = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)
    key = next((k for k in ("ground_truth", "Region", "region", "domain")
                if k in adata.obs), None)
    if key is None:
        raise SystemExit(f"no region column in {list(adata.obs.columns)}")
    y = adata.obs[key].astype(str).to_numpy()

    print(f"  {X.shape[0]} cells x {X.shape[1]} genes; regions:")
    for r, n in zip(*np.unique(y, return_counts=True)):
        print(f"    {r:32s} {n:5d}")

    print("\n  Can a SUPERVISED classifier separate the confusable pairs?")
    print("  Balanced 5-fold CV accuracy. 0.5 = the panel cannot tell them")
    print("  apart at all; 1.0 = trivially separable. Read the interval, not")
    print("  a threshold -- these land in a band where neither verdict is clean.")
    for A, B in CONFUSABLE:
        m = np.isin(y, [A, B])
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            print(f"    {A} vs {B}: labels not present as written -- check names")
            continue
        acc = cross_val_score(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            X[m], y[m], cv=5, scoring="balanced_accuracy")
        n_a, n_b = (y[m] == A).sum(), (y[m] == B).sum()
        lo, hi = acc.mean() - acc.std(), acc.mean() + acc.std()
        print(f"    {A} ({n_a}) vs {B} ({n_b}): "
              f"{acc.mean():.3f} +- {acc.std():.3f}  [{lo:.2f}, {hi:.2f}]")

    # ---- what does this annotation actually encode? ----------------------
    # Four attempts at this, three of them wrong, and the errors are worth
    # recording because each produced a plausible-looking number:
    #
    #   v1  expression-only supervised classifier, reported as a "ceiling".
    #       spSheaf scored 164% of it. A ceiling you can exceed is not one:
    #       position carries more layer information than a 33-gene panel.
    #   v2  added coordinates, kept RANDOM 5-fold CV. Coordinates-only hit
    #       ARI 0.930 -- spatial leakage, not skill. With random folds a
    #       held-out cell has training cells microns away, so the model does
    #       nearest-neighbour lookup on position.
    #   v3  blocked contiguously along the longest axis, which on a layered
    #       cortex runs ACROSS the layers. Whole classes vanished from
    #       training and recall hit exactly 0.00 for three domains.
    #   v4  chose the blocking axis to preserve the label mix. Better, but
    #       Internal Capsule (106 cells) and Ventricle (129) are spatially
    #       COMPACT: any contiguous block contains each of them entirely, so
    #       no axis distributes them across folds. Contiguous spatial CV
    #       cannot preserve class balance for compact rare classes. This is
    #       not fixable by a better axis.
    #
    # The mistake underneath all four was reaching for a supervised ceiling at
    # all. spSheaf, SSAM and Points2Regions are UNSUPERVISED partitions of
    # these cells into 11 groups. The directly comparable reference is another
    # unsupervised partition -- so cluster on coordinates alone, on expression
    # alone, and on both, and score all of them the same way. No labels are
    # used to fit anything, so there is no CV, no leakage, and no class-balance
    # problem, and every number sits in the same units as the published ones.
    print("\n  WHAT DOES THE osmFISH ANNOTATION ENCODE?")
    print("  Unsupervised baselines: k-means into 11 groups, scored against the")
    print("  annotation exactly as the real methods are. No labels are used to")
    print("  fit these, so they are directly comparable to spSheaf and SSAM.")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    xy = np.asarray(adata.obsm["spatial"], float)
    xy_z = StandardScaler().fit_transform(xy)
    Xz = StandardScaler().fit_transform(X)
    k = len(np.unique(y))

    variants = {
        "coordinates only        ": xy_z,
        "expression only         ": Xz,
        "expression + coordinates": np.hstack([Xz, xy_z]),
    }
    out = {}
    for name, feat in variants.items():
        aris, nmis, preds = [], [], None
        for seed in range(5):
            lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(feat)
            aris.append(adjusted_rand_score(y, lab))
            nmis.append(normalized_mutual_info_score(y, lab))
            if seed == 0:
                preds = lab
        out[name.strip()] = (float(np.mean(aris)), float(np.mean(nmis)), preds)
        print(f"    {name}  ARI={np.mean(aris):.3f}+-{np.std(aris):.3f}  "
              f"NMI={np.mean(nmis):.3f}+-{np.std(nmis):.3f}")

    print(f"\n    spSheaf (published)        ARI=0.538  NMI=0.639")
    print(f"    SSAM (best sweep)          ARI=0.566  NMI=0.721")

    xy_ari, xy_nmi = out["coordinates only"][0], out["coordinates only"][1]
    ex_ari, ex_nmi = out["expression only"][0], out["expression only"][1]
    both_ari = out["expression + coordinates"][0]
    best_naive = max(xy_ari, ex_ari, both_ari)

    print(f"\n    Best naive baseline: ARI {best_naive:.3f}. "
          f"spSheaf: 0.538 ({0.538/max(best_naive,1e-9):.1f}x).")
    print(f"    Naive concatenation of expression and coordinates ({both_ari:.3f})")
    print(f"    does no better than coordinates alone ({xy_ari:.3f}) -- combining")
    print(f"    the two signals is not trivial, which is what the method is for.")

    # Only call it dominance if the margin is real; 0.22 vs 0.19 is not.
    margin = abs(xy_ari - ex_ari)
    if margin < 0.05:
        print(f"\n    Geometry and expression are comparably weak on their own")
        print(f"    (ARI {xy_ari:.3f} vs {ex_ari:.3f}, margin {margin:.3f}). On NMI the")
        print(f"    gap is wider ({xy_nmi:.3f} vs {ex_nmi:.3f}), so position does carry")
        print(f"    more of the annotation's structure, but neither signal alone")
        print(f"    gets close to the published numbers.")
    elif xy_ari > ex_ari:
        print(f"\n    Position carries more of this annotation than the 33-gene")
        print(f"    panel does (ARI {xy_ari:.3f} vs {ex_ari:.3f}).")
    else:
        print(f"\n    Expression carries more than position "
              f"(ARI {ex_ari:.3f} vs {xy_ari:.3f}).")

    print(f"\n    These are FLOORS, not ceilings: k-means on raw coordinates")
    print(f"    makes compact blobs, not cortical bands, so a smarter geometric")
    print(f"    baseline would score higher. Report them as 'naive baselines',")
    print(f"    not as the best a geometry-only method could do.")

    print("\n    per-domain agreement, best baseline (Hungarian-matched):")
    from scipy.optimize import linear_sum_assignment
    best_name = max(out, key=lambda n: out[n][0])
    pred = out[best_name][2]
    classes = np.unique(y)
    C = np.zeros((len(classes), k), int)
    for i, c in enumerate(classes):
        for j in range(k):
            C[i, j] = np.sum((y == c) & (pred == j))
    ri, ci = linear_sum_assignment(-C)
    mapping = dict(zip(ci, ri))
    matched = np.array([classes[mapping[p]] if p in mapping else "" for p in pred])
    print(f"      (baseline: {best_name})")
    for c in classes:
        m = y == c
        print(f"      {c:32s} {(matched[m] == c).mean():.2f}  (n={m.sum()})")


def stage_attribute(a):
    """Which of the three transferred changes costs the accuracy?

    --stage rule changed bandwidth (500 -> 250 px), smoothing (300 -> 600) and
    the stalk dimension (9 -> 4) simultaneously, and lost 0.110 ARI. That is a
    confounded comparison: it says the transfer fails but not why.

    This runs the eight corners of {bandwidth} x {smoothing} x {d} so the drop
    can be attributed. It is an ABLATION, not a search: the point is to explain
    a result, and the best-scoring corner must NOT be reported as the osmFISH
    number, because picking it would be selection on the annotation -- exactly
    what --stage cv exists to avoid.
    """
    mol_p, h5_p = _paths()
    mol = load_osmfish_molecules(str(mol_p))
    cell_coords, cell_gt = load_cell_ground_truth(str(h5_p))

    levels = {"bandwidth": (500, 250), "smooth_radius": (300, 600),
              "uniform_dim": (9, 4)}
    print("  published value first, transferred value second\n")
    print(f"  {'bw':>5} {'sig':>5} {'d':>3} | {'ARI':>7} {'NMI':>7}   note")
    print("  " + "-" * 46)

    rows = []
    for bw in levels["bandwidth"]:
        for sr in levels["smooth_radius"]:
            for ud in levels["uniform_dim"]:
                res = pipeline.run_single_slice(
                    mol=mol, slice_name=SLICE_NAME, n_clusters=a.n_clusters,
                    grid_spacing=GRID_PX, bandwidth=bw, smooth_radius=sr,
                    uniform_dim=ud, k_graph=6, k_nbrs=50, spatial_weight=0.5,
                    boundary_alpha=0.5, cleanup=True, verbose=False)
                m, *_ = evaluate_at_cell_level(
                    res["grid_coords"], res["labels"], cell_coords, cell_gt,
                    verbose=False)
                note = ""
                if (bw, sr, ud) == (500, 300, 9):
                    note = "<- published"
                elif (bw, sr, ud) == (250, 600, 4):
                    note = "<- transferred"
                rows.append(dict(bandwidth=bw, smooth_radius=sr,
                                 uniform_dim=ud, **m))
                print(f"  {bw:5d} {sr:5d} {ud:3d} | {m['ari']:7.4f} "
                      f"{m['nmi']:7.4f}   {note}")

    df = pd.DataFrame(rows)
    print("\n  main effect on ARI (mean over the other two factors):")
    for f, (pub, tr) in levels.items():
        a_pub = df[df[f] == pub]["ari"].mean()
        a_tr = df[df[f] == tr]["ari"].mean()
        print(f"    {f:14s} {pub:>4} -> {tr:>4}:  {a_tr - a_pub:+.4f}")
    print("\n  The factor with the largest negative effect is what the")
    print("  MERFISH transfer got wrong for this dataset. Report that, not")
    print("  the best corner -- selecting a corner on ARI is annotation-informed")
    print("  tuning, which is the practice the review memo objects to.")

    out = paths.results("osmfish") / "tune_attribute.csv"
    df.to_csv(out, index=False)
    print(f"\n-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["rule", "cv", "diagnose", "attribute"])
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-clusters", type=int, default=11)
    ap.add_argument("--var-target", type=float, default=0.90)
    ap.add_argument("--base", choices=["published", "transferred"],
                    default="published",
                    help="which delta/h/d to hold fixed while CVing "
                         "(sigma_s, alpha_b, alpha_s)")
    a = ap.parse_args()
    {"rule": stage_rule, "cv": stage_cv, "diagnose": stage_diagnose,
     "attribute": stage_attribute}[a.stage](a)


if __name__ == "__main__":
    main()
