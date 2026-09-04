"""Turn FICTURE's pixel-level factor output into domain labels and score it.

Run as: python eval_ficture.py <ficture_out_dir> <ficture_input_dir> [--tag NAME]

Why an aggregation step is needed. FICTURE infers a soft factor mixture per
pixel, not a hard partition into tissue domains. The manuscript currently avoids
the comparison on the grounds that the output granularity differs; the reviewer's
point is that the honest response is to build a domain-level readout from the
factor map rather than to skip the comparison. This script does that, with three
aggregation strategies so the result does not hinge on one arbitrary choice:

  argmax   Each cell takes the modal top-1 factor of its nearby pixels. This is
           FICTURE's own factor map read directly as a partition. It is the
           natural readout when the factor count is set to the domain count, and
           is the least favourable to FICTURE when factors are cell types rather
           than domains.

  kmeans   Each cell gets the mean factor-probability vector of its nearby
           pixels; those vectors are clustered by k-means into k domains. This is
           the "collapse the factor map into k regions" construction the reviewer
           proposed, and it lets FICTURE run with more factors than domains
           (factors as cell types, domains as their spatial composition) which is
           how FICTURE is used in its own paper.

  spatial  As kmeans, but with standardised spatial coordinates appended to the
           feature vector at the same weight the sheaf pipeline uses (alpha_s =
           0.5). Without this, k-means on composition alone is scored against a
           spatially-regularised competitor, which understates FICTURE.

For each strategy we report the score at the ground-truth k. Because FICTURE's
factor count is a free parameter, several settings are run and the best is
reported, which is a bound in FICTURE's favour (its number of factors is chosen
against the ground truth; the sheaf parameters are cross-validated).
"""
import argparse
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

from benchmarks.segfree import common as bc


def read_pixel_output(path):
    """Read a FICTURE *.pixel.sorted.tsv.gz into (xy_um, top-k factors, top-k probs).

    The file carries three '##' metadata lines then a '#' header. Coordinates are
    stored as integers scaled by SCALE and offset by OFFSET_X/OFFSET_Y, both
    declared in the metadata, so the physical position is X/SCALE + OFFSET_X.
    """
    meta = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                for kv in line.strip("#\n").split(";"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        meta[k] = v
            elif line.startswith("#"):
                cols = line.strip("#\n").split("\t")
                break
    df = pd.read_csv(path, sep="\t", comment="#", names=cols, header=None)
    scale = float(meta.get("SCALE", 100))
    ox, oy = float(meta.get("OFFSET_X", 0)), float(meta.get("OFFSET_Y", 0))
    xy = np.c_[df["X"].values / scale + ox, df["Y"].values / scale + oy]
    kcols = [c for c in cols if c.startswith("K")]
    pcols = [c for c in cols if c.startswith("P")]
    K = int(meta.get("K", df[kcols].values.max() + 1))
    return xy, df[kcols].values.astype(int), df[pcols].values.astype(float), K, meta


def cell_factor_profiles(cell_xy, pix_xy, kmat, pmat, K, radius=30.0, min_pix=5):
    """Mean factor-probability vector per cell over pixels within `radius` um.

    Returns (profiles, n_pixels_per_cell). Cells with too few nearby pixels get a
    zero profile and are filled from their nearest labelled neighbour later, the
    same convention used for the other methods' background pixels.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(pix_xy)
    prof = np.zeros((len(cell_xy), K))
    npix = np.zeros(len(cell_xy), dtype=int)
    for i, ids in enumerate(tree.query_ball_point(cell_xy, radius)):
        if not ids:
            continue
        npix[i] = len(ids)
        ids = np.asarray(ids)
        for j in range(kmat.shape[1]):
            np.add.at(prof[i], kmat[ids, j], pmat[ids, j])
    s = prof.sum(1, keepdims=True)
    s[s == 0] = 1
    return prof / s, npix


def argmax_labels(cell_xy, pix_xy, kmat, radius=30.0):
    """Modal top-1 factor among pixels near each cell."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pix_xy)
    out = np.full(len(cell_xy), -1)
    for i, ids in enumerate(tree.query_ball_point(cell_xy, radius)):
        if not ids:
            continue
        v, c = np.unique(kmat[np.asarray(ids), 0], return_counts=True)
        out[i] = v[np.argmax(c)]
    return out


def cluster_profiles(prof, k, cell_xy=None, alpha_s=0.5, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    F = StandardScaler().fit_transform(prof)
    if cell_xy is not None:
        S = StandardScaler().fit_transform(np.asarray(cell_xy, dtype=float))
        F = np.c_[F, alpha_s * S]
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(F)


def evaluate_run(out_dir, in_dir, tag=None, radius=30.0, seeds=(0, 1, 2, 3, 4)):
    with open(f"{in_dir}/offset.json") as f:
        meta_in = json.load(f)
    cell_xy = np.load(f"{in_dir}/cell_xy_um.npy")
    gt = pd.read_csv(f"{in_dir}/ground_truth.csv")["gt"].astype(str).values
    k_gt = int(meta_in["n_gt_domains"])

    hits = sorted(glob.glob(f"{out_dir}/analysis/*/*.pixel.sorted.tsv.gz"))
    if not hits:
        raise SystemExit(f"no pixel output under {out_dir}")
    rows = []
    for path in hits:
        model_id = os.path.basename(os.path.dirname(path))
        pix_xy, kmat, pmat, K, meta = read_pixel_output(path)
        prof, npix = cell_factor_profiles(cell_xy, pix_xy, kmat, pmat, K, radius=radius)
        frac_bare = float(np.mean(npix == 0))

        lab = bc.fill_background(argmax_labels(cell_xy, pix_xy, kmat, radius=radius),
                                 cell_xy, bg=-1)
        m = bc.all_metrics(gt, lab, cell_xy)
        m.update(strategy="argmax", model=model_id, n_factors=K, seed=None,
                 frac_cells_no_pixels=frac_bare, n_pixels=int(len(pix_xy)))
        rows.append(m)

        for strat, xy in (("kmeans", None), ("spatial", cell_xy)):
            for sd in seeds:
                lab = cluster_profiles(prof, k_gt, cell_xy=xy, seed=sd)
                m = bc.all_metrics(gt, lab, cell_xy)
                m.update(strategy=strat, model=model_id, n_factors=K, seed=sd,
                         frac_cells_no_pixels=frac_bare, n_pixels=int(len(pix_xy)))
                rows.append(m)
        print(f"[ficture {tag or out_dir} {model_id}] K={K} pixels={len(pix_xy):,} "
              f"argmax ARI={rows[-1 - 2 * len(seeds)]['ARI']:.3f} "
              f"| kmeans ARI={np.mean([r['ARI'] for r in rows if r['strategy']=='kmeans' and r['model']==model_id]):.3f} "
              f"| spatial ARI={np.mean([r['ARI'] for r in rows if r['strategy']=='spatial' and r['model']==model_id]):.3f}",
              flush=True)

    df = pd.DataFrame(rows)
    df["tag"] = tag or os.path.basename(out_dir)
    return df, gt, cell_xy


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("in_dir")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--radius", type=float, default=30.0)
    a = ap.parse_args()
    os.makedirs("results/ficture", exist_ok=True)
    df, gt, cell_xy = evaluate_run(a.out_dir, a.in_dir, tag=a.tag, radius=a.radius)
    name = a.tag or os.path.basename(a.out_dir)
    df.to_csv(f"results/ficture/metrics_{name}.csv", index=False)
    best = df.loc[df.ARI.idxmax()]
    print(f"\nbest: {best.strategy} model={best.model} ARI={best.ARI:.3f} NMI={best.NMI:.3f}")
