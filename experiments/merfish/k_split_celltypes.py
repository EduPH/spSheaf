"""Which neuronal populations do the two BST halves at k = 9 contain?

The division of the annotated BST at k = 9 reproduces on the two posterior
MERFISH sections (0.19 and 0.24; k_split_analysis.py). This asks whether the
two halves differ in the neuronal clusters the dataset's authors assigned to
every cell (Moffitt et al. 2018, Dryad cell table: ``neuron_cluster_id``),
which is independent of the fibers the division was computed on.

For each slice: every annotated BST cell is assigned to the spSheaf cluster of
its nearest grid point at k = 9 (cell centroids are in the section frame of
the molecule table; the grid is in that frame shifted to its minimum). The
two clusters whose majority annotation is BST are the halves (larger share
first, as in Supplementary Table 4); cells landing on any other cluster are
dropped. Per neuronal cluster with at least MIN_CELLS cells across the two
halves: the count in each half, the odds ratio of being in the smaller half
against all other neurons, Fisher's exact test and Benjamini-Hochberg over
clusters. Cells are the sampling unit, so this is a test on independent
objects, unlike the grid-point test of the division itself; the spatial
autocorrelation of cell types still makes the effect sizes the result and the
q-values a guide.

Reproducibility: the per-cluster log odds on 0.19 against 0.24 (Pearson r over
clusters present on both), and the sign agreement among clusters significant
on either. The three anterior slices, where the BST is divided only across
hemispheres or not at all at k = 9, are the control: there the halves are
expected to differ in laterality, not composition.

    python -m experiments.merfish.k_split_celltypes

Writes data/results/merfish/k_split_celltypes.csv (one row per slice x
cluster) and k_split_celltypes_pairs.csv (the cross-slice correlations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import fisher_exact

from spsheaf import paths
from experiments.merfish import _cache
from experiments.merfish.k_split_analysis import children

SLICES = ["0.04", "0.09", "0.14", "0.19", "0.24"]
K = 9
DOMAIN = "BST"
MIN_CELLS = 20
RAW = "moffitt2018_export_full"


def cells():
    root = paths.raw("merfish") / RAW
    meta = pd.read_csv(root / "cell_metadata_add.csv.gz")
    xy = pd.read_csv(root / "spatial_coords.csv.gz")
    return xy.merge(meta, on="cell_id")


def bh(p):
    p = np.asarray(p, float); n = len(p)
    order = np.argsort(p); q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def main():
    allcells = cells()
    rows, summary = [], []
    for s in SLICES:
        z = np.load(paths.results("merfish") / f"k_split_MERFISH_{s}.npz", allow_pickle=False)
        grid, gt, names = z["grid_coords"], z["gt_int"], list(z["gt_names"])
        lab = z[f"labels_k{K}"]
        valid = gt >= 0
        d = names.index(DOMAIN)
        kids = children(lab, gt, valid).get(d, [])
        if len(kids) < 2:
            print(f"{s}: {DOMAIN} not divided at k = {K}"); continue
        (c1, n1, p1), (c2, n2, p2) = kids[:2]
        # cells of this slice, in the grid's frame
        mol = pd.read_parquet(_cache.parquet_path(f"MERFISH_{s}.h5ad"), columns=["x", "y"])
        x0, y0 = mol.x.min(), mol.y.min()
        c = allcells[(allcells.matched_h5ad_file == f"MERFISH_{s}.h5ad") & (allcells.domain == DOMAIN)].copy()
        c["gx"], c["gy"] = c.x - x0, c.y - y0
        dist, idx = cKDTree(grid).query(c[["gx", "gy"]].values, k=1)
        c["cluster"] = lab[idx]
        c["half"] = np.where(c.cluster == c1, "larger", np.where(c.cluster == c2, "smaller", "other"))
        n_other = int((c.half == "other").sum())
        c = c[c.half != "other"]
        offs = grid[(lab == c1) & (gt == d) & valid].mean(0) - grid[(lab == c2) & (gt == d) & valid].mean(0)
        hemi = abs(offs[0]) > 800
        neu = c[c.neuron_cluster_id.notna()]
        print(f"\n{s}: BST cells {len(c) + n_other}, on the two halves {len(c)} "
              f"(larger {int((c.half == 'larger').sum())}, smaller {int((c.half == 'smaller').sum())}, "
              f"elsewhere {n_other}); neurons {len(neu)}; half offset ({offs[0]:+.0f}, {offs[1]:+.0f}) um"
              f"{' hemispheric' if hemi else ''}; nearest-grid distance median {np.median(dist):.1f} um")
        classes = c.cell_class.value_counts()
        tab = []
        for cl, n in neu.neuron_cluster_id.value_counts().items():
            if n < MIN_CELLS:
                continue
            a = int(((neu.neuron_cluster_id == cl) & (neu.half == "smaller")).sum())
            b = int(((neu.neuron_cluster_id == cl) & (neu.half == "larger")).sum())
            cc = int(((neu.neuron_cluster_id != cl) & (neu.half == "smaller")).sum())
            dd = int(((neu.neuron_cluster_id != cl) & (neu.half == "larger")).sum())
            orr, p = fisher_exact([[a, b], [cc, dd]])
            lo = np.log((a + 0.5) * (dd + 0.5) / ((b + 0.5) * (cc + 0.5)))
            tab.append(dict(slice=s, cluster=cl, n=int(n), n_smaller=a, n_larger=b,
                            frac_in_smaller=a / n, odds_ratio=orr, log_odds=lo, p=p))
        t = pd.DataFrame(tab)
        t["q"] = bh(t.p)
        t["hemispheric"] = hemi
        t = t.sort_values("log_odds")
        rows.append(t)
        print(t[["cluster", "n", "n_smaller", "n_larger", "frac_in_smaller", "odds_ratio", "q"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        # non-neuronal classes, for the record
        nn = c[c.neuron_cluster_id.isna()]
        for cl, n in nn.cell_class.value_counts().items():
            if n < MIN_CELLS:
                continue
            a = int(((nn.cell_class == cl) & (nn.half == "smaller")).sum())
            rows.append(pd.DataFrame([dict(slice=s, cluster=f"class:{cl}", n=int(n), n_smaller=a,
                                           n_larger=int(n) - a, frac_in_smaller=a / n,
                                           odds_ratio=np.nan, log_odds=np.nan, p=np.nan, q=np.nan,
                                           hemispheric=hemi)]))
        summary.append(dict(slice=s, n_bst_cells=len(c) + n_other, n_on_halves=len(c), n_neurons=len(neu),
                            frac_neurons_smaller=float((neu.half == "smaller").mean()),
                            offset_x=offs[0], offset_y=offs[1], hemispheric=hemi))
    out = pd.concat(rows, ignore_index=True)
    res = paths.results("merfish")
    out.to_csv(res / "k_split_celltypes.csv", index=False)
    pd.DataFrame(summary).to_csv(res / "k_split_celltypes_summary.csv", index=False)

    # cross-slice reproducibility of the composition difference
    neu = out[~out.cluster.str.startswith("class:")]
    pairs = []
    sl = sorted(neu.slice.unique())
    for i in range(len(sl)):
        for j in range(i + 1, len(sl)):
            a = neu[neu.slice == sl[i]].set_index("cluster"); b = neu[neu.slice == sl[j]].set_index("cluster")
            common = a.index.intersection(b.index)
            if len(common) < 4:
                continue
            r = float(np.corrcoef(a.loc[common, "log_odds"], b.loc[common, "log_odds"])[0, 1])
            sig = [cl for cl in common if a.loc[cl, "q"] < 0.05 or b.loc[cl, "q"] < 0.05]
            agree = sum(np.sign(a.loc[cl, "log_odds"]) == np.sign(b.loc[cl, "log_odds"]) for cl in sig)
            pairs.append(dict(slice_a=sl[i], slice_b=sl[j], n_clusters=len(common), r=r,
                              n_sig=len(sig), n_sign_agree=int(agree),
                              hemispheric=bool(a.hemispheric.iloc[0] or b.hemispheric.iloc[0])))
    pr = pd.DataFrame(pairs)
    pr.to_csv(res / "k_split_celltypes_pairs.csv", index=False)
    print("\ncross-slice correlation of per-cluster log odds (smaller half vs larger):")
    print(pr.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def cell_counts():
    """Per-cell gene counts of every annotated cell, aggregated once from the
    molecule table (~20 M rows, ~20 s) into cache/merfish/cell_counts.parquet."""
    out = paths.cache("merfish") / "cell_counts.parquet"
    if out.exists():
        return pd.read_parquet(out)
    src = paths.raw("merfish") / RAW / "molecules_full_add.csv.gz"
    parts = []
    for ch in pd.read_csv(src, usecols=["gene", "cell_id", "matched_h5ad_file"],
                          chunksize=2_000_000, dtype=str):
        ch = ch[ch.matched_h5ad_file.notna()]
        parts.append(ch.groupby(["cell_id", "gene"]).size())
    cnt = pd.concat(parts).groupby(level=[0, 1]).sum().unstack(fill_value=0)
    cnt.to_parquet(out)
    return cnt


def markers():
    """Marker genes of the neuronal clusters that distinguish the halves, from
    the cells themselves (counts per 1,000, BST neurons of slices 0.19 and
    0.24), and the cell-level difference between the halves, to set against
    the grid-level genes of Fig. 4."""
    res = paths.results("merfish")
    ct = pd.read_csv(res / "k_split_celltypes.csv")
    ct = ct[~ct.cluster.str.startswith("class:")]
    cnt = cell_counts()
    allcells = cells().set_index("cell_id")
    rows = []
    for s in ["0.19", "0.24"]:
        z = np.load(res / f"k_split_MERFISH_{s}.npz", allow_pickle=False)
        grid, gt, names, lab = z["grid_coords"], z["gt_int"], list(z["gt_names"]), z[f"labels_k{K}"]
        d = names.index(DOMAIN)
        (c1, _, _), (c2, _, _) = children(lab, gt, gt >= 0)[d][:2]
        mol = pd.read_parquet(_cache.parquet_path(f"MERFISH_{s}.h5ad"), columns=["x", "y"])
        c = allcells[(allcells.matched_h5ad_file == f"MERFISH_{s}.h5ad") & (allcells.domain == DOMAIN)
                     & allcells.neuron_cluster_id.notna()].copy()
        _, idx = cKDTree(grid).query(np.c_[c.x - mol.x.min(), c.y - mol.y.min()], k=1)
        c["half"] = np.where(lab[idx] == c1, "larger", np.where(lab[idx] == c2, "smaller", "other"))
        c = c[c.half != "other"]
        X = cnt.loc[c.index]
        N = X.div(X.sum(1), axis=0) * 1000
        # the halves, at cell level
        diff = N[c.half == "smaller"].mean() - N[c.half == "larger"].mean()
        lfc = np.log2((N[c.half == "smaller"].mean() + 0.1) / (N[c.half == "larger"].mean() + 0.1))
        rows.append(dict(slice=s, what="smaller half vs larger (cells)",
                         up=" ".join(lfc.sort_values(ascending=False).index[:8]),
                         down=" ".join(lfc.sort_values().index[:8])))
        # each cluster against the other BST neurons
        for cl in ct[(ct.slice == float(s)) & (ct.q < 0.05)].cluster:
            m = c.neuron_cluster_id == cl
            l = np.log2((N[m].mean() + 0.1) / (N[~m].mean() + 0.1))
            rows.append(dict(slice=s, what=f"{cl} vs other BST neurons",
                             up=" ".join(l.sort_values(ascending=False).index[:8]),
                             down=" ".join(l.sort_values().index[:8])))
    t = pd.DataFrame(rows)
    t.to_csv(res / "k_split_celltypes_markers.csv", index=False)
    for _, r in t.iterrows():
        print(f"{r.slice}  {r.what:34s} up: {r.up}\n{'':40s} down: {r.down}")


if __name__ == "__main__":
    import sys
    markers() if "--markers" in sys.argv else main()
