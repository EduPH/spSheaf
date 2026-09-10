"""Naive baselines on a SpaceTx dataset, same protocol as manuscript 2.8:
k-means into the annotated number of classes on (i) standardized cell
centroids, (ii) the log-normalized cell-by-gene matrix, (iii) both
concatenated; 20 initializations each, nothing fit on labels.

    python -m experiments.spacetx.baselines MERFISH_Allen_Prototype --cell-id sample_name --cell-x rotated_x --cell-y rotated_y --label layer
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from spsheaf import paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset"); p.add_argument("--cell-id", default="sample_name")
    p.add_argument("--cell-x", default="rotated_x"); p.add_argument("--cell-y", default="rotated_y")
    p.add_argument("--label", default="layer"); p.add_argument("--n-seeds", type=int, default=20)
    a = p.parse_args()
    root = paths.raw("spacetx") / a.dataset
    cells = pd.read_csv(root / "s3_mapped_cell_table.csv")
    cbg = pd.read_csv(root / "s3_cell_by_gene.csv", index_col=0)
    if "gene_name" in cbg.index:
        cbg.columns = cbg.loc["gene_name"].astype(str).values
        cbg = cbg.drop(index="gene_name"); cbg.index = cbg.index.astype(str)
        cbg = cbg.apply(pd.to_numeric, errors="coerce").fillna(0)
    X = cbg.reindex(cells[a.cell_id].astype(str)).fillna(0).values.astype(float)
    X = np.log1p(1e4 * X / np.maximum(X.sum(1, keepdims=True), 1))
    Z = StandardScaler().fit_transform(PCA(n_components=min(10, X.shape[1])).fit_transform(X))
    S = StandardScaler().fit_transform(cells[[a.cell_x, a.cell_y]].values)
    gt = cells[a.label].values; ok = pd.notna(gt); k = len(set(gt[ok]))
    for name, F in (("coordinates only", S), ("expression only", Z), ("concatenated", np.hstack([Z, S]))):
        ari, nmi = [], []
        for seed in range(a.n_seeds):
            lab = KMeans(k, n_init=10, random_state=seed).fit_predict(F)
            ari.append(adjusted_rand_score(gt[ok], lab[ok])); nmi.append(normalized_mutual_info_score(gt[ok], lab[ok]))
        print(f"{name:18s} ARI {np.mean(ari):.3f} ± {np.std(ari):.3f}   NMI {np.mean(nmi):.3f} ± {np.std(nmi):.3f}")


if __name__ == "__main__":
    main()
