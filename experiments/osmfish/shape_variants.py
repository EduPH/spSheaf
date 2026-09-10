"""Clustering variants that have no compactness prior, on osmFISH.

The osmFISH analysis (manuscript 2.8, Extended Data Fig. 2) attributes the two
missed domains partly to the compactness bias of k-means in
``[features | alpha_s * (x, y)]``: raising alpha_s recovers Layer 2-3 lateral
and destroys Pia Layer 1 (aspect ratio 10). This script keeps steps 1-6 of the
pipeline byte-identical (fibers, graph, stalks, disagreement, smoothed field)
and swaps only step 7, the partition, for methods whose clusters follow the
*spatial graph* rather than Euclidean balls in coordinate space:

  kmeans      the published step 7 (reference)
  ward        connectivity-constrained Ward on the boundary-weighted PCA
              features, connectivity = the spatial k-NN graph; no coordinates
  ward_xy     the same with alpha_s-scaled coordinates appended
  ward_cut    ward, with graph edges removed where the smoothed field is in
              its top q (the boundary field as a cut, not a weight)
  spectral    normalised spectral clustering of the spatial graph with edge
              affinity exp(-|dz|^2 / 2 sigma^2) * exp(-alpha_b * mean B / med B)

Every variant is followed by the published spatial cleanup and scored at cell
level with the published evaluation.

    python -m experiments.osmfish.shape_variants

Result (5 Sep 2026). Connectivity-constrained Ward does what the shape
analysis predicts on osmFISH -- Pia Layer 1 and Layer 4 are recovered at
0.9+ together -- and its best in-sample setting (alpha_b 0.25, 20 PCs,
alpha_s 0.25) reaches ARI 0.555 / NMI 0.673 against 0.538 / 0.639 for k-means,
still below SSAM's 0.566 / 0.721. But the landscape is jagged (0.45-0.56 over
neighbouring settings) and, decisively, the same step-7 swap collapses on
MERFISH: over the five slices, grid-level ARI 0.28-0.32 for every Ward
setting against 0.575 for k-means (``experiments/merfish`` cache, seed 42).
Over-clustering with adjacency-constrained merging (``overcluster_factor 2``)
gives 0.343 on osmFISH. Spectral coordinates of the boundary-weighted spatial
graph in place of (x, y) score 0.23-0.49. None of these is adopted: the
published step 7 stays, and the osmFISH shortfall stays attributed to the two
degenerate domains rather than to the partition method.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from spsheaf import clustering, graph, paths, pipeline
from experiments.osmfish import run_osmFISH as osm

CACHE = Path(paths.cache("osmfish")) / "shape_variants_state.pkl"
K = 11


def state():
    if CACHE.exists():
        return pickle.load(open(CACHE, "rb"))
    mol = osm.load_osmfish_molecules(str(paths.raw("osmfish") / "mRNA_coords_raw_counting.hdf5"))
    cells, gt = osm.load_cell_ground_truth(str(paths.raw("osmfish") / "osmfish_20260512084934.h5ad"))
    r = pipeline.run_single_slice(
        mol=mol, slice_name=osm.SLICE_NAME, n_clusters=K, grid_spacing=150,
        bandwidth=500, smooth_radius=300, spatial_weight=0.5, k_graph=6,
        k_nbrs=50, uniform_dim=9, boundary_alpha=0.5, cleanup=True,
        eval_at_cell_level=False, verbose=False)
    edges = graph.build_spatial_graph(r["grid_coords"], k=6, verbose=False)
    st = dict(grid=r["grid_coords"], fib=r["fibers_norm"], bsm=r["boundary_smooth"],
              edges=np.asarray(edges), labels_ref=r["labels"], cells=cells, gt=gt)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(st, open(CACHE, "wb"))
    return st


def features(st, alpha_b=0.5, n_pca=10):
    z = StandardScaler().fit_transform(PCA(n_components=n_pca).fit_transform(st["fib"]))
    w = np.exp(-alpha_b * st["bsm"] / (np.median(st["bsm"]) + 1e-10))
    xy = StandardScaler().fit_transform(st["grid"])
    return z * w[:, None], xy


def adjacency(st, keep=None):
    e = st["edges"] if keep is None else st["edges"][keep]
    n = len(st["grid"])
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    a = a + a.T
    a.data[:] = 1.0
    return a.tocsr()


def variant(name, st, alpha_s=0.5, q_cut=0.10, sigma_f=None):
    zw, xy = features(st)
    n = len(st["grid"])
    if name == "kmeans":
        X = np.hstack([zw, xy * alpha_s])
        lab = KMeans(K, n_init=20, random_state=42).fit_predict(X)
    elif name in ("ward", "ward_xy", "ward_cut"):
        X = zw if name == "ward" else np.hstack([zw, xy * alpha_s]) if name == "ward_xy" else zw
        keep = None
        if name == "ward_cut":
            e = st["edges"]
            edge_b = 0.5 * (st["bsm"][e[:, 0]] + st["bsm"][e[:, 1]])
            keep = edge_b < np.quantile(edge_b, 1 - q_cut)
        A = adjacency(st, keep)
        lab = AgglomerativeClustering(n_clusters=K, linkage="ward", connectivity=A).fit_predict(X)
    elif name == "spectral":
        e = st["edges"]
        dz = np.linalg.norm(zw[e[:, 0]] - zw[e[:, 1]], axis=1)
        s = sigma_f or np.median(dz)
        edge_b = 0.5 * (st["bsm"][e[:, 0]] + st["bsm"][e[:, 1]])
        w = np.exp(-dz ** 2 / (2 * s ** 2)) * np.exp(-0.5 * edge_b / np.median(st["bsm"]))
        A = sparse.coo_matrix((w, (e[:, 0], e[:, 1])), shape=(n, n)); A = (A + A.T).tocsr()
        lab = SpectralClustering(K, affinity="precomputed", random_state=42,
                                 assign_labels="kmeans", n_init=10).fit_predict(A)
    else:
        raise ValueError(name)
    lab, _ = clustering.spatial_cleanup(lab, [tuple(x) for x in st["edges"]],
                                        grid_coords=st["grid"], verbose=False)
    return lab


def score(st, lab):
    m, pred, cxy, gti, names = osm.evaluate_at_cell_level(
        st["grid"], lab, st["cells"], st["gt"], verbose=False)
    # per-domain recall after Hungarian matching
    from spsheaf import evaluation
    valid = gti >= 0
    matched = evaluation.hungarian_match_labels(pred, gti, valid)
    rec = {nm: float(np.mean(matched[valid & (gti == i)] == i)) for i, nm in enumerate(names)}
    return m, rec


def main():
    st = state()
    names = sys.argv[1:] or ["kmeans", "ward", "ward_xy", "ward_cut", "spectral"]
    rows = []
    for nm in names:
        t = time.time()
        lab = variant(nm, st)
        m, rec = score(st, lab)
        rows.append((nm, m["ari"], m["nmi"], rec, time.time() - t))
        print(f"{nm:10s} ARI {m['ari']:.3f}  NMI {m['nmi']:.3f}  ({time.time()-t:.0f}s)")
        print("   " + "  ".join(f"{k[:14]}={v:.2f}" for k, v in rec.items()))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
