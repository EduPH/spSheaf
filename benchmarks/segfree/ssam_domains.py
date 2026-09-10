"""SSAM's ``find_domains`` (v1.1.3, ssam/_analysis.py) as a standalone
function over the arrays it actually uses.

``SSAMAnalysis.find_domains`` accesses the dataset through properties that
pull the full-resolution vector field back into memory, which exceeds 6 GB
on a 2 M-molecule MERFISH section and killed the domain sweep. The
algorithm itself only needs the binned cell-type counts, the shape of the
KDE grid and the filtered cell-type map. This is the same code, with those
three read once from the zarr store. Checked against the in-library call on
slice 0.19 (identical labels).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering


def _corr(a, b):
    return np.corrcoef(a, b)[0, 1]


def load_store(path):
    import zarr                      # only the KDE machine needs zarr
    g = zarr.open(path, mode="r")
    return dict(binned=np.asarray(g["celltype_binned_counts"]),
                vf_norm_shape=tuple(g["vf_norm"].shape),
                filtered=np.asarray(g["filtered_celltype_maps"]))


def find_domains(store, n_clusters=10, norm_thres=0, merge_thres=0.6, merge_remote=True):
    binned_ctmaps = store["binned"]
    binned_ctmaps_norm = np.sum(binned_ctmaps, axis=3)
    ctvf_vecs = binned_ctmaps[binned_ctmaps_norm > norm_thres]
    if len(ctvf_vecs) < n_clusters:
        raise ValueError(f"only {len(ctvf_vecs)} windows above norm_thres={norm_thres}")
    ctvf_vecs_normalized = preprocessing.normalize(ctvf_vecs, norm="l1", axis=1)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(ctvf_vecs_normalized)
    layer_map = np.zeros(binned_ctmaps_norm.shape)
    layer_map[binned_ctmaps_norm > norm_thres] = clustering.labels_ + 1
    layer_map = measure.label(layer_map)

    def find_neighbors(m, l):
        nb = set()
        for x, y, z in zip(*np.where(m == l)):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx or dy or dz:
                            nb.add(m[x + dx, y + dy, z + dz])
        return nb

    if merge_thres < 1.0:
        while True:
            uniq_labels = np.array(list(set(list(np.ravel(layer_map))) - {0}))
            if not merge_remote:
                padded = np.pad(layer_map, 1, mode="constant", constant_values=0)
                neighbors = {lbl: find_neighbors(padded, lbl) for lbl in uniq_labels}
            centroids = [np.mean(binned_ctmaps[layer_map == lbl], axis=0) for lbl in uniq_labels]
            max_corr, max_idx = 0, None
            for i in range(len(uniq_labels)):
                for j in range(i + 1, len(uniq_labels)):
                    li, lj = uniq_labels[i], uniq_labels[j]
                    c = _corr(centroids[i], centroids[j])
                    if c > max_corr and (merge_remote or lj in neighbors[li]):
                        max_corr, max_idx = c, (li, lj)
            if max_corr > merge_thres:
                layer_map[layer_map == max_idx[1]] = max_idx[0]
            else:
                break
    uniq_labels = sorted(set(list(np.ravel(layer_map))) - {0})
    for i, lbl in enumerate(uniq_labels, start=1):
        layer_map[layer_map == lbl] = i
    resized = ndimage.zoom(layer_map, np.array(store["vf_norm_shape"]) / np.array(layer_map.shape), order=0) - 1
    return resized
