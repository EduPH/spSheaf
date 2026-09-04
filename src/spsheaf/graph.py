"""
Spatial k-nearest neighbor graph construction.

The spatial graph G = (V, E) is built on the fiber grid coordinates
using symmetric k-NN: an edge (u, v) exists if u is among v's k
nearest neighbors OR v is among u's k nearest neighbors.

With k=6, the resulting graph has ~50,000 edges per slice and mean
vertex degree ~7 (slightly above k due to symmetrization).
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_spatial_graph(grid_coords, k=6, verbose=True):
    """
    Build a symmetric k-NN spatial graph on grid coordinates.

    Parameters
    ----------
    grid_coords : ndarray, shape (N, 2)
        Spatial positions of grid points.
    k : int
        Number of nearest neighbors (before symmetrization).
    verbose : bool
        Print graph statistics.

    Returns
    -------
    edges : list of (int, int)
        Sorted list of undirected edges (u, v) with u < v.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(grid_coords)
    _, indices = nbrs.kneighbors(grid_coords)

    edge_set = set()
    for i in range(len(grid_coords)):
        for j in indices[i, 1:]:  # skip self (index 0)
            u, v = min(i, j), max(i, j)
            edge_set.add((u, v))

    edges = sorted(edge_set)

    if verbose:
        n_v = len(grid_coords)
        n_e = len(edges)
        print(
            f"  Spatial graph: {n_v} vertices, {n_e} edges, "
            f"mean degree {2 * n_e / n_v:.1f}"
        )

    return edges
