"""
Clustering methods for spatial domain inference.

Provides:

1. **Boundary-aware clustering** — The sheaf boundary field is used
   to attenuate ambiguous boundary points via exponential weighting.
   Supports an over-cluster-then-merge strategy to preserve small
   domains (e.g. PVT) that standard k-means tends to absorb into
   larger neighbors.

2. **Over-cluster-then-merge** — Run k-means with more clusters than
   needed (e.g. 2×), then merge the most expression-similar clusters
   via agglomerative clustering on their expression centroids. This
   gives small but expression-distinct domains their own initial
   cluster, which survives the merge because its centroid is far
   from all others.

3. **Spatial connectivity cleanup** — After clustering, small
   disconnected fragments are detected via BFS on the spatial graph
   and reassigned to the label of their nearest large-component
   neighbor.
"""

import numpy as np
from collections import deque
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering


def boundary_aware_clustering(fibers_norm, grid_coords, boundary_smooth,
                              edges, n_clusters=8, spatial_weight=0.5,
                              boundary_alpha=0.5, n_pca=10, n_init=20,
                              overcluster_factor=1.0,
                              random_state=42, verbose=True):
    """
    Cluster with boundary-aware exponential weighting.

    When ``overcluster_factor > 1``, k-means is run with
    ``n_clusters * overcluster_factor`` clusters, then fine clusters
    are merged back to ``n_clusters`` using adjacency-constrained
    agglomerative clustering on expression centroids. Only spatially
    neighboring clusters can merge, so sub-clusters of the same domain
    reunify first, while small expression-distinct domains survive.

    Steps:
        1. PCA embedding of normalized fibers
        2. Exponential interior weight: w = exp(-α · b / median(b))
        3. Weight expression features row-wise
        4. Concatenate with scaled spatial coordinates
        5. K-means with k_fine = n_clusters * overcluster_factor
        6. Adjacency-constrained merge → n_clusters

    Parameters
    ----------
    fibers_norm : ndarray, shape (N, G)
        Normalized expression fibers.
    grid_coords : ndarray, shape (N, 2)
        Spatial positions.
    boundary_smooth : ndarray, shape (N,)
        Smoothed boundary scores.
    edges : list of (int, int)
        Spatial graph edges (used for adjacency-constrained merge).
    n_clusters : int
        Final number of output domains.
    spatial_weight : float
        Weight for spatial coordinates in clustering input.
    boundary_alpha : float
        Exponential weighting strength. At median boundary score,
        w ≈ 0.607; at twice median, w ≈ 0.368.
    n_pca : int
        Number of PCA components.
    n_init : int
        Number of k-means random initializations.
    overcluster_factor : float
        Over-clustering factor. Set to 1.0 to disable. Default 2.0
        runs k-means with 2× clusters then merges back.
    random_state : int
        Random seed for reproducibility.
    verbose : bool
        Print cluster sizes.

    Returns
    -------
    labels : ndarray, shape (N,)
        Predicted domain labels (0-indexed).
    """
    # PCA embedding
    pca = PCA(n_components=n_pca).fit(fibers_norm)
    features = pca.transform(fibers_norm)

    # Exponential boundary weighting
    med_b = np.median(boundary_smooth) + 1e-10
    interior_weight = np.exp(-boundary_alpha * boundary_smooth / med_b)

    # Weight and scale
    features_scaled = StandardScaler().fit_transform(features)
    features_weighted = features_scaled * interior_weight[:, None]

    spatial_scaled = StandardScaler().fit_transform(grid_coords)
    cluster_input = np.hstack([
        features_weighted,
        spatial_scaled * spatial_weight,
    ])

    # K-means (possibly over-clustered)
    k_fine = int(round(n_clusters * max(overcluster_factor, 1.0)))
    km = KMeans(
        n_clusters=k_fine, n_init=n_init, random_state=random_state,
    )
    labels_fine = km.fit_predict(cluster_input)

    # Merge if over-clustered
    if k_fine > n_clusters:
        labels = merge_fine_clusters(
            labels_fine, features, edges, n_clusters, verbose=verbose,
        )
    else:
        labels = labels_fine

    if verbose:
        sizes = sorted(
            [int((labels == c).sum()) for c in range(n_clusters)],
            reverse=True,
        )
        print(f"  Clusters ({n_clusters}): {sizes}")

    return labels


def merge_fine_clusters(labels_fine, features, edges, n_clusters,
                        verbose=True):
    """
    Merge fine clusters down to n_clusters using adjacency-constrained
    agglomerative clustering on expression centroids.

    Only spatially neighboring clusters (sharing at least one edge in
    the spatial graph) can be merged. This ensures that sub-clusters
    of the same domain reunify first, while preventing merges between
    expression-similar but spatially distant clusters.

    Small expression-distinct domains (e.g. PVT) survive because their
    centroids are far from their spatial neighbors.

    Parameters
    ----------
    labels_fine : ndarray, shape (N,)
        Fine cluster labels from over-clustered k-means.
    features : ndarray, shape (N, d)
        PCA expression features (unweighted — the raw expression
        signal is used for merging decisions).
    edges : list of (int, int)
        Spatial graph edges.
    n_clusters : int
        Target number of merged clusters.
    verbose : bool
        Print merge summary.

    Returns
    -------
    labels_merged : ndarray, shape (N,)
        Merged cluster labels (0-indexed, contiguous).
    """
    from scipy.sparse import lil_matrix

    k_fine = len(set(labels_fine))

    # Compute expression centroid for each fine cluster
    centroids = np.zeros((k_fine, features.shape[1]))
    for c in range(k_fine):
        mask = labels_fine == c
        if mask.sum() > 0:
            centroids[c] = features[mask].mean(axis=0)

    # Build cluster-level adjacency: two fine clusters are adjacent
    # if any spatial edge connects a point in one to a point in the other
    adjacency = lil_matrix((k_fine, k_fine), dtype=np.int8)
    for u, v in edges:
        lu, lv = labels_fine[u], labels_fine[v]
        if lu != lv:
            adjacency[lu, lv] = 1
            adjacency[lv, lu] = 1
    connectivity = adjacency.tocsr()

    # Adjacency-constrained agglomerative clustering on centroids
    agg = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward",
        connectivity=connectivity,
    )
    merge_labels = agg.fit_predict(centroids)

    # Map fine → merged
    fine_to_merged = {c: merge_labels[c] for c in range(k_fine)}
    labels_merged = np.array([fine_to_merged[l] for l in labels_fine])

    if verbose:
        print(
            f"  Merged {k_fine} fine clusters → {n_clusters} domains "
            f"(adjacency-constrained)"
        )

    return labels_merged


def spatial_cleanup(labels, edges, grid_coords=None,
                    min_fragment_frac=0.01, max_iterations=5, verbose=True):
    """
    Remove small disconnected fragments from cluster assignments.

    After k-means, some clusters may have spatially disconnected fragments
    (small groups of points assigned to a cluster far from its main body).
    This function finds connected components of each cluster on the spatial
    graph and reassigns small fragments to the label of their nearest
    large-component neighbor.

    Parameters
    ----------
    labels : ndarray, shape (N,)
        Cluster assignments from k-means.
    edges : list of (int, int)
        Spatial graph edges.
    grid_coords : ndarray, shape (N, 2) or None
        Spatial positions (used as fallback for unresolved points).
    min_fragment_frac : float
        Components smaller than this fraction of total points are
        considered fragments. Default: 1%.
    max_iterations : int
        Maximum number of cleanup passes.
    verbose : bool
        Print reassignment statistics.

    Returns
    -------
    labels_clean : ndarray, shape (N,)
        Cleaned cluster assignments.
    n_reassigned : int
        Total number of points reassigned.
    """
    N = len(labels)
    min_size = max(3, int(N * min_fragment_frac))
    labels_clean = labels.copy()

    # Build adjacency list from edges
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    total_reassigned = 0

    for iteration in range(max_iterations):
        n_reassigned = 0
        unique_labels = sorted(set(labels_clean))

        # Find connected components for each cluster
        fragment_points = []

        for lab in unique_labels:
            members = set(np.where(labels_clean == lab)[0])
            if len(members) == 0:
                continue

            # BFS to find connected components within this cluster
            visited = set()
            components = []

            for start in members:
                if start in visited:
                    continue
                component = []
                queue = deque([start])
                visited.add(start)
                while queue:
                    node = queue.popleft()
                    component.append(node)
                    for nb in adj[node]:
                        if nb not in visited and nb in members:
                            visited.add(nb)
                            queue.append(nb)
                components.append(component)

            if len(components) <= 1:
                continue

            # Keep the largest component, mark others as fragments
            components.sort(key=len, reverse=True)
            for comp in components[1:]:
                if len(comp) < min_size:
                    fragment_points.extend(comp)

        if not fragment_points:
            break

        # Reassign fragment points to nearest non-fragment neighbor
        fragment_set = set(fragment_points)
        unresolved = []

        for pt in fragment_points:
            reassigned = False
            for nb in adj[pt]:
                if nb not in fragment_set:
                    labels_clean[pt] = labels_clean[nb]
                    n_reassigned += 1
                    reassigned = True
                    break
            if not reassigned:
                unresolved.append(pt)

        # Fallback: nearest non-fragment point by spatial distance
        if unresolved and grid_coords is not None:
            non_frag = np.array([i for i in range(N) if i not in fragment_set])
            if len(non_frag) > 0:
                tree = cKDTree(grid_coords[non_frag])
                for pt in unresolved:
                    _, nearest = tree.query(grid_coords[pt])
                    labels_clean[pt] = labels_clean[non_frag[nearest]]
                    n_reassigned += 1

        total_reassigned += n_reassigned

        if n_reassigned == 0:
            break

    if verbose and total_reassigned > 0:
        print(
            f"  Cleanup: {total_reassigned} points reassigned "
            f"({total_reassigned / N:.1%})"
        )

    return labels_clean, total_reassigned