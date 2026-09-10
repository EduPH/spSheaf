"""
Multi-Slice Alignment via Gene Centroid Registration
=====================================================
Aligns tissue slices by finding the best translation between
matched gene centroid point clouds.  Each slice yields G centroids
(one per gene); the per-gene shift vectors are clustered to reject
outliers, and the median of the main cluster gives a robust
translation estimate.

Alignment only — no joint clustering.  Per-slice clustering already
achieves ARI=0.557, NMI=0.631; joint clustering degrades to ~0.29.

Usage:
    python multislice_alignment.py [path_to_molecules_csv]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import time
import os
import sys

from spsheaf import io, fibers, graph, sheaf, evaluation, pipeline

OUTPUT_DIR = "output_sheaf_boundary"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {path}")


# ============================================================================
# 0.  ENRICH SINGLE-SLICE RESULTS
# ============================================================================

def enrich_result(mol, result, grid_spacing=15, bandwidth=25,
                  k_graph=6, k_nbrs=50, uniform_dim=6):
    """
    Add fibers_norm, stalks, edges, and restrictions to a result dict
    from run_single_slice().  No-op if already present.
    """
    if "fibers_norm" in result and "stalks" in result:
        return

    slice_name = result["slice_name"]
    print(f"  Enriching {slice_name} with fibers/stalks...")

    mol_single = io.select_slice(mol, slice_name)   # planes follow io.MERFISH_INPUT

    grid_coords, fib, gene_names = fibers.build_kernel_fibers(
        mol_single, grid_spacing=grid_spacing, bandwidth=bandwidth)
    fibers_norm = fibers.normalize_fibers(fib)

    edges = graph.build_spatial_graph(grid_coords, k=k_graph)

    stalks = sheaf.compute_stalks(fibers_norm, grid_coords,
                                  uniform_dim=uniform_dim, k_nbrs=k_nbrs)

    restrictions = sheaf.compute_restriction_maps(stalks, edges, uniform_dim)

    result["fibers_norm"] = fibers_norm
    result["stalks"] = stalks
    result["edges"] = edges
    result["restrictions"] = restrictions
    result["gene_names"] = gene_names


# ============================================================================
# 1.  GENE CENTROIDS FROM GRID DATA
# ============================================================================

def compute_gene_centroids(slice_results):
    """
    For each slice, compute per-gene weighted centroids from the grid.

    For gene g in slice k:

        c_g^(k) = sum_v [ f_{v,g} * x_v ] / sum_v [ f_{v,g} ]

    Parameters
    ----------
    slice_results : list of dicts, each with grid_coords (N,2),
        fibers_norm (N,G), gene_names (G,)

    Returns
    -------
    centroids : list of dicts, gene_name -> (cx, cy)
    gene_names : list of str
    """
    gene_names = slice_results[0]["gene_names"]
    G = len(gene_names)

    centroids = []
    for sr in slice_results:
        coords = sr["grid_coords"]
        fib = sr["fibers_norm"]
        c = {}
        for g in range(G):
            w = fib[:, g]
            total = w.sum()
            if total > 1e-10:
                cx = np.dot(w, coords[:, 0]) / total
                cy = np.dot(w, coords[:, 1]) / total
                c[gene_names[g]] = (cx, cy)
        centroids.append(c)

    return centroids, gene_names


# ============================================================================
# 2.  ROBUST TRANSLATION FROM MATCHED POINT CLOUDS
# ============================================================================

def compute_shift(centroids_ref, centroids_tgt, gene_names, eps=None):
    """
    Best translation between two matched gene centroid point clouds.

    For each gene g, the shift vector is:
        delta_g = c_ref[g] - c_tgt[g]

    The shift vectors may form several clusters (genes in different
    tissue regions can behave differently).  We build a radius graph
    on the shifts, find connected components, and use the largest
    component to estimate the translation.

    Parameters
    ----------
    centroids_ref, centroids_tgt : dict gene -> (cx, cy)
    gene_names : list of str
    eps : float or None — neighborhood radius in shift space.
        If None, defaults to 1.5x the median 5-NN distance.

    Returns
    -------
    shift : (dx, dy) in um
    inlier_frac : fraction of genes in the largest component
    iqr : (iqr_x, iqr_y) of inlier shifts
    """
    common = [g for g in gene_names
              if g in centroids_ref and g in centroids_tgt]
    shifts = np.array([[centroids_ref[g][0] - centroids_tgt[g][0],
                        centroids_ref[g][1] - centroids_tgt[g][1]]
                       for g in common])
    N = len(shifts)

    # Adaptive radius
    if eps is None:
        tree = cKDTree(shifts)
        k_nn = min(6, N)
        dists, _ = tree.query(shifts, k=k_nn)
        eps = np.median(dists[:, -1]) * 1.5

    # Build radius graph and find connected components via BFS
    tree = cKDTree(shifts)
    neighbors = tree.query_ball_tree(tree, r=eps)

    comp_label = -np.ones(N, dtype=int)
    comp_id = 0
    for start in range(N):
        if comp_label[start] >= 0:
            continue
        # BFS
        queue = [start]
        comp_label[start] = comp_id
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for nb in neighbors[node]:
                if comp_label[nb] < 0:
                    comp_label[nb] = comp_id
                    queue.append(nb)
        comp_id += 1

    # Largest connected component
    unique, counts = np.unique(comp_label, return_counts=True)
    main = unique[np.argmax(counts)]
    inlier_mask = comp_label == main

    n_components = len(unique)
    inlier_shifts = shifts[inlier_mask]
    n_in = inlier_mask.sum()
    n_out = N - n_in

    dx = np.median(inlier_shifts[:, 0])
    dy = np.median(inlier_shifts[:, 1])

    iqr_x = np.percentile(inlier_shifts[:, 0], 75) - np.percentile(inlier_shifts[:, 0], 25)
    iqr_y = np.percentile(inlier_shifts[:, 1], 75) - np.percentile(inlier_shifts[:, 1], 25)

    print(f"    {N} genes, {n_components} components: "
          f"largest={n_in}, outliers={n_out}, "
          f"shift=({dx:.1f}, {dy:.1f}) um, IQR=({iqr_x:.1f}, {iqr_y:.1f}) um")

    return (dx, dy), n_in / N, (iqr_x, iqr_y)


def align_all_slices(slice_results, ref_idx=None):
    """
    Align all slices to a reference via gene centroid translation.

    Returns
    -------
    transforms : list of dicts with shift_um, inlier_frac, iqr
    ref_idx : int
    """
    K = len(slice_results)
    if ref_idx is None:
        ref_idx = K // 2

    print(f"  Reference slice: {ref_idx}")

    centroids, gene_names = compute_gene_centroids(slice_results)
    c_ref = centroids[ref_idx]

    transforms = []
    for k in range(K):
        if k == ref_idx:
            transforms.append(dict(shift_um=(0.0, 0.0),
                                   inlier_frac=1.0, iqr=(0.0, 0.0)))
            continue

        print(f"\n  Slice {k} -> ref {ref_idx}:")
        shift, inlier_frac, iqr = compute_shift(
            c_ref, centroids[k], gene_names)
        transforms.append(dict(shift_um=shift,
                                inlier_frac=inlier_frac, iqr=iqr))

    return transforms, ref_idx


# ============================================================================
# 3.  APPLY TRANSFORMS
# ============================================================================

def apply_transforms(slice_results, transforms):
    """Add grid_coords_aligned to each result."""
    for k, sr in enumerate(slice_results):
        shift = np.array(transforms[k]["shift_um"])
        sr["grid_coords_aligned"] = sr["grid_coords"] + shift
    print(f"  Applied transforms to {len(slice_results)} slices")


# ============================================================================
# 4.  RASTERIZE BOUNDARY FIELDS (FOR DIAGNOSTIC PLOTS)
# ============================================================================

def rasterize_to_common_grid(slice_results, grid_spacing=15.0):
    """Put each slice's boundary field onto a common pixel grid."""
    all_x_min = min(r["grid_coords"][:, 0].min() for r in slice_results)
    all_y_min = min(r["grid_coords"][:, 1].min() for r in slice_results)
    all_x_max = max(r["grid_coords"][:, 0].max() for r in slice_results)
    all_y_max = max(r["grid_coords"][:, 1].max() for r in slice_results)

    pad = 2 * grid_spacing
    all_x_min -= pad;  all_y_min -= pad
    all_x_max += pad;  all_y_max += pad

    M = int(np.ceil((all_x_max - all_x_min) / grid_spacing)) + 1
    N = int(np.ceil((all_y_max - all_y_min) / grid_spacing)) + 1

    images = []
    for sr in slice_results:
        coords = sr["grid_coords"]
        bfield = sr["boundary_smooth"]

        pi = np.clip(
            np.round((coords[:, 0] - all_x_min) / grid_spacing).astype(int),
            0, M - 1)
        pj = np.clip(
            np.round((coords[:, 1] - all_y_min) / grid_spacing).astype(int),
            0, N - 1)

        img = np.zeros((M, N), dtype=np.float64)
        for idx in range(len(coords)):
            img[pi[idx], pj[idx]] = bfield[idx]
        images.append(img)

    print(f"  Rasterized {len(images)} slices ({M} x {N} px)")
    return images, (all_x_min, all_y_min), (M, N)


# ============================================================================
# 5.  INTER-SLICE EDGES AND LABEL MATCHING
# ============================================================================

def build_interslice_edges(slice_results, radius_um=45.0):
    """Connect aligned grid points across adjacent slices."""
    edges = []
    for k in range(len(slice_results) - 1):
        ck = slice_results[k]["grid_coords_aligned"]
        cl = slice_results[k + 1]["grid_coords_aligned"]
        tree = cKDTree(cl)
        for i in range(len(ck)):
            for j in tree.query_ball_point(ck[i], r=radius_um):
                edges.append(((k, i), (k + 1, j)))

    print(f"  Inter-slice edges: {len(edges)} (radius={radius_um} um)")
    return edges


def compute_interslice_disagreement(slice_results, interslice_edges):
    """Sheaf disagreement on inter-slice edges."""
    disagree = np.zeros(len(interslice_edges))

    for ei, ((sk, si), (sl, sj)) in enumerate(interslice_edges):
        U_u = slice_results[sk]["stalks"][si]
        U_v = slice_results[sl]["stalks"][sj]
        fk = slice_results[sk]["fibers_norm"]
        fl = slice_results[sl]["fibers_norm"]

        M = U_u.T @ U_v
        try:
            L, _, Rt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            d = U_u.shape[1]
            L, Rt = np.eye(d), np.eye(d)

        x_u = U_u.T @ (fk[si] - fk.mean(axis=0))
        x_v = U_v.T @ (fl[sj] - fl.mean(axis=0))

        disagree[ei] = np.sum((L.T @ x_u - Rt @ x_v) ** 2)

    print(f"  Inter-slice disagreement: "
          f"mean={disagree.mean():.4f}, max={disagree.max():.4f}")
    return disagree


def match_labels_across_slices(slice_results, interslice_edges):
    """Hungarian matching of per-slice labels across adjacent slices."""
    K = len(slice_results)
    pair_metrics = []

    for k in range(K - 1):
        labels_k = slice_results[k]["labels"]
        labels_l = slice_results[k + 1]["labels"]

        pairs_k, pairs_l = [], []
        for (sk, si), (sl, sj) in interslice_edges:
            if sk == k and sl == k + 1:
                pairs_k.append(labels_k[si])
                pairs_l.append(labels_l[sj])

        if len(pairs_k) == 0:
            pair_metrics.append(dict(pair=(k, k+1), n_matched=0,
                                     agreement_rate=0.0))
            continue

        pairs_k = np.array(pairs_k)
        pairs_l = np.array(pairs_l)

        n = max(labels_k.max() + 1, labels_l.max() + 1)
        cost = np.zeros((n, n))
        for i in range(labels_k.max() + 1):
            for j in range(labels_l.max() + 1):
                cost[i, j] = -np.sum((pairs_k == i) & (pairs_l == j))

        row_ind, col_ind = linear_sum_assignment(cost)
        inv = {c: r for r, c in zip(row_ind, col_ind)}
        mapped = np.array([inv.get(l, -1) for l in pairs_l])
        agreement = np.mean(pairs_k == mapped)

        pair_metrics.append(dict(pair=(k, k+1), n_matched=len(pairs_k),
                                 agreement_rate=agreement))
        print(f"  Pair {k}->{k+1}: {len(pairs_k)} matched, "
              f"agreement={agreement:.3f}")

    return pair_metrics


# ============================================================================
# 6.  PERTURBATION EXPERIMENT
# ============================================================================

def run_perturbation_experiment(slice_results, perturbations=None,
                                grid_spacing=15):
    """
    Apply known translations to grid coords, run alignment on both
    original and perturbed, compare difference to ground truth.

    No molecules needed — operates entirely on grid data.
    """
    K = len(slice_results)
    if perturbations is None:
        perturbations = [(0, 0), (75, 45), (-60, 90),
                         (30, -75), (-90, 60)]
    perturbations = perturbations[:K]
    ref_idx = K // 2

    print("\n" + "=" * 64)
    print("  PERTURBATION EXPERIMENT")
    print("=" * 64)
    print(f"\n  Perturbations (um):")
    for k, (dx, dy) in enumerate(perturbations):
        ref = " (ref)" if k == ref_idx else ""
        print(f"    Slice {k}{ref}: ({dx:+.0f}, {dy:+.0f})")

    # A: baseline alignment
    print("\n--- A: Baseline alignment ---")
    baseline_transforms, _ = align_all_slices(slice_results, ref_idx=ref_idx)

    # B: perturb grid_coords directly
    print("\n--- B: Perturbing grid coordinates ---")
    perturbed = []
    for k, sr in enumerate(slice_results):
        dx, dy = perturbations[k]
        sr_copy = dict(sr)
        sr_copy["grid_coords"] = sr["grid_coords"] + np.array([dx, dy])
        perturbed.append(sr_copy)
        print(f"    Slice {k}: ({dx:+.0f}, {dy:+.0f}) um")

    # C: alignment on perturbed
    print("\n--- C: Alignment on perturbed data ---")
    perturbed_transforms, _ = align_all_slices(perturbed, ref_idx=ref_idx)

    # D: compare
    print(f"\n--- Recovery (ref=slice {ref_idx}) ---")
    print(f"  {'Slice':>5s}  {'True':>14s}  {'Recovered':>14s}  {'Err':>7s}")

    recovery_errors = []
    for k in range(K):
        gt_dx = -(perturbations[k][0] - perturbations[ref_idx][0])
        gt_dy = -(perturbations[k][1] - perturbations[ref_idx][1])

        delta_dx = (perturbed_transforms[k]["shift_um"][0] -
                    baseline_transforms[k]["shift_um"][0])
        delta_dy = (perturbed_transforms[k]["shift_um"][1] -
                    baseline_transforms[k]["shift_um"][1])

        err_dx = delta_dx - gt_dx
        err_dy = delta_dy - gt_dy
        err = np.sqrt(err_dx**2 + err_dy**2)

        recovery_errors.append(dict(
            slice=k, gt_dx=gt_dx, gt_dy=gt_dy,
            rec_dx=delta_dx, rec_dy=delta_dy,
            err_dx=err_dx, err_dy=err_dy, err=err))

        print(f"  {k:>5d}  ({gt_dx:+6.0f},{gt_dy:+6.0f})  "
              f"({delta_dx:+6.1f},{delta_dy:+6.1f})  {err:>6.1f}")

    mean_err = np.mean([e["err"] for e in recovery_errors])
    print(f"\n  Mean error: {mean_err:.1f} um "
          f"({mean_err / grid_spacing:.2f} grid spacings)")

    plot_perturbation_results(perturbations, recovery_errors,
                              ref_idx, grid_spacing=grid_spacing)

    return dict(perturbations=perturbations,
                recovery_errors=recovery_errors,
                baseline_transforms=baseline_transforms,
                perturbed_transforms=perturbed_transforms)


# ============================================================================
# 6b. SELF-ALIGNMENT EXPERIMENT
# ============================================================================

def run_self_alignment_experiment(slice_result, shift=(50.0, -70.0),
                                  interslice_radius=45.0, grid_spacing=15):
    """
    Align a slice with a shifted copy of itself.

    This is the "perfect" case: identical tissue, identical gene
    expression, known translation.  Expected results:
      - Exact shift recovery (0.0 um error)
      - 100% label agreement after alignment

    Parameters
    ----------
    slice_result : dict — a single slice from the pipeline
    shift : (dx, dy) in um — translation applied to the copy
    interslice_radius : float — radius for inter-slice edges
    grid_spacing : float — for reporting
    """
    print("\n" + "=" * 64)
    print("  SELF-ALIGNMENT EXPERIMENT")
    print("=" * 64)
    sname = slice_result["slice_name"]
    print(f"\n  Slice: {sname}")
    print(f"  Applied shift: ({shift[0]:+.1f}, {shift[1]:+.1f}) um")

    # Build the pair: original (ref=0) and shifted copy (target=1)
    original = dict(slice_result)
    shifted = dict(slice_result)
    shifted["grid_coords"] = slice_result["grid_coords"] + np.array(shift)

    pair = [original, shifted]

    # Align (ref = slice 0)
    print("\n--- Alignment ---")
    transforms, ref_idx = align_all_slices(pair, ref_idx=0)

    recovered = transforms[1]["shift_um"]
    err_dx = recovered[0] - (-shift[0])
    err_dy = recovered[1] - (-shift[1])
    err = np.sqrt(err_dx**2 + err_dy**2)

    print(f"\n  True shift:      ({-shift[0]:+.1f}, {-shift[1]:+.1f}) um")
    print(f"  Recovered shift: ({recovered[0]:+.1f}, {recovered[1]:+.1f}) um")
    print(f"  Error:           {err:.1f} um ({err / grid_spacing:.2f} grid spacings)")

    # Apply transforms and measure label agreement
    apply_transforms(pair, transforms)
    edges = build_interslice_edges(pair, radius_um=interslice_radius)

    # Label agreement (same labels on both sides, so should be 100%)
    labels = slice_result["labels"]
    n_matched = 0
    n_agree = 0
    for (sk, si), (sl, sj) in edges:
        if sk == 0 and sl == 1:
            n_matched += 1
            if labels[si] == labels[sj]:
                n_agree += 1

    agreement = n_agree / n_matched if n_matched > 0 else 0.0

    print(f"\n  Label agreement: {agreement:.3f} ({n_agree}/{n_matched})")
    print(f"  Inlier fraction: {transforms[1]['inlier_frac']:.0%}")

    return dict(shift=shift, recovered=recovered, err=err,
                agreement=agreement, n_matched=n_matched,
                transforms=transforms)


# ============================================================================
# 7.  FULL PIPELINE
# ============================================================================

def run_multislice_alignment(
        slice_results, mol=None,
        grid_spacing=15, bandwidth=25,
        k_graph=6, k_nbrs=50, uniform_dim=6,
        interslice_radius=45.0,
        compute_disagree=False, plot=True):
    """
    Full multi-slice alignment pipeline:
      1. Compute gene centroids from grid data
      2. Find best translation per slice via robust centroid matching
      3. Apply transforms
      4. Build inter-slice edges and measure label agreement
    """
    print("\n" + "=" * 64)
    print("  MULTI-SLICE ALIGNMENT (Gene Centroid Registration)")
    print("=" * 64)
    t0 = time.time()
    K = len(slice_results)
    print(f"  Slices: {K}")

    # Enrich if needed
    if any("fibers_norm" not in r or "stalks" not in r
           for r in slice_results):
        print("\n--- Enriching slice results ---")
        for r in slice_results:
            enrich_result(mol, r, grid_spacing=grid_spacing,
                          bandwidth=bandwidth, k_graph=k_graph,
                          k_nbrs=k_nbrs, uniform_dim=uniform_dim)

    # Rasterize for diagnostics
    print("\n--- Rasterize boundary fields ---")
    images, origin, shape = rasterize_to_common_grid(
        slice_results, grid_spacing=grid_spacing)

    # Align
    print("\n--- Gene centroid registration ---")
    transforms, ref_idx = align_all_slices(slice_results)

    # Apply
    print("\n--- Apply transforms ---")
    apply_transforms(slice_results, transforms)

    print("\n  Summary:")
    for k in range(K):
        s = transforms[k]["shift_um"]
        frac = transforms[k]["inlier_frac"]
        print(f"    Slice {k}: shift=({s[0]:.1f}, {s[1]:.1f}) um, "
              f"inliers={frac:.0%}")

    # Inter-slice disagreement (optional)
    disagree_before, disagree_after = [], []
    if compute_disagree:
        print("\n--- Inter-slice disagreement (before vs after) ---")
        for sr in slice_results:
            sr["_backup"] = sr["grid_coords_aligned"].copy()
            sr["grid_coords_aligned"] = sr["grid_coords"]
        eb = build_interslice_edges(slice_results, interslice_radius)
        db = compute_interslice_disagreement(slice_results, eb)
        for k in range(K - 1):
            mask = [i for i, ((sk,_),(sl,_)) in enumerate(eb)
                    if sk == k and sl == k+1]
            disagree_before.append(db[mask].mean() if mask else 0.0)
        for sr in slice_results:
            sr["grid_coords_aligned"] = sr.pop("_backup")
        ea = build_interslice_edges(slice_results, interslice_radius)
        da = compute_interslice_disagreement(slice_results, ea)
        for k in range(K - 1):
            mask = [i for i, ((sk,_),(sl,_)) in enumerate(ea)
                    if sk == k and sl == k+1]
            disagree_after.append(da[mask].mean() if mask else 0.0)
    else:
        print("\n--- Skipped disagreement (compute_disagree=False) ---")

    # Inter-slice edges and label agreement
    edges_aligned = build_interslice_edges(
        slice_results, radius_um=interslice_radius)
    print("\n--- Cross-slice label agreement ---")
    pair_metrics = match_labels_across_slices(slice_results, edges_aligned)

    mean_agr = np.mean([m["agreement_rate"] for m in pair_metrics])
    print(f"\n  Mean agreement: {mean_agr:.3f}")

    # Plots
    if plot:
        plot_alignment_diagnostics(images, transforms,
                                   grid_spacing=grid_spacing,
                                   ref_idx=ref_idx)
        plot_cross_slice_validation(slice_results, pair_metrics,
                                   disagree_before, disagree_after)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    return dict(slice_results=slice_results, transforms=transforms,
                ref_idx=ref_idx, interslice_edges=edges_aligned,
                pair_metrics=pair_metrics, images=images)


# ============================================================================
# 8.  DIAGNOSTIC PLOTS
# ============================================================================

def plot_alignment_diagnostics(images, transforms, grid_spacing=15.0,
                               ref_idx=None, save=True):
    """Boundary fields per slice + overlay against reference."""
    K = len(images)
    if ref_idx is None:
        ref_idx = K // 2
    n_cols = max(K, 3)
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10))

    for k in range(K):
        ax = axes[0, k]
        img = images[k]
        vmax = np.percentile(img[img > 0], 95) if (img > 0).any() else 1.0
        ax.imshow(img.T, origin="lower", cmap="hot", vmax=vmax,
                  aspect="equal")
        t = transforms[k]
        ref_label = " (REF)" if k == ref_idx else ""
        ax.set_title(f"Slice {k}{ref_label}\n"
                     f"shift=({t['shift_um'][0]:.0f}, "
                     f"{t['shift_um'][1]:.0f}) um", fontsize=9)
        if k == 0:
            ax.set_ylabel("Boundary field", fontsize=11)
    for k in range(K, n_cols):
        axes[0, k].set_visible(False)

    ref_n = images[ref_idx] / (images[ref_idx].max() + 1e-10)
    for k in range(min(K, n_cols)):
        ax = axes[1, k]
        if k == ref_idx:
            ax.imshow(ref_n.T, origin="lower", cmap="gray", aspect="equal")
            ax.set_title(f"Slice {k} (reference)", fontsize=9)
            if k == 0:
                ax.set_ylabel("Overlay vs ref", fontsize=10)
            continue

        tgt_n = images[k] / (images[k].max() + 1e-10)
        shift = transforms[k]["shift_um"]
        sp = (int(round(shift[0] / grid_spacing)),
              int(round(shift[1] / grid_spacing)))
        M, N = tgt_n.shape
        tgt_s = np.zeros_like(tgt_n)
        si0, si1 = max(0, -sp[0]), min(M, M - sp[0])
        sj0, sj1 = max(0, -sp[1]), min(N, N - sp[1])
        di0, dj0 = max(0, sp[0]), max(0, sp[1])
        h, w = si1 - si0, sj1 - sj0
        if h > 0 and w > 0 and di0 + h <= M and dj0 + w <= N:
            tgt_s[di0:di0+h, dj0:dj0+w] = tgt_n[si0:si0+h, sj0:sj0+w]

        overlay = np.zeros((M, N, 3))
        overlay[:, :, 0] = ref_n.T
        overlay[:, :, 1] = tgt_s.T
        overlay[:, :, 2] = tgt_s.T
        ax.imshow(overlay, origin="lower", aspect="equal")
        ax.set_title(f"Slice {k} vs ref", fontsize=9)
        if k == 0:
            ax.set_ylabel("Overlay vs ref", fontsize=10)
    for k in range(K, n_cols):
        axes[1, k].set_visible(False)

    fig.suptitle("Multi-Slice Alignment Diagnostics",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "multislice_alignment_diagnostics.png")
    return fig


def plot_cross_slice_validation(slice_results, pair_metrics,
                                inter_disagree_before, inter_disagree_after,
                                save=True):
    """Per-slice labels + agreement bar charts."""
    K = len(slice_results)
    fig, axes = plt.subplots(2, max(K, 3), figsize=(5 * max(K, 3), 10))

    n_cl = max(sr["labels"].max() for sr in slice_results) + 1
    cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_cl, 2))

    for k in range(K):
        ax = axes[0, k]
        sr = slice_results[k]
        coords = sr["grid_coords"]
        labels = sr["labels"]
        if sr["gt_int"] is not None:
            labels_vis = evaluation.hungarian_match_labels(
                labels, sr["gt_int"], sr["gt_valid"])
        else:
            labels_vis = labels
        for c in range(n_cl):
            mask = labels_vis == c
            if mask.sum():
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           s=1, alpha=0.5, color=cmap(c))
        ax.set_aspect("equal")
        ari = sr.get("ari", np.nan)
        nmi = sr.get("nmi", np.nan)
        ax.set_title(f"Slice {k}\nARI={ari:.3f}, NMI={nmi:.3f}", fontsize=9)
    for k in range(K, axes.shape[1]):
        axes[0, k].set_visible(False)

    ax = axes[1, 0]
    pairs = [f"{m['pair'][0]}->{m['pair'][1]}" for m in pair_metrics]
    rates = [m["agreement_rate"] for m in pair_metrics]
    bars = ax.bar(range(len(pairs)), rates, color="steelblue",
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=9)
    ax.set_ylabel("Label agreement rate")
    ax.set_title("Cross-slice label agreement", fontsize=10)
    ax.set_ylim(0, 1)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.3f}", ha="center", fontsize=9)

    ax = axes[1, 1]
    if inter_disagree_before and inter_disagree_after:
        x = np.arange(len(pairs))
        w = 0.35
        ax.bar(x - w/2, inter_disagree_before, w, label="Before",
               color="lightcoral", edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, inter_disagree_after, w, label="After",
               color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, fontsize=9)
        ax.set_ylabel("Mean disagreement")
        ax.set_title("Before vs after alignment", fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Disagreement\nnot computed",
                transform=ax.transAxes, fontsize=10,
                ha="center", va="center", color="gray")
        ax.set_axis_off()

    ax = axes[1, 2]
    info = "Summary\n\n"
    for k in range(K):
        sr = slice_results[k]
        info += f"Slice {k}: ARI={sr.get('ari', np.nan):.3f}, NMI={sr.get('nmi', np.nan):.3f}\n"
    info += f"\nMean agreement: {np.mean(rates):.3f}"
    ax.text(0.1, 0.5, info, transform=ax.transAxes,
            fontsize=10, va="center", fontfamily="monospace")
    ax.set_axis_off()

    for k in range(3, axes.shape[1]):
        axes[1, k].set_visible(False)

    fig.suptitle("Cross-Slice Validation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "multislice_cross_validation.png")
    return fig


def plot_perturbation_results(perturbations, recovery_errors,
                              ref_idx, grid_spacing=15.0, save=True):
    """Perturbation recovery figure."""
    K = len(perturbations)
    fig, axes = plt.subplots(2, max(K, 3), figsize=(5 * max(K, 3), 10))

    for k in range(K):
        ax = axes[0, k]
        e = recovery_errors[k]
        ref_label = " (REF)" if k == ref_idx else ""
        text = (f"Slice {k}{ref_label}\n\n"
                f"True:      ({e['gt_dx']:+.0f}, {e['gt_dy']:+.0f})\n"
                f"Recovered: ({e['rec_dx']:+.1f}, {e['rec_dy']:+.1f})\n"
                f"Error:     {e['err']:.1f} um")
        ax.text(0.5, 0.5, text, transform=ax.transAxes, fontsize=10,
                ha="center", va="center", fontfamily="monospace")
        color = "lightgreen" if e["err"] < grid_spacing else (
                "lightyellow" if e["err"] < 3 * grid_spacing else "mistyrose")
        ax.set_facecolor(color)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(K, axes.shape[1]):
        axes[0, k].set_visible(False)

    ax = axes[1, 0]
    gt_dx = [e["gt_dx"] for e in recovery_errors]
    gt_dy = [e["gt_dy"] for e in recovery_errors]
    rec_dx = [e["rec_dx"] for e in recovery_errors]
    rec_dy = [e["rec_dy"] for e in recovery_errors]
    ax.scatter(gt_dx, rec_dx, marker="o", s=80, color="steelblue",
               edgecolor="black", label="dx", zorder=3)
    ax.scatter(gt_dy, rec_dy, marker="s", s=80, color="coral",
               edgecolor="black", label="dy", zorder=3)
    all_vals = gt_dx + gt_dy + rec_dx + rec_dy
    lim = max(abs(v) for v in all_vals) * 1.2 + 10
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Ground truth (um)")
    ax.set_ylabel("Recovered (um)")
    ax.set_title("Shift recovery", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    errs = [e["err"] for e in recovery_errors]
    colors = ["steelblue" if e < grid_spacing else "coral" for e in errs]
    bars = ax.bar(range(K), errs, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.axhline(grid_spacing, color="red", linestyle="--", alpha=0.5,
               label=f"1 grid ({grid_spacing} um)")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"Slice {k}" for k in range(K)], fontsize=9)
    ax.set_ylabel("Error (um)")
    ax.set_title("Per-slice error", fontsize=10)
    ax.legend(fontsize=8)
    for bar, e in zip(bars, errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{e:.0f}", ha="center", fontsize=9)

    ax = axes[1, 2]
    s = f"{'Slice':>5s}  {'True':>12s}  {'Recovered':>12s}  {'Err':>5s}\n"
    s += "-" * 40 + "\n"
    for e in recovery_errors:
        s += (f"{e['slice']:>5d}  ({e['gt_dx']:+.0f},{e['gt_dy']:+.0f})"
              f"{'':>4s}  ({e['rec_dx']:+.0f},{e['rec_dy']:+.0f})"
              f"{'':>4s}  {e['err']:>4.0f}\n")
    s += f"\nMean: {np.mean(errs):.1f} um"
    ax.text(0.05, 0.5, s, transform=ax.transAxes,
            fontsize=9, va="center", fontfamily="monospace")
    ax.set_axis_off()

    for k in range(3, axes.shape[1]):
        axes[1, k].set_visible(False)

    fig.suptitle("Perturbation Recovery", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "multislice_perturbation_experiment.png")
    return fig


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    mol_path = "moffitt2018_export_full/molecules_full_add.csv.gz"
    if len(sys.argv) > 1:
        mol_path = sys.argv[1]

    # ---- Per-slice pipeline ----
    print("=" * 64)
    print("  STEP 0: Running per-slice sheaf pipeline")
    print("=" * 64)

    mol = io.load_data(mol_path)
    slice_info = io.identify_slices(mol)
    slice_names = sorted(slice_info["slice_name"].tolist())

    all_results = []
    for sn in slice_names:
        result = pipeline.run_single_slice(
            mol, sn, n_clusters=8,
            grid_spacing=15, bandwidth=25,
            k_graph=6, k_nbrs=50, uniform_dim=6,
            smooth_radius=60.0, spatial_weight=0.5,
            boundary_alpha=0.5, overcluster_factor=1.0,
            cleanup=True)
        all_results.append(result)

    # ---- Multi-slice alignment ----
    print("\n\n")
    results = run_multislice_alignment(
        slice_results=all_results, mol=mol,
        grid_spacing=15, bandwidth=25,
        k_graph=6, k_nbrs=50, uniform_dim=6,
        interslice_radius=45.0, plot=True)

    # ---- Perturbation experiment ----
    print("\n\n")
    run_perturbation_experiment(
        slice_results=all_results,
        perturbations=[(0, 0), (75, 45), (-60, 90),
                       (30, -75), (-90, 60)],
        grid_spacing=15)

    # ---- Self-alignment experiment ----
    print("\n\n")
    run_self_alignment_experiment(
        all_results[2],  # middle slice (best coverage)
        shift=(50.0, -70.0),
        interslice_radius=45.0,
        grid_spacing=15)

    print("\n\nDone. Outputs in:", OUTPUT_DIR)