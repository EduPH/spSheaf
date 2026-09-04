#!/usr/bin/env python
"""
Score sheaf-inferred domains against the Allen Mouse Brain Common Coordinate
Framework (CCFv3) as an external proxy ground truth, yielding ARI / NMI.

Why a proxy
-----------
The Vizgen Brain Receptor Showcase ships no spatial-domain annotation, and its
receptor/GPCR panel lacks the canonical cell-type markers a data-driven
reference would need. The Allen CCF is an independent, anatomy-based partition
of the mouse brain, so aligning our section to a CCF coronal plane and reading
off each grid point's anatomical division gives a defensible reference to score
against.

What this script does
---------------------
1. Downloads the Allen mouse atlas once via ``brainglobe-atlasapi`` (cached in
   ~/.brainglobe; ~150-600 MB depending on resolution).
2. Extracts a coronal annotation slice at the antero-posterior (AP) level you
   choose (``--ap-um``), or searches a range for the best-fitting level
   (``--scan-ap``).
3. Collapses the fine CCF labels to major anatomical *divisions* (Isocortex,
   Hippocampal formation, Thalamus, Hypothalamus, Striatum, fiber tracts, ...).
4. Registers the sheaf grid to that CCF slice with an automatic
   similarity transform: it tries all 8 flip/rotation orientations, keeps the
   one whose grid best overlaps the atlas tissue, then refines with ICP.
5. Assigns each grid point its CCF division and computes ARI / NMI against the
   sheaf domains.
6. Writes ``allen_reference_labels.csv``, ``metrics_vs_allen.csv`` and a 3-panel
   ``allen_overlay.png`` (sheaf domains | CCF divisions on the grid |
   registration check). ALWAYS eyeball the overlay: ARI/NMI are only meaningful
   if the middle/right panels line up anatomically. If not, change ``--ap-um``
   (and, rarely, pass ``--no-auto-orient`` with explicit ``--flip-x/--flip-y/
   --swap-xy``).

Usage
-----
    pip install brainglobe-atlasapi
    python reference_allen.py --labels output_brain_receptor/domain_labels.csv \\
        --ap-um 6800 --out-dir output_brain_receptor

    # or let it search the AP level automatically:
    python reference_allen.py --labels output_brain_receptor/domain_labels.csv \\
        --scan-ap 5000,9000,250 --out-dir output_brain_receptor
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)


# CCFv3 major-division structure IDs (root -> these are the coarse classes).
DIVISION_IDS = {
    315: "Isocortex", 698: "OLF", 1089: "HPF", 703: "CTXsp",
    477: "STR", 803: "PAL", 549: "TH", 1097: "HY",
    313: "MB", 771: "P", 354: "MY", 512: "CB",
    1009: "fiber tracts", 73: "VS",
}


# ----------------------------------------------------------------------------
# Geometry helpers (pure; unit-tested separately in _test_reference_allen.py)
# ----------------------------------------------------------------------------

def umeyama_similarity(src, dst):
    """Least-squares similarity transform (scale s, rotation R, translation t)
    mapping src -> dst. src, dst: (N, 2). Returns (s, R, t)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim(pts, s, R, t):
    return (s * (R @ pts.T).T) + t


ORIENTATIONS = []
for _sx in (1, -1):
    for _sy in (1, -1):
        for _swap in (False, True):
            ORIENTATIONS.append((_sx, _sy, _swap))


def _orient(pts, sx, sy, swap):
    q = pts.copy().astype(float)
    if swap:
        q = q[:, ::-1].copy()
    q[:, 0] *= sx
    q[:, 1] *= sy
    return q


def register_grid_to_atlas(grid_xy, atlas_xy, auto_orient=True,
                           fixed_orient=None, icp_iters=25, sample=4000,
                           seed=0):
    """Register grid points onto the atlas tissue point cloud.

    Coarse init: normalise both to zero mean / unit RMS, try candidate
    orientations (flips + axis swap), pick the one whose transformed grid best
    overlaps the atlas tissue (max fraction within one atlas spacing), then
    refine with ICP (similarity). Returns (transformed_grid, params, overlap).
    """
    rng = np.random.default_rng(seed)
    tree_a = cKDTree(atlas_xy)
    # atlas nearest-neighbour spacing (for the overlap tolerance)
    dd, _ = tree_a.query(atlas_xy[rng.choice(len(atlas_xy),
                         size=min(2000, len(atlas_xy)), replace=False)], k=2)
    tol = float(np.median(dd[:, 1])) * 1.5

    orients = ORIENTATIONS if auto_orient else [fixed_orient]
    best = None
    for o in orients:
        q = _orient(grid_xy, *o)
        # coarse similarity from moments (match centroid + RMS scale + PCA axes)
        s, R, t = _moment_align(q, atlas_xy)
        qt = apply_sim(q, s, R, t)
        d, _ = tree_a.query(qt, k=1)
        overlap = float((d < tol).mean())
        if best is None or overlap > best[0]:
            best = (overlap, o, (s, R, t), qt)
    overlap0, o, (s, R, t), qt = best

    # ICP refinement (similarity) on a subsample
    idx = rng.choice(len(qt), size=min(sample, len(qt)), replace=False)
    cur = qt.copy()
    for _ in range(icp_iters):
        d, j = tree_a.query(cur[idx], k=1)
        keep = d < tol * 2
        if keep.sum() < 20:
            break
        s2, R2, t2 = umeyama_similarity(cur[idx][keep], atlas_xy[j[keep]])
        cur = apply_sim(cur, s2, R2, t2)
    d, _ = tree_a.query(cur, k=1)
    overlap = float((d < tol).mean())
    return cur, dict(orient=o, overlap0=overlap0, overlap=overlap, tol=tol), overlap


def _moment_align(src, dst):
    """Similarity from second moments: match centroid, RMS scale, PCA axes."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    rms_s = np.sqrt((sc ** 2).sum(1).mean())
    rms_d = np.sqrt((dc ** 2).sum(1).mean())
    s = rms_d / max(rms_s, 1e-9)
    # principal axes
    Us, _, _ = np.linalg.svd(np.cov(sc.T))
    Ud, _, _ = np.linalg.svd(np.cov(dc.T))
    R = Ud @ Us.T
    if np.linalg.det(R) < 0:
        Ud[:, -1] *= -1
        R = Ud @ Us.T
    t = mu_d - s * R @ mu_s
    return s, R, t


# ----------------------------------------------------------------------------
# Atlas access
# ----------------------------------------------------------------------------

def load_atlas(atlas_name):
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas
    except Exception:
        try:
            from bg_atlasapi import BrainGlobeAtlas  # older package name
        except Exception as e:
            raise SystemExit(
                "Could not import brainglobe. Install with:\n"
                "    pip install brainglobe-atlasapi\n"
                f"(original error: {e})")
    print(f"Loading atlas '{atlas_name}' (downloads + caches on first use)...")
    return BrainGlobeAtlas(atlas_name)


def division_map(atlas):
    """id -> division acronym, via each structure's ancestor path."""
    out = {}
    for sid in atlas.structures.keys():
        try:
            path = atlas.structures[sid]["structure_id_path"]
        except Exception:
            continue
        div = None
        for anc in path:  # root -> leaf; keep the deepest division ancestor
            if anc in DIVISION_IDS:
                div = DIVISION_IDS[anc]
        if div:
            out[sid] = div
    return out


def coronal_slice(atlas, ap_idx, id2div):
    """Return (atlas_xy_um (M,2)=[ml,dv], labels (M,) division strings)."""
    ann = atlas.annotation
    res_ap, res_dv, res_ml = atlas.resolution
    sl = ann[ap_idx]  # (dv, ml)
    dv, ml = np.nonzero(sl > 0)
    ids = sl[dv, ml]
    labels = np.array([id2div.get(int(i), "other") for i in ids], dtype=object)
    keep = labels != "other"
    xy = np.column_stack([ml[keep] * res_ml, dv[keep] * res_dv]).astype(float)
    return xy, labels[keep]


def fair_scores(ref, sheaf, mask):
    """Scores that are fair to a fine partition scored against coarse anatomy.

    Raw ARI/NMI penalise the method's intentional over-segmentation (15 domains
    subdividing single CCF divisions). Homogeneity asks the fairer question --
    does each sheaf domain stay inside ONE division -- and the majority-merge
    collapses each domain to its dominant division for an apples-to-apples score.
    """
    r = np.asarray(ref)[mask]
    s = np.asarray(sheaf)[mask]
    ari = adjusted_rand_score(r, s)
    nmi = normalized_mutual_info_score(r, s)
    homo, compl, vmeas = homogeneity_completeness_v_measure(r, s)
    dfm = pd.DataFrame({"s": s, "r": r})
    maj = dfm.groupby("s")["r"].agg(lambda x: x.value_counts().idxmax())
    pred = dfm["s"].map(maj).values
    acc = float((pred == r).mean())
    return dict(ari=ari, nmi=nmi, homogeneity=homo, completeness=compl,
                v_measure=vmeas, merged_accuracy=acc,
                merged_ari=adjusted_rand_score(r, pred),
                merged_nmi=normalized_mutual_info_score(r, pred))


def assign_and_score(res, sheaf, tol_mult=2.0):
    """Map registered grid points to CCF divisions and score vs sheaf."""
    tree = cKDTree(res["atlas_xy"])
    d, j = tree.query(res["tgrid"], k=1)
    inside = d < res["params"]["tol"] * tol_mult
    ref = np.where(inside, res["atlas_lab"][j], "outside")
    mask = ref != "outside"
    if mask.sum() < 10 or len(set(sheaf[mask])) < 2 or len(set(ref[mask])) < 2:
        return ref, float("nan"), float("nan"), mask
    ari = adjusted_rand_score(ref[mask], sheaf[mask])
    nmi = normalized_mutual_info_score(ref[mask], sheaf[mask])
    return ref, ari, nmi, mask


def contact_sheet(grid, sheaf, eval_ap, spec, out_dir):
    lo, hi, step = [float(v) for v in spec.split(",")]
    aps = np.arange(lo, hi + 1e-6, step)
    n = len(aps)
    ncol = min(5, n)
    nrow = int(np.ceil((n + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    # first panel: the sheaf domains, for reference
    uniq = sorted(set(sheaf))
    scm = plt.cm.get_cmap("tab20", max(len(uniq), 2))
    for i, c in enumerate(uniq):
        m = sheaf == c
        axes[0].scatter(grid[m, 0], grid[m, 1], s=2, color=scm(i))
    axes[0].set_title("YOUR sheaf domains", fontsize=10, fontweight="bold")
    axes[0].set_aspect("equal"); axes[0].invert_yaxis()
    axes[0].set_xticks([]); axes[0].set_yticks([])

    print("\nAP scan (pick the panel whose anatomy matches YOUR domains):")
    for k, ap in enumerate(aps):
        ax = axes[k + 1]
        r = eval_ap(ap)
        if r is None:
            ax.set_visible(False); continue
        ref, ari, nmi, mask = assign_and_score(r, sheaf)
        divs = sorted(set(ref))
        dcm = plt.cm.get_cmap("tab20", max(len(divs), 2))
        for i, dn in enumerate(divs):
            m = ref == dn
            ax.scatter(grid[m, 0], grid[m, 1], s=2, color=dcm(i))
        has_th = "TH" in set(ref[mask])
        ax.set_title(f"AP={r['ap_um']:.0f}um  ARI={ari:.2f}"
                     f"{'  [has TH]' if has_th else ''}", fontsize=9)
        ax.set_aspect("equal"); ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        print(f"  AP {r['ap_um']:6.0f} um  overlap={r['overlap']:.3f}  "
              f"ARI={ari:.3f}  NMI={nmi:.3f}  divisions={divs}")
    for k in range(len(aps) + 1, len(axes)):
        axes[k].set_visible(False)
    fig.suptitle("Allen CCF division map at each AP level - pick the match, "
                 "then rerun with --ap-um <that value>", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "allen_contact_sheet.png")
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {path} - open it, pick the best-matching AP, then run "
          "with --ap-um <that value>.")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default="output_brain_receptor/domain_labels.csv")
    p.add_argument("--atlas", default="allen_mouse_25um")
    p.add_argument("--ap-um", type=float, default=None,
                   help="AP position of the coronal plane in microns.")
    p.add_argument("--scan-ap", default=None,
                   help="min,max,step (um) to search for best-overlap AP level.")
    p.add_argument("--out-dir", default="output_brain_receptor")
    p.add_argument("--contact-sheet", default=None,
                   help="min,max,step (um): render CCF-division-on-grid "
                        "thumbnails across AP levels so you can pick the "
                        "anatomically matching one by eye. Writes "
                        "allen_contact_sheet.png and exits.")
    p.add_argument("--no-auto-orient", action="store_true")
    p.add_argument("--flip-x", action="store_true")
    p.add_argument("--flip-y", action="store_true")
    p.add_argument("--swap-xy", action="store_true")
    return p


def main():
    a = build_parser().parse_args()
    df = pd.read_csv(a.labels)
    grid = df[["x", "y"]].values.astype(float)
    sheaf = df["label"].values
    print(f"Loaded {len(grid):,} grid points, {len(set(sheaf))} sheaf domains")

    atlas = load_atlas(a.atlas)
    res_ap = atlas.resolution[0]
    n_ap = atlas.annotation.shape[0]
    id2div = division_map(atlas)
    print(f"Atlas AP extent: 0..{(n_ap-1)*res_ap:.0f} um "
          f"({n_ap} slices @ {res_ap:.0f} um)")

    fixed = ((-1 if a.flip_x else 1), (-1 if a.flip_y else 1), a.swap_xy)

    def eval_ap(ap_um):
        ap_idx = int(round(ap_um / res_ap))
        ap_idx = max(0, min(n_ap - 1, ap_idx))
        axy, alab = coronal_slice(atlas, ap_idx, id2div)
        if len(axy) < 100:
            return None
        tg, params, ov = register_grid_to_atlas(
            grid, axy, auto_orient=not a.no_auto_orient, fixed_orient=fixed)
        return dict(ap_um=ap_idx * res_ap, ap_idx=ap_idx, atlas_xy=axy,
                    atlas_lab=alab, tgrid=tg, params=params, overlap=ov)

    if a.contact_sheet:
        os.makedirs(a.out_dir, exist_ok=True)
        contact_sheet(grid, sheaf, eval_ap, a.contact_sheet, a.out_dir)
        return

    if a.scan_ap:
        lo, hi, step = [float(v) for v in a.scan_ap.split(",")]
        cands = np.arange(lo, hi + 1e-6, step)
        print(f"Scanning {len(cands)} AP levels for best overlap...")
        best = None
        for ap in cands:
            r = eval_ap(ap)
            if r is None:
                continue
            print(f"  AP {r['ap_um']:6.0f} um  overlap={r['overlap']:.3f}")
            if best is None or r["overlap"] > best["overlap"]:
                best = r
        res = best
        print(f"Best AP = {res['ap_um']:.0f} um (overlap {res['overlap']:.3f})")
    else:
        if a.ap_um is None:
            raise SystemExit("Pass --ap-um <microns> or --scan-ap min,max,step")
        res = eval_ap(a.ap_um)

    # Assign each grid point its CCF division (nearest atlas voxel)
    ref, ari, nmi, mask = assign_and_score(res, sheaf)
    fs = fair_scores(ref, sheaf, mask)
    print(f"\nGrid points scored: {mask.sum():,}/{len(grid):,} "
          f"({100*mask.mean():.0f}% inside tissue)")
    print(f"CCF divisions present: {sorted(set(ref[mask]))}")
    print("\n  --- raw (15 sheaf domains vs coarse CCF divisions) ---")
    print(f"  ARI                = {fs['ari']:.3f}")
    print(f"  NMI                = {fs['nmi']:.3f}")
    print("  --- fair to over-segmentation ---")
    print(f"  homogeneity        = {fs['homogeneity']:.3f}   "
          "(each sheaf domain stays within one division)")
    print(f"  completeness       = {fs['completeness']:.3f}")
    print(f"  V-measure          = {fs['v_measure']:.3f}")
    print("  --- majority-merged sheaf domains -> divisions ---")
    print(f"  accuracy           = {fs['merged_accuracy']:.3f}   "
          "(grid points whose domain's majority division is correct)")
    print(f"  merged ARI         = {fs['merged_ari']:.3f}")
    print(f"  merged NMI         = {fs['merged_nmi']:.3f}")

    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame({"x": grid[:, 0], "y": grid[:, 1],
                  "allen_division": ref, "sheaf_label": sheaf}
                 ).to_csv(os.path.join(a.out_dir, "allen_reference_labels.csv"),
                          index=False)
    pd.DataFrame({"metric": ["ARI", "NMI", "homogeneity", "completeness",
                             "v_measure", "merged_accuracy", "merged_ARI",
                             "merged_NMI", "n_scored", "frac_inside", "ap_um",
                             "overlap", "n_ccf_divisions"],
                  "value": [fs["ari"], fs["nmi"], fs["homogeneity"],
                            fs["completeness"], fs["v_measure"],
                            fs["merged_accuracy"], fs["merged_ari"],
                            fs["merged_nmi"], int(mask.sum()),
                            float(mask.mean()), res["ap_um"], res["overlap"],
                            len(set(ref[mask]))]}
                 ).to_csv(os.path.join(a.out_dir, "metrics_vs_allen.csv"),
                          index=False)
    _overlay_plot(grid, sheaf, ref, res, a.out_dir, fs)
    print(f"\nWrote outputs to {a.out_dir}/  "
          f"(allen_reference_labels.csv, metrics_vs_allen.csv, allen_overlay.png)")
    print("!! Eyeball allen_overlay.png: trust ARI/NMI only if the middle and "
          "right panels align anatomically.")


def _overlay_plot(grid, sheaf, ref, res, out_dir, fs):
    fig, ax = plt.subplots(1, 3, figsize=(22, 8))
    uniq = sorted(set(sheaf))
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 2))
    for i, c in enumerate(uniq):
        m = sheaf == c
        ax[0].scatter(grid[m, 0], grid[m, 1], s=3, color=cmap(i))
    ax[0].set_title(f"Sheaf domains (k={len(uniq)})")
    ax[0].set_aspect("equal"); ax[0].invert_yaxis()

    divs = sorted(set(ref))
    dcm = plt.cm.get_cmap("tab20", max(len(divs), 2))
    for i, dname in enumerate(divs):
        m = ref == dname
        ax[1].scatter(grid[m, 0], grid[m, 1], s=3, color=dcm(i), label=dname)
    ax[1].set_title(f"Allen CCF divisions on grid (AP={res['ap_um']:.0f}um)")
    ax[1].set_aspect("equal"); ax[1].invert_yaxis()
    ax[1].legend(markerscale=3, fontsize=7, loc="center left",
                 bbox_to_anchor=(1.0, 0.5))

    ax[2].scatter(res["atlas_xy"][:, 0], res["atlas_xy"][:, 1], s=2,
                  color="0.8", label="atlas tissue")
    ax[2].scatter(res["tgrid"][:, 0], res["tgrid"][:, 1], s=1, color="crimson",
                  alpha=0.4, label="registered grid")
    ax[2].set_title(f"Registration check (overlap={res['overlap']:.2f})")
    ax[2].set_aspect("equal"); ax[2].invert_yaxis(); ax[2].legend(fontsize=8)

    fig.suptitle(
        f"Sheaf domains vs Allen CCF (AP={res['ap_um']:.0f}um)   |   "
        f"ARI={fs['ari']:.2f}  NMI={fs['nmi']:.2f}   "
        f"homogeneity={fs['homogeneity']:.2f}  "
        f"merged-acc={fs['merged_accuracy']:.2f}",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "allen_overlay.png"), dpi=400,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
