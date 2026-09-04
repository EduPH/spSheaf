"""
Diagnostic plots for sheaf boundary detection on a single slice.

Produces a 2×2 figure:
  Row 0: Ground truth | Sheaf boundary prediction
  Row 1: Raw boundary score | Smoothed boundary score
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from ..evaluation import hungarian_match_labels


def plot_single_slice_results(grid_coords, labels_sheaf,
                              gt_int, gt_names, gt_valid,
                              boundary_raw, boundary_smooth,
                              ari_sheaf, nmi_sheaf,
                              slice_name="", output_dir=None):
    """
    Diagnostic 2×2 figure for a single slice.

    Parameters
    ----------
    grid_coords : ndarray, shape (N, 2)
    labels_sheaf : ndarray, shape (N,)
    gt_int : ndarray, shape (N,)
    gt_names : list of str
    gt_valid : ndarray of bool, shape (N,)
    boundary_raw, boundary_smooth : ndarray, shape (N,)
    ari_sheaf, nmi_sheaf : float
    slice_name : str
    output_dir : str or None
        Directory to save figure. If None, displays interactively.

    Returns
    -------
    fig : matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    n_gt = len(gt_names)

    # --- Ground truth ---
    ax = axes[0, 0]
    cmap_gt = plt.cm.get_cmap("tab10", max(n_gt, 2))
    for c in range(n_gt):
        mask = gt_int == c
        ax.scatter(
            grid_coords[mask, 0], grid_coords[mask, 1],
            s=2, alpha=0.5, color=cmap_gt(c), label=gt_names[c],
        )
    ax.set_aspect("equal")
    ax.set_title("Ground truth domains", fontsize=11)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.legend(markerscale=5, fontsize=5, loc="best", ncol=2)

    # --- Sheaf boundary (Hungarian-matched colors) ---
    ax = axes[0, 1]
    n_k = len(set(labels_sheaf))
    cmap_k = plt.cm.get_cmap("tab10", max(n_k, n_gt, 2))
    labels_matched = hungarian_match_labels(labels_sheaf, gt_int, gt_valid)
    for c in range(max(n_k, n_gt)):
        mask = labels_matched == c
        if mask.sum() > 0:
            ax.scatter(
                grid_coords[mask, 0], grid_coords[mask, 1],
                s=2, alpha=0.5, color=cmap_k(c),
            )
    ax.set_aspect("equal")
    ax.set_title(
        f"Sheaf boundary detection\nARI={ari_sheaf:.3f}, NMI={nmi_sheaf:.3f}",
        fontsize=11,
    )
    ax.set_xlabel("x (μm)")

    # --- Raw boundary score ---
    ax = axes[1, 0]
    vmax_raw = np.percentile(boundary_raw, 95)
    sc = ax.scatter(
        grid_coords[:, 0], grid_coords[:, 1],
        c=boundary_raw, s=2, cmap="hot",
        vmax=vmax_raw if vmax_raw > 0 else 1,
    )
    ax.set_aspect("equal")
    ax.set_title("Raw sheaf boundary score", fontsize=10)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    plt.colorbar(sc, ax=ax, shrink=0.7)

    # --- Smoothed boundary score ---
    ax = axes[1, 1]
    vmax_sm = np.percentile(boundary_smooth, 95)
    sc = ax.scatter(
        grid_coords[:, 0], grid_coords[:, 1],
        c=boundary_smooth, s=2, cmap="hot",
        vmax=vmax_sm if vmax_sm > 0 else 1,
    )
    ax.set_aspect("equal")
    ax.set_title("Smoothed boundary score\n(Gaussian σ = 60 μm)", fontsize=10)
    ax.set_xlabel("x (μm)")
    plt.colorbar(sc, ax=ax, shrink=0.7)

    title = "Sheaf Boundary Detection — Domain Inference"
    if slice_name:
        title += f"\n{slice_name}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        safe_name = slice_name.replace("/", "_").replace(".", "_")
        path = os.path.join(output_dir, f"sheaf_boundary_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → Saved: {path}")

    return fig