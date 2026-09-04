"""
Multi-slice summary plots for sheaf boundary detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_all_slices_summary(summary, output_dir=None):
    """
    Bar chart: ARI and NMI for each slice.

    Parameters
    ----------
    summary : DataFrame
        Output from :func:`~spsheaf.pipeline.run_all_slices`.
    output_dir : str or None
        Directory to save figure.
    """
    valid = summary.dropna(subset=["ari"]).copy()
    valid = valid.sort_values("ari", ascending=False).reset_index(drop=True)
    n = len(valid)

    valid["label"] = valid["slice"].apply(
        lambda s: s.replace("MERFISH_", "").replace(".h5ad", "")
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(max(12, n * 1.2), 8), sharex=True,
    )

    x = np.arange(n)
    w = 0.5

    # ARI
    ax = axes[0]
    ax.bar(x, valid["ari"], w, color="steelblue", edgecolor="black",
           linewidth=0.5)
    ax.axhline(valid["ari"].mean(), color="steelblue", ls="--", alpha=0.6,
               label=f"Mean ARI = {valid['ari'].mean():.3f}")
    for i, v in enumerate(valid["ari"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("ARI", fontsize=12)
    ax.set_title("Adjusted Rand Index — Per Slice", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")

    # NMI
    ax = axes[1]
    ax.bar(x, valid["nmi"], w, color="indianred", edgecolor="black",
           linewidth=0.5)
    ax.axhline(valid["nmi"].mean(), color="indianred", ls="--", alpha=0.6,
               label=f"Mean NMI = {valid['nmi'].mean():.3f}")
    for i, v in enumerate(valid["nmi"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("NMI", fontsize=12)
    ax.set_title("Normalized Mutual Information — Per Slice", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")

    ax.set_xticks(x)
    ax.set_xticklabels(valid["label"], rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("Slice", fontsize=11)

    fig.suptitle("Sheaf Boundary Detection — All Slices",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()

    if output_dir is not None:
        _save_fig(fig, output_dir, "all_slices_bar_chart.png")
    return fig


def plot_per_slice_comparison(summary, output_dir=None):
    """
    ARI vs NMI scatter for each slice + score distributions.

    Parameters
    ----------
    summary : DataFrame
    output_dir : str or None
    """
    valid = summary.dropna(subset=["ari"]).copy()
    n = len(valid)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ARI vs NMI scatter
    ax = axes[0]
    ax.scatter(valid["ari"], valid["nmi"],
               s=80, c="steelblue", edgecolors="black", linewidth=0.5,
               zorder=3, alpha=0.8)
    for _, r in valid.iterrows():
        label = r["slice"].replace("MERFISH_", "").replace(".h5ad", "")
        ax.annotate(label, (r["ari"], r["nmi"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xlabel("ARI", fontsize=11)
    ax.set_ylabel("NMI", fontsize=11)
    ax.set_title("ARI vs NMI per slice", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Score distributions
    ax = axes[1]
    ax.hist(valid["ari"], bins=max(5, n // 2), alpha=0.7,
            color="steelblue", edgecolor="black", label="ARI")
    ax.hist(valid["nmi"], bins=max(5, n // 2), alpha=0.5,
            color="indianred", edgecolor="black", label="NMI")
    ax.axvline(valid["ari"].mean(), color="steelblue", ls="--", alpha=0.7,
               label=f"mean ARI = {valid['ari'].mean():.3f}")
    ax.axvline(valid["nmi"].mean(), color="indianred", ls="--", alpha=0.7,
               label=f"mean NMI = {valid['nmi'].mean():.3f}")
    ax.set_xlabel("Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Score Distribution", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Sheaf Boundary Detection — Cross-Slice Analysis",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()

    if output_dir is not None:
        _save_fig(fig, output_dir, "all_slices_scatter_comparison.png")
    return fig


def _save_fig(fig, output_dir, filename):
    """Save figure to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved: {path}")
