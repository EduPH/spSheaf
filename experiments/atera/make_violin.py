#!/usr/bin/env python
"""
Stacked violin plot of marker genes vs sheaf domains for the Atera runs.

Unit of observation is the grid point (the method is segmentation-free, so there
is no per-cell matrix); each grid point carries a per-gene, z-standardized
kernel-smoothed expression value (the fiber). For each marker gene we draw one
violin per domain, so you can read which domain(s) express each marker.

Usage
-----
    python make_violin.py --run-dir output_atera_breast_fulldensity --tissue breast
    python make_violin.py --run-dir output_atera_cervical_markeraware --tissue cervical
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PANEL = {
    # colleague's breast panel A (luminal markers + spatial-pattern genes)
    "breast":   ["ESR1","PGR","ERBB2","GATA3","FOXA1","SCGB2A2","KRT5","CDK1","PLIN1","COL1A2"],
    # colleague's cervical panel B (squamous markers + spatial-pattern genes)
    "cervical": ["CDKN2A","KRT5","KRT14","KRT17","TP63","CD68","XBP1","PTPRC","DCN","GZMB"],
}

# marker genes grouped by the cancer cell type they define (for the cell-type
# annotated violin, Fig. 2f style). Genes are drawn in this order so the
# diagonal (each marker high in its own cell type) is visible.
CT_MARKERS = {
    "breast": [
        ("Luminal tumour",  ["ESR1","PGR","GATA3","FOXA1","ERBB2","XBP1"]),
        ("Basal/squamous",  ["KRT5","KRT14","KRT17","TP63","CDKN2A"]),
        ("Stroma / CAF",    ["COL1A1","COL1A2","DCN","POSTN"]),
        ("Immune",          ["PTPRC","CD68","GZMB"]),
        ("Vascular",        ["PECAM1","VWF"]),
        ("Proliferation",   ["MKI67","TOP2A","CDK1"]),
    ],
    "cervical": [
        ("Squamous tumour", ["KRT5","KRT14","KRT17","TP63","CDKN2A","SOX2"]),
        ("Stroma / CAF",    ["COL1A1","DCN","POSTN"]),
        ("Immune",          ["PTPRC","CD68","GZMB"]),
        ("Vascular",        ["PECAM1","VWF"]),
        ("Proliferation",   ["MKI67","TOP2A","CDK1"]),
    ],
}
# canonical cell-type column order (present ones kept; Unassigned last)
CT_ORDER = ["Luminal tumour","Squamous tumour","Basal/squamous","Stroma / CAF",
            "Immune","Vascular","Proliferation","Unassigned (low signal)"]


def celltype_violin(fib, genes, labels, ct_map, tissue, out, row_h=1.0):
    """Fig. 2f-style stacked violin: rows = marker genes (grouped by cell type),
    columns = cell-type annotations (domains pooled by their assigned type)."""
    gidx = {g: i for i, g in enumerate(genes)}
    # grid-point cell type via its domain
    ct_of = np.array([ct_map.get(int(l), "Unassigned (low signal)") for l in labels])
    cts = [c for c in CT_ORDER if c in set(ct_of)]
    colors = {c: plt.cm.tab10(i % 10) for i, c in enumerate(cts)}
    # gene rows in cell-type order, keeping only detected genes
    rows = [(ct, g) for ct, gl in CT_MARKERS[tissue] for g in gl if g in gidx]
    # wider columns + taller rows so each violin is easy to read
    fig, axes = plt.subplots(len(rows), 1, sharex=True,
                             figsize=(1.9*len(cts)+3.5, row_h*len(rows)+2),
                             gridspec_kw={"hspace": 0.0})
    axes = np.atleast_1d(axes)
    for r, (ct, g) in enumerate(rows):
        ax = axes[r]
        data = [fib[ct_of == c, gidx[g]] for c in cts]
        parts = ax.violinplot(data, positions=range(len(cts)), widths=0.9,
                              showmedians=True)
        for k, b in enumerate(parts["bodies"]):
            b.set_facecolor(colors[cts[k]]); b.set_alpha(0.85)
            b.set_edgecolor("grey"); b.set_linewidth(0.3)
        for key in ("cbars","cmins","cmaxes","cmedians"):
            if key in parts: parts[key].set_color("grey"); parts[key].set_linewidth(0.5)
        ax.set_ylabel(g, rotation=0, ha="right", va="center", fontsize=11, style="italic")
        ax.axhline(0, color="k", lw=0.3, alpha=0.4); ax.set_yticks([]); ax.margins(y=0.08)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.set_xlim(-0.6, len(cts) - 0.4)
    axes[-1].set_xticks(range(len(cts)))
    axes[-1].set_xticklabels(cts, rotation=30, ha="right", fontsize=11)
    axes[-1].set_xlabel("inferred cancer cell type (annotation)", fontsize=11)
    fig.suptitle(f"Atera {tissue}: marker expression by cell-type annotation "
                 f"(z-scored fiber)", y=0.998, fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=400, bbox_inches="tight"); plt.close(fig)
    print(f"-> {out}  ({len(rows)} genes x {len(cts)} cell types)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tissue", default="breast", choices=["breast","cervical"])
    ap.add_argument("--genes", default=None, help="comma list overriding the default panel")
    ap.add_argument("--out", default=None)
    ap.add_argument("--row-height", type=float, default=1.0,
                    help="inches per gene row in the cell-type violin (raise to see each violin bigger)")
    a = ap.parse_args()

    d = np.load(os.path.join(a.run_dir, "fibers.npz"), allow_pickle=True)
    fib = d["fib"].astype(float); genes = list(d["genes"])
    labels = np.load(os.path.join(a.run_dir, "labels.npy"))
    gidx = {g: i for i, g in enumerate(genes)}
    panel = (a.genes.split(",") if a.genes else PANEL[a.tissue])
    panel = [g for g in panel if g in gidx]
    doms = sorted(set(labels))
    pal = [plt.cm.tab10(k % 10) for k in range(len(doms))]

    # cell-type-annotated violin (Fig. 2f style), if the assignment exists
    ct_csv = os.path.join(a.run_dir, "domain_celltypes.csv")
    if os.path.exists(ct_csv):
        cc = pd.read_csv(ct_csv)
        ct_map = dict(zip(cc.domain.astype(int), cc.cell_type))
        celltype_violin(fib, genes, labels, ct_map, a.tissue,
                        os.path.join(a.run_dir, "violin_markers_celltype.png"),
                        row_h=a.row_height)

    fig, axes = plt.subplots(len(panel), 1, figsize=(1.0*len(doms)+2, 1.05*len(panel)+1),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for r, g in enumerate(panel):
        ax = axes[r]
        data = [fib[labels == dd, gidx[g]] for dd in doms]
        parts = ax.violinplot(data, positions=range(len(doms)), showmedians=True,
                              widths=0.85)
        for k, b in enumerate(parts["bodies"]):
            b.set_facecolor(pal[k]); b.set_alpha(0.8); b.set_edgecolor("grey"); b.set_linewidth(0.4)
        for key in ("cbars","cmins","cmaxes","cmedians"):
            if key in parts: parts[key].set_color("grey"); parts[key].set_linewidth(0.6)
        ax.set_ylabel(g, rotation=0, ha="right", va="center", fontsize=9)
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
        ax.set_yticks([]); ax.margins(y=0.05)
    axes[-1].set_xticks(range(len(doms)))
    axes[-1].set_xticklabels([f"D{dd}" for dd in doms])
    axes[-1].set_xlabel("sheaf domain")
    fig.suptitle(f"Atera {a.tissue}: marker expression per domain "
                 f"(z-scored fiber; violin per domain)", y=0.995)
    fig.tight_layout()
    out = a.out or os.path.join(a.run_dir, "violin_markers.png")
    fig.savefig(out, dpi=400, bbox_inches="tight"); plt.close(fig)
    print(f"-> {out}  ({len(panel)} genes x {len(doms)} domains)")


if __name__ == "__main__":
    main()
