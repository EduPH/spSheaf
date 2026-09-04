#!/usr/bin/env python
"""
Signature-enrichment analysis of Atera sheaf domains (no ground truth needed).

For each inferred domain, score its expression against canonical tissue-
compartment signatures (tumour epithelium / stroma-CAF / immune / vascular /
proliferation), assign the best-matching compartment, and quantify how cleanly
the domains partition into compartments. This turns the qualitative marker story
into a reproducible figure + numbers for a dataset that has no manual
annotation.

It re-derives each domain's mean expression by streaming the transcripts parquet
once (grid points -> nearest, per transcript), so it needs only the parquet and
the run's ``domain_labels.csv``.

Outputs (in --output-dir):
  domain_signature_scores.csv   z-scored signature score per domain
  domain_compartments.csv       assigned compartment + top signature margin
  signature_heatmap.png         domain x signature heatmap
  compartment_map.png           spatial map recoloured by compartment
  signature_metrics.csv         summary (mean assignment margin, n confident)

Usage
-----
    python signature_enrichment.py \
        --input breast/transcripts.parquet \
        --labels output_atera_breast/domain_labels.csv \
        --output-dir output_atera_breast
"""
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Canonical tumour-microenvironment signatures (human gene symbols). Stroma /
# immune / vascular / proliferation are shared; the tumour-epithelium set is
# tissue-specific (breast = luminal, cervical = squamous).
_COMMON = {
    "Stroma / CAF":    ["COL1A1", "COL1A2", "COL3A1", "COL6A1", "COL6A2",
                        "POSTN", "FN1", "PDGFRB", "PDGFRA", "ACTA2", "TAGLN",
                        "FAP", "THBS1", "AEBP1", "LUM", "DCN"],
    "Immune":          ["PTPRC", "CD3D", "CD3E", "CD8A", "CD68", "CD14",
                        "LYZ", "MS4A1", "CD79A", "IGKC", "IGHG1", "CCL5",
                        "HLA-DRA", "CXCR4"],
    "Vascular / endo": ["PECAM1", "VWF", "CLDN5", "CDH5", "EGFL7", "PLVAP",
                        "FLT1", "KDR"],
    "Proliferation":   ["MKI67", "TOP2A", "CCNB1", "PCNA", "BIRC5", "UBE2C",
                        "CENPF", "CDK1"],
}
_TUMOUR = {
    "breast":   ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "GATA3", "FOXA1",
                 "ESR1", "ERBB2", "MUC1", "ELF3", "XBP1", "TFF1", "TFF3",
                 "AGR2", "AGR3"],
    "cervical": ["EPCAM", "KRT5", "KRT14", "KRT6A", "KRT17", "TP63", "SOX2",
                 "S100A8", "S100A9", "SERPINB3", "SERPINB4", "KRT15", "DSP",
                 "CDH1"],
    "generic":  ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1", "KRT5", "KRT14",
                 "KRT17", "ELF3", "CLDN4", "CLDN7"],
}


def signatures_for(tissue):
    sig = {"Tumour epithelium": _TUMOUR.get(tissue, _TUMOUR["generic"])}
    sig.update(_COMMON)
    return sig


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="transcripts.parquet")
    p.add_argument("--labels", required=True, help="domain_labels.csv from the run")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--gene-col", default="feature_name")
    p.add_argument("--x-col", default="x_location")
    p.add_argument("--y-col", default="y_location")
    p.add_argument("--qv-col", default="qv")
    p.add_argument("--min-qv", type=float, default=20.0)
    p.add_argument("--chunk-rows", type=int, default=5_000_000)
    p.add_argument("--margin", type=float, default=0.5,
                   help="min z-score gap between top-2 signatures to call a "
                        "domain 'confident' (default 0.5).")
    p.add_argument("--tissue", default="breast",
                   choices=["breast", "cervical", "generic"],
                   help="tumour-epithelium signature set (default breast).")
    return p


def domain_expression(a, grid, labels, sig_genes):
    """Stream the parquet, accumulate summed expression of signature genes per
    domain (each transcript -> nearest grid point -> that domain)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(a.input)
    names = pf.schema.names
    read = [a.gene_col, a.x_col, a.y_col] + ([a.qv_col] if a.qv_col in names else [])
    tree = cKDTree(grid)
    n_dom = int(labels.max()) + 1
    genes = sorted(sig_genes)
    gidx = {g: i for i, g in enumerate(genes)}
    counts = np.zeros((n_dom, len(genes)), dtype=np.float64)
    dom_totals = np.zeros(n_dom, dtype=np.float64)  # total transcripts per domain
    seen = 0
    for batch in pf.iter_batches(batch_size=a.chunk_rows, columns=read):
        ch = batch.to_pandas()
        if a.qv_col in ch.columns:
            ch = ch[ch[a.qv_col] >= a.min_qv]
        gname = ch[a.gene_col].astype(str)
        _, j = tree.query(ch[[a.x_col, a.y_col]].to_numpy(), k=1)
        dom = labels[j]
        np.add.at(dom_totals, dom, 1)
        m = gname.isin(gidx)
        if m.any():
            gi = gname[m].map(gidx).to_numpy()
            np.add.at(counts, (dom[m.to_numpy()], gi), 1)
        seen += len(ch)
    return genes, counts, dom_totals


def main():
    a = build_parser().parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    lab = pd.read_csv(a.labels)
    grid = lab[["x", "y"]].to_numpy()
    labels = lab["domain"].to_numpy()

    SIGNATURES = signatures_for(a.tissue)
    all_sig_genes = sorted({g for v in SIGNATURES.values() for g in v})
    genes, counts, totals = domain_expression(a, grid, labels, all_sig_genes)

    dom_order = sorted(set(labels))
    # per-domain, per-gene rate (CPM-like), log-stabilised
    rate = np.log1p(counts[dom_order] / np.maximum(totals[dom_order, None], 1) * 1e4)
    gi = {g: i for i, g in enumerate(genes)}
    # cache the rate matrix so signatures can be re-scored without re-streaming
    pd.DataFrame(rate, index=[f"domain {d}" for d in dom_order], columns=genes
                 ).to_csv(os.path.join(a.output_dir, "domain_gene_rate.csv"))

    # PROPER signature score: z-score EACH gene across domains (equal weight,
    # abundance-independent), then average the z-scores within each signature.
    Z = (rate - rate.mean(axis=0)) / (rate.std(axis=0) + 1e-9)
    sig_names = list(SIGNATURES)
    S = np.zeros((len(dom_order), len(sig_names)))
    for si, s in enumerate(sig_names):
        # only average over signature genes actually detected (nonzero variance);
        # absent genes would otherwise dilute the score toward zero.
        cols = [gi[g] for g in SIGNATURES[s]
                if g in gi and rate[:, gi[g]].std() > 0]
        if cols:
            S[:, si] = Z[:, cols].mean(axis=1)

    scores = pd.DataFrame(S, index=[f"domain {d}" for d in dom_order],
                          columns=sig_names)
    scores.to_csv(os.path.join(a.output_dir, "domain_signature_scores.csv"))

    # assign compartment + confidence margin
    order = np.argsort(-S, axis=1)
    top = [sig_names[order[i, 0]] for i in range(len(dom_order))]
    margin = S[np.arange(len(dom_order)), order[:, 0]] - \
        S[np.arange(len(dom_order)), order[:, 1]]
    comp = pd.DataFrame({"domain": dom_order, "compartment": top,
                         "margin": margin.round(3),
                         "confident": margin >= a.margin})
    comp.to_csv(os.path.join(a.output_dir, "domain_compartments.csv"), index=False)

    n_conf = int(comp.confident.sum())
    pd.DataFrame([{"n_domains": len(dom_order), "n_confident": n_conf,
                   "frac_confident": round(n_conf / len(dom_order), 3),
                   "mean_margin": round(float(margin.mean()), 3),
                   "compartments_used": comp.compartment.nunique()}]
                 ).to_csv(os.path.join(a.output_dir, "signature_metrics.csv"),
                          index=False)

    _heatmap(scores, comp, a.output_dir)
    _compartment_map(lab, comp, a.output_dir)

    print("Per-domain compartment assignment:")
    print(comp.to_string(index=False))
    print(f"\nConfident (margin>={a.margin}): {n_conf}/{len(dom_order)}  "
          f"mean margin {margin.mean():.2f}")
    print(f"-> {a.output_dir}/ (signature_heatmap.png, compartment_map.png, "
          "domain_compartments.csv, signature_metrics.csv)")


def _heatmap(scores, comp, out_dir):
    fig, ax = plt.subplots(figsize=(1.4 * scores.shape[1] + 3,
                                    0.42 * scores.shape[0] + 2))
    im = ax.imshow(scores.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(scores.shape[1])); ax.set_xticklabels(scores.columns,
                                                              rotation=30, ha="right")
    ax.set_yticks(range(scores.shape[0]))
    ax.set_yticklabels([f"{r}  → {c}" for r, c in
                        zip(scores.index, comp.compartment)])
    for i in range(scores.shape[0]):
        j = int(np.argmax(scores.values[i]))
        ax.text(j, i, "★", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7, label="signature score (z)")
    ax.set_title("Domain × compartment-signature enrichment")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "signature_heatmap.png"), dpi=400,
                bbox_inches="tight")
    plt.close(fig)


def _compartment_map(lab, comp, out_dir):
    m = dict(zip(comp.domain, comp.compartment))
    lab = lab.copy(); lab["compartment"] = lab["domain"].map(m)
    comps = sorted(lab["compartment"].dropna().unique())
    palette = {c: plt.cm.tab10(i) for i, c in enumerate(comps)}
    fig, ax = plt.subplots(figsize=(11, 7))
    for c in comps:
        d = lab[lab.compartment == c]
        ax.scatter(d.x, d.y, s=4, color=palette[c], label=c)
    ax.set_aspect("equal"); ax.invert_yaxis()
    ax.legend(markerscale=3, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("Sheaf domains recoloured by inferred compartment")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compartment_map.png"), dpi=400,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
