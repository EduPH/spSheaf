#!/usr/bin/env python
"""
Build a per-domain annotation worksheet for collaborators to assign compartment
labels themselves.

The worksheet centres on the DATA-DRIVEN evidence — each domain's size and its
top marker genes (from run_atera's `domain_markers.csv`, i.e. genes most
enriched in the domain vs the global average, no interpretation). If the
signature-enrichment outputs are present it appends the literature-signature
scores and an auto-suggested compartment, clearly flagged as *guidance only*.
Two blank columns (`collaborator_label`, `collaborator_notes`) are left for the
experts to fill in.

Usage
-----
    python make_annotation_worksheet.py --output-dir output_atera_breast
    # explicit paths:
    python make_annotation_worksheet.py \
        --labels output_atera_breast/domain_labels.csv \
        --markers output_atera_breast/domain_markers.csv \
        --scores output_atera_breast/domain_signature_scores.csv \
        --compartments output_atera_breast/domain_compartments.csv \
        --output-dir output_atera_breast
"""
import argparse
import os
import pandas as pd


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True,
                   help="run output dir (used to locate defaults + write worksheet)")
    p.add_argument("--labels", default=None)
    p.add_argument("--markers", default=None)
    p.add_argument("--scores", default=None)
    p.add_argument("--compartments", default=None)
    p.add_argument("--n-markers", type=int, default=12)
    return p


def _default(path, out_dir, name):
    return path or os.path.join(out_dir, name)


def main():
    a = build_parser().parse_args()
    labels = _default(a.labels, a.output_dir, "domain_labels.csv")
    markers = _default(a.markers, a.output_dir, "domain_markers.csv")
    scores = _default(a.scores, a.output_dir, "domain_signature_scores.csv")
    comps = _default(a.compartments, a.output_dir, "domain_compartments.csv")

    lab = pd.read_csv(labels)
    n_tot = len(lab)
    sizes = (lab["domain"].value_counts().sort_index()
             .rename("n_gridpoints").reset_index()
             .rename(columns={"index": "domain"}))
    sizes["pct_area"] = (sizes["n_gridpoints"] / n_tot * 100).round(1)

    # data-driven top markers per domain
    mk = pd.read_csv(markers)
    top = (mk.sort_values(["domain", "rank"])
           .groupby("domain")["gene"]
           .apply(lambda s: ", ".join(s.head(a.n_markers)))
           .rename("top_markers").reset_index())

    ws = sizes.merge(top, on="domain", how="left")

    # optional: signature scores (guidance only)
    if os.path.exists(scores):
        sc = pd.read_csv(scores, index_col=0)
        sc.index = [int(str(i).replace("domain ", "")) for i in sc.index]
        sc = sc.round(2).add_prefix("sig_")
        sc.index.name = "domain"
        ws = ws.merge(sc.reset_index(), on="domain", how="left")
    if os.path.exists(comps):
        cp = pd.read_csv(comps)[["domain", "compartment", "margin", "confident"]]
        cp = cp.rename(columns={"compartment": "auto_suggestion (guidance only)",
                                "margin": "auto_margin",
                                "confident": "auto_confident"})
        ws = ws.merge(cp, on="domain", how="left")

    # blank columns for the collaborators
    ws["collaborator_label"] = ""
    ws["collaborator_notes"] = ""

    out = os.path.join(a.output_dir, "domain_annotation_worksheet.csv")
    ws.to_csv(out, index=False)
    print(f"Wrote {out}  ({len(ws)} domains, {len(ws.columns)} columns)")
    print("\nColumns:", list(ws.columns))
    print("\nNOTE for collaborators:")
    print("  - `top_markers` are DATA-DRIVEN (genes most enriched in the domain "
          "vs global mean); this is the primary evidence.")
    print("  - `sig_*` / `auto_suggestion` come from literature marker "
          "signatures and are GUIDANCE ONLY, not a ground-truth label.")
    print("  - Please fill `collaborator_label` / `collaborator_notes`.")


if __name__ == "__main__":
    main()
