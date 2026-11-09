"""Copy the result tables the manuscript is built from into ``reported/``.

``data/`` is deliberately untracked: it holds raw molecule tables, caches and
per-cell dumps that are far too large for git, and it lives outside the
repository on most machines. That left an asymmetry -- every competitor's
numbers were versioned under ``benchmarks/segfree/results/``, while spSheaf's
own were not -- so a clone could rebuild the competitor columns of the comparison table but
not ours, and the figures that read the sweeps could not be drawn at all.

``reported/`` closes that gap. It holds only small summary tables: one row per
configuration or per slice, no per-cell arrays. Each entry below records what
the table supports in the manuscript, so a reader can go from a number to the
file it came from and from the file to the command that wrote it.

    python tools/collect_reported.py            # refresh from the results root
    python tools/collect_reported.py --check    # report drift, write nothing
    python tools/collect_reported.py --readme   # regenerate reported/README.md

Nothing here is a substitute for re-running: ``make`` targets named in the
README produce these files from the raw data.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path

from spsheaf import paths

HERE = Path(__file__).resolve().parent.parent
DEST = HERE / "reported"
#: --tissue: the same manifest collected from a results root whose runs were
#: made at the tissue reference (the package default since 28 Sep 2026), into
#: reported/tissue/. reported/ itself holds the local-reference runs, which
#: the MERFISH benchmark numbers in the paper are drawn from.
TISSUE = False

#: (destination subpath, source glob under the results root, what it supports)
MANIFEST: list[tuple[str, str, str]] = [
    # ---- Table 1: MERFISH, five slices, 20 k-means seeds -------------
    ("merfish/seed_sensitivity_per_slice.csv",
     "benchmark_sdmbench/output_seed_sensitivity/seed_sensitivity_per_slice.csv",
     "Table 1 and ED Fig. 1c: per-slice ARI, mean over 20 seeds"),
    ("merfish/seed_sensitivity_summary.csv",
     "benchmark_sdmbench/output_seed_sensitivity/seed_sensitivity_summary.csv",
     "Table 1: the seed spread quoted in the caption"),
    ("merfish/seed_sensitivity_per_run.csv",
     "benchmark_sdmbench/output_seed_sensitivity/seed_sensitivity_per_run.csv",
     "Table 1: every (slice, seed) run behind the means"),
    ("merfish/all_slices_results.csv",
     "merfish/output_merfish/all_slices_results.csv",
     "Results 2.2: the five-slice run at the MERFISH configuration"),
    ("merfish/loo_cv_table.csv", "merfish/loo_cv_table.csv",
     "Methods (selection rules): leave-one-slice-out selection of (sigma_s, alpha_b, alpha_s)"),
    ("merfish/multislice.json", "merfish/panels/multislice.json",
     "Supplementary Fig. 2: cross-slice label agreement (69.8 %, 63.1-78.5 %) "
     "and the self-alignment bound (91.5 %); `make panels-multislice`"),
    ("osmfish/tune_cv.json", "osmfish/tune_cv.json",
     "osmFISH leave-one-spatial-block-out (not in the manuscript since 12 Sep 2026; "
     "kept as the record of the check)"),
    ("merfish/loo_cv_table_union.csv", "merfish/loo_cv_table_union.csv",
     "Table 1 and Fig. 2b: the 27-configuration grid at both coordinate "
     "references (54 rows per slice), the table the three selection rules are "
     "applied to (`make matched-selection-union`)"),
    ("merfish/k_selection_*.csv", "merfish/k_selection_MERFISH_*.csv",
     "Choosing k (Results, ED k-selection figure): silhouette, grid-phase "
     "stability, the seed control and ARI against the annotation for k = 2-16 "
     "on each of the five slices, at the MERFISH defaults"),
    ("merfish/k_split_events.csv", "merfish/k_split_events.csv",
     "Structure below the annotation, Supplementary Table 4: every division of "
     "an annotated domain at k = 8, 9, 10 on the five slices (`make merfish-k-split`)"),
    ("merfish/k_split_pairs.csv", "merfish/k_split_pairs.csv",
     "Fig. 4 (ksplit): correlation of the per-gene difference vectors between "
     "divisions of the same domain on different slices"),
    ("merfish/k_split_across_k.csv", "merfish/k_split_across_k.csv",
     "Whether a division at k = 9 is inherited from k = 8 or persists to k = 10"),
    ("merfish/graph_k.csv", "merfish/graph_k.csv",
     "Not in the paper (private notes, experiments_not_in_paper.md): k_graph = 4, 6, 8 on the five slices over three seeds; "
     "the square lattice's tie-break does not matter (`make merfish-graph-k`)"),
    ("osmfish/output_osmFISH_transfer/osmfish_metrics.csv", "osmfish/osmfish_metrics_transfer.csv",
     "Extended Data Table 1, osmFISH row: the MERFISH development configuration carried over unchanged "
     "(h = 385 px, sigma_s = 923 px, d = 6, alpha_s = 0.5, local reference; `_staging/run_osmfish_transfer.sh`)"),
    ("merfish/stalk_angles_summary.csv", "merfish/stalk_angles_summary.csv",
     "Methods 4.5: principal angles between adjacent stalks and the distance of the aligning rotation from the identity, "
     "five slices at the reference configuration, with a random-neighbourhood null (`make merfish-stalk-angles`)"),
    ("merfish/k_split_celltypes.csv", "merfish/k_split_celltypes.csv",
     "Fig. 4e: the Moffitt neuronal clusters in the two BST halves at k = 9, per "
     "slice and cluster (counts, odds ratio, Fisher q); `make merfish-k-split`"),
    ("merfish/k_split_celltypes_pairs.csv", "merfish/k_split_celltypes_pairs.csv",
     "Results 2.7: cross-slice correlation of the per-cluster log odds (0.94 for 0.19 vs 0.24)"),
    ("merfish/k_split_celltypes_summary.csv", "merfish/k_split_celltypes_summary.csv",
     "Methods: cells per half and cells dropped, per slice"),
    ("merfish/k_split_celltypes_markers.csv", "merfish/k_split_celltypes_markers.csv",
     "Results 2.7: cell-level marker genes of the enriched clusters and of the halves"),
    ("merfish/reference_field.csv", "merfish/reference_field.csv",
     "Results 2.3 and Methods: what the coordinate reference does to the boundary field -- the "
     "localization/partition dissociation, and the field's width after the "
     "median normalization the weighting applies"),

    # ---- Table 1, spSheaf columns ------------------------------------
    ("osmfish/osmfish_metrics.csv",
     "osmfish/output_osmFISH_boundary/osmfish_metrics.csv",
     "ED Table 1, osmFISH transferred row"),
    ("osmfish/osmfish_sweep.csv",
     "osmfish/output_osmFISH_boundary/osmfish_sweep.csv",
     "Table 1, osmFISH row, and ED Fig. 3 row 1 (the 108+24 grid)"),
    ("osmfish/osmfish_metrics_tissue_reference.csv",
     "osmfish/tissue_reference/osmfish_metrics.csv",
     "osmFISH at the local-reference scales with the reference switched to tissue (0.588; not in the manuscript -- Table 1's osmFISH value is the tissue-reference argmax in osmfish_sweep.csv)"),
    ("spacetx/MERFISH_Allen_Prototype_metrics.csv",
     "spacetx/MERFISH_Allen_Prototype/metrics.csv",
     "ED Table 1, Allen transferred row (one row per k-means seed)"),
    ("spacetx/MERFISH_Allen_Prototype_sweep.csv",
     "spacetx/MERFISH_Allen_Prototype/sweep.csv",
     "Table 1, Allen row, and ED Fig. 3 row 2 (the 162 grid)"),
    ("spacetx/BARISTASeq_metrics.csv", "spacetx/BARISTASeq/metrics.csv",
     "ED Table 1, BaristaSeq transferred row"),
    ("spacetx/BARISTASeq_sweep.csv", "spacetx/BARISTASeq/sweep.csv",
     "Table 1, BaristaSeq row, and ED Fig. 3 row 3 (the 162 grid)"),
    ("spacetx/sheaf_ablation_*.csv", "spacetx/sheaf_ablation_*.csv",
     "Supplementary Fig. 3: the ten-arm edge-score ablation on each SpaceTx "
     "section at its selected configuration (`make spacetx-ablation`; tissue "
     "reference; the `_merfishscales` files are the same arms at h 25, sigma_s 60, "
     "not in the paper, see the private notes) (tissue "
     "reference, so present under reported/tissue/)"),

    # ---- why osmFISH is the one dataset we do not lead ---------------
    ("osmfish/resolution.csv", "osmfish/resolution.csv",
     "Not in the submitted version (docs/notes/osmfish_diagnostics_unreported.md, private repo): does domain WIDTH predict which domains are lost"),
    ("osmfish/elongation.csv", "osmfish/elongation.csv",
     "Not in the submitted version: are ELONGATED domains lost, and is alpha_s the cause"),
    ("osmfish/alpha_sweep.csv", "osmfish/alpha_sweep.csv",
     "Not in the submitted version: ARI and NMI at alpha_s in {0, 0.25, 0.5, 0.75}. "
     "Written by elongation.py only since 11 Sep 2026 -- it used to print these four "
     "numbers and save nothing, so re-run `make osmfish-elongation` if the copy here "
     "predates that"),
    ("osmfish/tune_attribute.csv", "osmfish/tune_attribute.csv",
     "Not in the submitted version: which transferred change costs the accuracy (8 runs)"),
    ("osmfish/seed_sensitivity.csv", "osmfish/seed_sensitivity.csv",
     "Table 1 caption: the osmFISH seed spread"),

    # ---- reviewer-memo experiments (MERFISH, all five slices) --------
    ("merfish/boundary_eval_*.csv", "merfish/boundary_eval_MERFISH_*.csv",
     "memo 3: the boundary field scored directly, without clustering"),
    ("merfish/sheaf_ablation_*.csv", "merfish/sheaf_ablation_MERFISH_*.csv",
     "memo 2: sheaf against the simpler local-subspace controls"),
    ("merfish/k_sensitivity_*.csv", "merfish/k_sensitivity_MERFISH_*.csv",
     "memo 5: k swept under label-free and label-dependent criteria"),

    # ---- the non-cortex applications --------------------------------
    ("brain_receptor/metrics.csv", "brain_receptor/output_brain_receptor/metrics.csv",
     "ED Fig. 5 (scale): the receptor-panel section"),
    ("brain_receptor/metrics_vs_allen.csv",
     "brain_receptor/output_brain_receptor/metrics_vs_allen.csv",
     "ED Fig. 5: agreement with the Allen reference annotation"),
    ("atera/breast_metrics.csv", "atera/output_atera_breast/metrics.csv",
     "ATERA breast section (not in the manuscript)"),
    ("atera/breast_signature_metrics.csv",
     "atera/output_atera_breast/signature_metrics.csv",
     "ATERA breast, signature scores per domain (not in the manuscript)"),
    ("atera/cervical_metrics.csv", "atera/output_atera_cervical/metrics.csv",
     "ATERA cervical section (not in the manuscript)"),
    ("atera/cervical_signature_metrics.csv",
     "atera/output_atera_cervical/signature_metrics.csv",
     "ATERA cervical, signature scores per domain (not in the manuscript)"),
    ("pecab/Rep1_summary.csv", "pecab/output_pecab_signaling_Rep1/summary.csv",
     "ED Fig. 8 (PECAb): replicate 1"),
    ("pecab/Rep1_marker_corr.csv", "pecab/output_pecab_signaling_Rep1/marker_corr.csv",
     "ED Fig. 8: marker correlations, replicate 1"),
    ("pecab/Rep1_emt_concordance.csv",
     "pecab/output_pecab_signaling_Rep1/emt_concordance.csv",
     "ED Fig. 8: EMT concordance"),
    ("pecab/Rep2_summary.csv", "pecab/output_pecab_signaling_Rep2/summary.csv",
     "ED Fig. 8: replicate 2"),
    ("pecab/Rep2_marker_corr.csv", "pecab/output_pecab_signaling_Rep2/marker_corr.csv",
     "ED Fig. 8: the r = 0.94 cross-replicate agreement"),
    ("pecab/edge_profiles.csv", "pecab/edge_profiles.csv",
     "ED Fig. 8g,h: EMT score and field against the signed distance to the mesenchymal "
     "compartment's edge, both replicates (`make pecab-edge`)"),
    ("pecab/edge_summary.csv", "pecab/edge_summary.csv",
     "ED Fig. 8 caption: the EMT change across the edge, its logistic width, the field at the edge"),

    # ---- can k be chosen without an annotation? ---------------------
    ("synthetic/k_selection_*.csv", "synthetic/k_selection_*.csv",
     "Choosing k: the annotation-free criteria against k on synthetic tissue "
     "of known K, two designs x two data seeds (figures/ed_kselection.py)"),

    # ---- nuclear architecture (ED Figs 6 and 7) ---------------------
    ("takei/zone_mark_heatmap.csv", "takei/output_takei_if/zone_mark_heatmap.csv",
     "ED Fig. 7b: zone-by-mark composition, row-centred"),
    ("takei/zone_archetype_sizes.csv", "takei/output_takei_if/zone_archetype_sizes.csv",
     "ED Fig. 7b: how many zones fall in each archetype"),
    ("takei/zone_archetype_assignment.csv",
     "takei/output_takei_if/zone_archetype_assignment.csv",
     "ED Fig. 7b: (nucleus, zone) -> archetype, the assignment behind the panel"),
    ("takei/if_certification_summary.csv",
     "takei/output_takei_if/certification_summary.csv",
     "ED Fig. 7: the IF certification summary"),
    ("takei/if_certification_zone_table.csv",
     "takei/output_takei_if/certification_zone_table.csv",
     "ED Fig. 7: per-zone certification"),
    ("takei/dna_certification_per_nucleus.csv",
     "takei/output_takei_feat_full1/certification_per_nucleus.csv",
     "ED Fig. 6 (DNA territories): per-nucleus certification"),
    ("takei/dna_certification_radial.csv",
     "takei/output_takei_feat_full1/certification_radial.csv",
     "ED Fig. 6: the radial profile"),
    ("takei/dna_radial_vs_genedensity.csv",
     "takei/output_takei_feat_full1/radial_vs_genedensity.csv",
     "ED Fig. 6: radial position against gene density"),
    ("takei/dna_summary.csv", "takei/output_takei_feat_full1/summary.csv",
     "ED Fig. 6: the run summary"),
    ("takei/output_takei_feat_full2/certification_per_nucleus.csv", "takei/dna_certification_per_nucleus_rep2.csv",
     "ED Fig. 6: per-nucleus certification, replicate 2 (200 nuclei)"),
    ("takei/output_takei_feat_full2/certification_radial.csv", "takei/dna_certification_radial_rep2.csv",
     "ED Fig. 6a: the radial profile, replicate 2"),
    ("takei/output_takei_feat_full2/summary.csv", "takei/dna_summary_rep2.csv",
     "ED Fig. 6: the run summary, replicate 2"),
    ("takei/dna_radial_replicates.csv", "takei/dna_radial_replicates.csv",
     "ED Fig. 6a: per-chromosome mean radial position in both replicates (compare_replicates.py)"),
    ("takei/dna_replicates_summary.csv", "takei/dna_replicates_summary.csv",
     "ED Fig. 6a caption: Pearson r = 0.91 between replicates, NMI per replicate, chr19 most interior in both"),
]

CAP_BYTES = 4 * 1024 * 1024          # a summary table, not a per-cell dump

#: destination directory -> the make target that produces the tables in it
PRODUCED_BY = {
    "merfish": "make merfish merfish-seeds merfish-loo-cv revision",
    "osmfish": "make osmfish osmfish-all osmfish-insample",
    "spacetx": "make spacetx-all",
    "brain_receptor": "python -m experiments.brain_receptor.run_brain_receptor",
    "atera": "python -m experiments.atera.run_atera",
    "pecab": "make panels-pecab",
    "takei": "python -m experiments.takei.make_zone_heatmaps (and takei.run_takei)",
    "synthetic": "make k-selection",
}

README_HEAD_TISSUE = """<!-- generated by tools/collect_reported.py --tissue --readme; do not edit by hand -->
# reported/tissue/ -- the same tables, from the runs at the tissue reference

The coordinate reference of the stalk sections is a tuned parameter with two
values. `reported/` holds the runs at the local reference, which the tuning
grid selects on the MERFISH benchmark and which the MERFISH numbers of the
paper are drawn from. This directory holds the runs at the tissue reference,
the package default, made with `SPSHEAF_REFERENCE=tissue` and a separate
results root: Table 1's tissue row, the osmFISH and SpaceTx argmaxes, the
ablation at the default (Discussion; Supplementary Fig. 3) and the k sweeps
repeated at the default. Only the experiments that were repeated are here.

Refresh or check with

    SPSHEAF_DATA=<tissue results root> python tools/collect_reported.py --tissue
    SPSHEAF_DATA=<tissue results root> python tools/collect_reported.py --tissue --check
"""

README_HEAD = """<!-- generated by tools/collect_reported.py --readme; do not edit by hand -->
# reported/ -- the tables behind every number in the manuscript

`data/` is untracked: it holds raw molecule tables, caches and per-cell dumps,
and on most machines it sits outside the repository altogether. This directory
is the small, versioned part of it -- one row per configuration or per slice,
nothing per-cell -- so that a clone can check any figure or table against the
numbers it was drawn from, and rebuild the figures that read the parameter
sweeps without first re-running days of compute.

The competitors' tables are NOT here; they live where they always did, under
`benchmarks/segfree/results/`, alongside the per-cell label CSVs that the
figures Hungarian-match. The two places together are the provenance of Table 1.

Refresh after a run, or check for drift, with

    python tools/collect_reported.py
    python tools/collect_reported.py --check

Every file is a copy. The commands under each heading are what actually produce
it; nothing here should be edited by hand.
"""


def write_readme():
    """reported/README.md, from the manifest itself, so it cannot drift."""
    groups = {}
    for dst, src, why in MANIFEST:
        if TISSUE and not list(DEST.glob(dst)):
            continue                      # only what was repeated at the default
        groups.setdefault(Path(dst).parts[0], []).append((dst, src, why))
    lines = [README_HEAD_TISSUE if TISSUE else README_HEAD]
    for g in sorted(groups):
        lines.append(f"\n## {g}/\n")
        if g in PRODUCED_BY:
            lines.append(f"Produced by `{PRODUCED_BY[g]}`.\n")
        lines.append("| file | supports | copied from (under the results root) |")
        lines.append("| --- | --- | --- |")
        for dst, src, why in groups[g]:
            name = Path(dst).name
            lines.append(f"| `{name}` | {why} | `{src}` |")
        lines.append("")
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {(DEST / 'README.md').relative_to(HERE)} "
          f"({len(MANIFEST)} entries in {len(groups)} groups)")


def resolve(entry):
    """(destination, source) pairs for one manifest entry, globs expanded."""
    dst, src, _ = entry
    root = paths.results()
    if "*" not in src:
        return [(DEST / dst, root / src)]
    out = []
    for p in sorted(root.glob(src)):
        out.append((DEST / Path(dst).parent / p.name, p))
    return out


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--check", action="store_true",
                   help="report what differs and exit non-zero; write nothing")
    a.add_argument("--readme", action="store_true",
                   help="regenerate reported/README.md from the manifest and stop")
    a.add_argument("--tissue", action="store_true",
                   help="collect into reported/tissue/ from a results root run at the tissue reference")
    a = a.parse_args()
    global DEST, TISSUE
    if a.tissue:
        DEST, TISSUE = DEST / "tissue", True
    if a.readme:
        write_readme()
        return

    copied = same = missing = drifted = 0
    oversize = []
    for entry in MANIFEST:
        pairs = resolve(entry)
        if not pairs:
            print(f"  missing  {entry[1]}")
            missing += 1
            continue
        for dst, src in pairs:
            if not src.exists():
                print(f"  missing  {src.relative_to(paths.results())}")
                missing += 1
                continue
            if src.stat().st_size > CAP_BYTES:
                oversize.append((src, src.stat().st_size))
                continue
            if TISSUE:
                # arms-style tables (the ablation, the boundary evaluation)
                # carry both references in one file, so the tissue root holds
                # a copy of the local one: not a second result, not tracked twice
                local = HERE / "reported" / dst.relative_to(DEST)
                if local.exists() and filecmp.cmp(src, local, shallow=False):
                    same += 1
                    continue
            fresh = dst.exists() and filecmp.cmp(src, dst, shallow=False)
            if fresh:
                same += 1
                continue
            if a.check:
                print(f"  drifted  {dst.relative_to(HERE)}")
                drifted += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            print(f"  wrote    {dst.relative_to(HERE)}")
            copied += 1

    for src, n in oversize:
        print(f"  too big  {src.name} ({n/1e6:.1f} MB) -- summarise it before tracking")
    verb = "would rewrite" if a.check else "wrote"
    print(f"\n{verb} {drifted or copied}, unchanged {same}, missing {missing}")
    if missing:
        print("missing tables are experiments that have not been run on this machine; "
              "see reported/README.md for the target that produces each one")
    if a.check and drifted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
