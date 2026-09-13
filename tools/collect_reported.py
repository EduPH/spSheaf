"""Copy the result tables the manuscript is built from into ``reported/``.

``data/`` is deliberately untracked: it holds raw molecule tables, caches and
per-cell dumps that are far too large for git, and it lives outside the
repository on most machines. That left an asymmetry -- every competitor's
numbers were versioned under ``benchmarks/segfree/results/``, while spSheaf's
own were not -- so a clone could rebuild the competitor columns of Table 2 but
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

#: (destination subpath, source glob under the results root, what it supports)
MANIFEST: list[tuple[str, str, str]] = [
    # ---- Table 1: MERFISH, five slices, 20 k-means seeds -------------
    ("merfish/seed_sensitivity_per_slice.csv",
     "benchmark_sdmbench/output_seed_sensitivity/seed_sensitivity_per_slice.csv",
     "Table 1 and ED Fig. 1: per-slice ARI, mean over 20 seeds"),
    ("merfish/seed_sensitivity_summary.csv",
     "benchmark_sdmbench/output_seed_sensitivity/seed_sensitivity_summary.csv",
     "Table 1: the seed spread quoted in the caption"),
    ("merfish/seed_sensitivity_per_run.csv",
     "benchmark_sdmbench/output_seed_sensitivity/seed_sensitivity_per_run.csv",
     "Table 1: every (slice, seed) run behind the means"),
    ("merfish/all_slices_results.csv",
     "merfish/output_merfish/all_slices_results.csv",
     "Sec 2.2: the headline five-slice run at the transferred configuration"),
    ("merfish/loo_cv_table.csv", "merfish/loo_cv_table.csv",
     "Methods 4.12/4.14: leave-one-slice-out selection of (sigma_s, alpha_b, alpha_s)"),
    ("merfish/multislice.json", "merfish/panels/multislice.json",
     "Sec 2.6 and ED Fig. 3: cross-slice label agreement (69.8 %, 63.1-78.5 %) "
     "and the self-alignment bound (91.5 %); `make panels-multislice`"),
    ("osmfish/tune_cv.json", "osmfish/tune_cv.json",
     "osmFISH leave-one-spatial-block-out (not in the manuscript since 12 Sep 2026; "
     "kept as the record of the check)"),
    ("merfish/reference_field.csv", "merfish/reference_field.csv",
     "Sec 2.9: what the coordinate reference does to the boundary field -- the "
     "localization/partition dissociation, and the field's width after the "
     "median normalization the weighting applies"),

    # ---- Table 2, spSheaf columns ------------------------------------
    ("osmfish/osmfish_metrics.csv",
     "osmfish/output_osmFISH_boundary/osmfish_metrics.csv",
     "Table 2, osmFISH transferred row"),
    ("osmfish/osmfish_sweep.csv",
     "osmfish/output_osmFISH_boundary/osmfish_sweep.csv",
     "Table 2, osmFISH in-sample row, and ED Fig. 5 row 1 (the 108+24 grid)"),
    ("osmfish/osmfish_metrics_tissue_reference.csv",
     "osmfish/tissue_reference/osmfish_metrics.csv",
     "osmFISH at the local-reference scales with the reference switched to tissue (0.588; not in the manuscript -- Table 2 tissue row is the argmax in osmfish_sweep.csv)"),
    ("spacetx/MERFISH_Allen_Prototype_metrics.csv",
     "spacetx/MERFISH_Allen_Prototype/metrics.csv",
     "Table 2, Allen transferred row (one row per k-means seed)"),
    ("spacetx/MERFISH_Allen_Prototype_sweep.csv",
     "spacetx/MERFISH_Allen_Prototype/sweep.csv",
     "Table 2, Allen in-sample row, and ED Fig. 5 row 2 (the 162 grid)"),
    ("spacetx/BARISTASeq_metrics.csv", "spacetx/BARISTASeq/metrics.csv",
     "Table 2, BaristaSeq transferred row"),
    ("spacetx/BARISTASeq_sweep.csv", "spacetx/BARISTASeq/sweep.csv",
     "Table 2, BaristaSeq in-sample row, and ED Fig. 5 row 3 (the 162 grid)"),

    # ---- why osmFISH is the one dataset we do not lead ---------------
    ("osmfish/resolution.csv", "osmfish/resolution.csv",
     "Sec 2.9 and Methods: does domain WIDTH predict which domains are lost"),
    ("osmfish/elongation.csv", "osmfish/elongation.csv",
     "Sec 2.9: are ELONGATED domains lost, and is alpha_s the cause"),
    ("osmfish/alpha_sweep.csv", "osmfish/alpha_sweep.csv",
     "Methods (osmFISH spatial weight): ARI and NMI at alpha_s in {0, 0.25, 0.5, 0.75}. "
     "Written by elongation.py only since 11 Sep 2026 -- it used to print these four "
     "numbers and save nothing, so re-run `make osmfish-elongation` if the copy here "
     "predates that"),
    ("osmfish/tune_attribute.csv", "osmfish/tune_attribute.csv",
     "Methods: which transferred change costs the accuracy (8 runs)"),
    ("osmfish/seed_sensitivity.csv", "osmfish/seed_sensitivity.csv",
     "Table 2 caption: the osmFISH seed spread"),

    # ---- reviewer-memo experiments (MERFISH, all five slices) --------
    ("merfish/boundary_eval_*.csv", "merfish/boundary_eval_MERFISH_*.csv",
     "memo 3: the boundary field scored directly, without clustering"),
    ("merfish/sheaf_ablation_*.csv", "merfish/sheaf_ablation_MERFISH_*.csv",
     "memo 2: sheaf against the simpler local-subspace controls"),
    ("merfish/k_sensitivity_*.csv", "merfish/k_sensitivity_MERFISH_*.csv",
     "memo 5: k swept under label-free and label-dependent criteria"),

    # ---- the non-cortex applications --------------------------------
    ("brain_receptor/metrics.csv", "brain_receptor/output_brain_receptor/metrics.csv",
     "ED Fig. 7 (scale): the receptor-panel section"),
    ("brain_receptor/metrics_vs_allen.csv",
     "brain_receptor/output_brain_receptor/metrics_vs_allen.csv",
     "ED Fig. 7: agreement with the Allen reference annotation"),
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
     "ED Fig. 10 (PECAb): replicate 1"),
    ("pecab/Rep1_marker_corr.csv", "pecab/output_pecab_signaling_Rep1/marker_corr.csv",
     "ED Fig. 10: marker correlations, replicate 1"),
    ("pecab/Rep1_emt_concordance.csv",
     "pecab/output_pecab_signaling_Rep1/emt_concordance.csv",
     "ED Fig. 10: EMT concordance"),
    ("pecab/Rep2_summary.csv", "pecab/output_pecab_signaling_Rep2/summary.csv",
     "ED Fig. 10: replicate 2"),
    ("pecab/Rep2_marker_corr.csv", "pecab/output_pecab_signaling_Rep2/marker_corr.csv",
     "ED Fig. 10: the r = 0.94 cross-replicate agreement"),

    # ---- nuclear architecture (ED Figs 8 and 9) ---------------------
    ("takei/zone_mark_heatmap.csv", "takei/output_takei_if/zone_mark_heatmap.csv",
     "ED Fig. 9b: zone-by-mark composition, row-centred"),
    ("takei/zone_archetype_sizes.csv", "takei/output_takei_if/zone_archetype_sizes.csv",
     "ED Fig. 9b: how many zones fall in each archetype"),
    ("takei/zone_archetype_assignment.csv",
     "takei/output_takei_if/zone_archetype_assignment.csv",
     "ED Fig. 9b: (nucleus, zone) -> archetype, the assignment behind the panel"),
    ("takei/if_certification_summary.csv",
     "takei/output_takei_if/certification_summary.csv",
     "ED Fig. 9: the IF certification summary"),
    ("takei/if_certification_zone_table.csv",
     "takei/output_takei_if/certification_zone_table.csv",
     "ED Fig. 9: per-zone certification"),
    ("takei/dna_certification_per_nucleus.csv",
     "takei/output_takei_feat_full1/certification_per_nucleus.csv",
     "ED Fig. 8 (DNA territories): per-nucleus certification"),
    ("takei/dna_certification_radial.csv",
     "takei/output_takei_feat_full1/certification_radial.csv",
     "ED Fig. 8: the radial profile"),
    ("takei/dna_radial_vs_genedensity.csv",
     "takei/output_takei_feat_full1/radial_vs_genedensity.csv",
     "ED Fig. 8: radial position against gene density"),
    ("takei/dna_summary.csv", "takei/output_takei_feat_full1/summary.csv",
     "ED Fig. 8: the run summary"),
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
}

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
figures Hungarian-match. The two places together are the provenance of Table 2.

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
        groups.setdefault(Path(dst).parts[0], []).append((dst, src, why))
    lines = [README_HEAD]
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
    a = a.parse_args()
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
