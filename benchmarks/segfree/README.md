# Segmentation-free comparison: SSAM, Points2Regions, FICTURE

The three closest methods, run on the identical molecule tables spSheaf reads,
their native output mapped to one label per annotated cell and scored with the
same metric implementation (`spsheaf.evaluation`; parity test in
`tests/test_metric_parity.py`). `results/` is tracked: it is the provenance of
Table 2 and Fig. 2 and can be re-derived without the competitors' inputs.

## Install

```bash
pip install -e ".[benchmark]"     # ssam 1.1.3, points2regions, anndata
```

FICTURE (v0.0.5.7) is an external CLI. SSAM's published `sctransform`
normalisation shells out to R (4.3.3, sctransform 0.4.1) through
`sctransform_csv.py`. `python -m benchmarks.segfree.inputs` reports which raw
inputs are present; the competitors' inputs are not redistributed.

## Protocol

One rule for every method and dataset: a single run at a fixed seed per
configuration, then the argmax over the method's own grid, selected on the
annotation of the section it is scored on. Grids: SSAM 72 per MERFISH slice,
144 osmFISH, 287–288 per SpaceTx section (four cell-type clusterings ×
`find_domains` sweep; resolution treated as a hyperparameter because SSAM
1.1.3 under-resolves the published recipe); Points2Regions 45 / 32 / 24;
FICTURE 18 (three factor counts × two widths × three pixel→domain mappings,
seeded mappings at seed 0); spSheaf 27 / 216 / 162. On MERFISH, where there
are five slices, three rules are reported for all four methods: per-slice
argmax, one configuration by mean ARI, leave-one-slice-out.

## Run

```bash
# MERFISH (per slice; SSAM is two-phase)
SSAM_PIPELINE_ONLY=1 python -m benchmarks.segfree.run_ssam [slice ...]       # KDE -> cell types -> store
python -m benchmarks.segfree.ssam_resolution <slice> --resolutions 4 8         # sctransform, filter off
python -m benchmarks.segfree.ssam_sweep_offline --tags "" _sct_r4 _sct_r8      # find_domains sweep
python -m benchmarks.segfree.run_p2r_clean [slice ...]                         # Points2Regions grid
python -m benchmarks.segfree.prep_ficture_input merfish; bash run_ficture.sh <in> <out> 8,12,20 12,18
python -m benchmarks.segfree.eval_ficture <out> <in>

# osmFISH (registers the molecules first: osm_register.py)
SSAM_NORMALIZATION=sctransform SSAM_OUTLIER=off SSAM_RESOLUTIONS="0.15 4 8" python -m benchmarks.segfree.run_osmfish ssam
python -m benchmarks.segfree.run_osmfish p2r

# SpaceTx
make spacetx-segfree SPACETX=<section>; make spacetx-ficture SPACETX=<section>

# tables (from the tracked metric files; no competitor run needed)
make matched-selection        # results/matched_selection_{ari,nmi}.csv and per-slice: the three MERFISH rules
make benchmark-table          # results/benchmark_table.csv: every whole-section value with its rule
make benchmark-table-check
```

Environment knobs for SSAM: `SSAM_NORMALIZATION=sctransform|builtin`,
`SSAM_OUTLIER=on|off`, `SSAM_RESOLUTION(S)`, `SSAM_TAG`, `SSAM_NCORES`.

## Results (ARI; Table 2 and Fig. 2c)

| | MERFISH per-slice argmax / one config / held-out | osmFISH | Allen | BaristaSeq |
|---|---|---|---|---|
| spSheaf | 0.582 / 0.571 / 0.552 | 0.658 (0.673 tissue ref.) | 0.665 (0.754) | 0.749 (0.757) |
| Points2Regions | 0.510 / 0.457 / 0.388 | 0.432 | 0.684 | 0.733 |
| SSAM | 0.464 / 0.401 / 0.343 | 0.729 | 0.410 | 0.738 |
| FICTURE | 0.353 / 0.342 / 0.342 | 0.465 | 0.475 | 0.631 |

Known limits: SSAM's `find_domains` labels connected components, so its
domain count cannot be fixed; its published MERFISH analysis curates
clusters by hand, which the resolution sweep stands in for. The MERFISH
`find_domains` sweep runs through `ssam_domains.py`, a re-implementation
checked identical on slice 0.19.

## Files

`results/metrics/` per-run JSON/CSV, `results/labels/` per-cell labels the
figures Hungarian-match, `results/*.csv` derived tables. `common.py` loaders
and metrics, `inputs.py` input manifest, `osm_register.py` osmFISH
registration, `ssam_domains.py` the `find_domains` re-implementation,
`make_benchmark_table.py` / `matched_selection.py` the table builders.
`_legacy/` holds the pre-8-Sep-2026 aggregation script and its tables
(contaminated MERFISH inputs; kept as a record).
