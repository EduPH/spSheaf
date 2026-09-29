# MERFISH — Moffitt et al. 2018 mouse hypothalamus

The development benchmark: 135 genes, five annotated slices (Bregma 0.04–0.24,
5,488–5,926 annotated cells each, 8 domains). Input is the whole physical
section by bregma, all seven focal planes, 1.8–2.0 M molecules per section
(`SPSHEAF_MERFISH_INPUT=section`, the default); every competitor reads the
same table.

Input: `data/raw/merfish/moffitt2018_export_full/molecules_full_add.csv.gz`
(processed molecule table, Y. Imoto; see `../../data/README.md`).

## Run

```bash
make merfish            # five slices at the configuration of Supplementary Table 2 -> data/results/merfish/output_merfish/
make merfish-seeds      # 20 k-means seeds x 5 slices: Table 1 and Supplementary Table 1 (means), ED 1c
make merfish-loo-cv     # the 27-configuration grid (sigma_s, alpha_b, alpha_s) x 5 slices: Fig 2b/c, Methods (run at both references for the union)
make cache SLICE=MERFISH_0.19.h5ad   # shared cache (fibers, stalks, ground truth) for the three below; once per slice
make sheaf-ablation SLICE=...        # ten edge-score arms, 5 seeds: Fig 2d
make boundary-eval SLICE=...         # the fields scored against annotated boundaries: Fig 2e
make k-sensitivity SLICE=...         # k = 4..16
make revision SLICE=...              # the three above; loop over the five slices
make merfish-reference-field         # local vs tissue reference at interfaces (Sec 2.9)
make merfish-k-selection             # k = 2..16 on one slice: silhouette, grid-phase stability, seed control (SLICE=...)
make merfish-k-split                 # k = 8, 9, 10 on the five slices, then the division record (k_split_*.csv)
make panels-segfree                  # per-cell maps of slice 0.19 for Fig 2a
make panels-multislice               # gene-centroid registration of the five slices: Supplementary Fig. 2
make merfish-section-all             # everything above, in order (hours)
```

All targets run at the package default, the tissue reference. The MERFISH
numbers in the paper are at the local reference, which the tuning grid
selects on this benchmark (`SPSHEAF_REFERENCE=local make ...`); the grid
itself covers both references (`loo_cv_table_union.csv`, 54 configurations).

## Results (in the paper)

| | ARI | NMI |
|---|---|---|
| Table 1, 20-seed mean over 5 slices | 0.571 ± 0.000 | 0.655 ± 0.000 |
| per slice (0.04 … 0.24) | 0.570, 0.632, 0.478, 0.656, 0.518 | |
| 54-grid (both references): per-slice argmax / one configuration / leave-one-slice-out | 0.584 / 0.571 / 0.552 | |
| tissue reference, same three rules | 0.564 / 0.548 / 0.540 | |

Ablation (grid-level ARI, 5 slices × 5 seeds): sheaf local 0.576, sheaf
tissue 0.536, neighbour 0.530, identity (tissue-referenced) 0.526, Procrustes
/ projection / Grassmann 0.491–0.495, shuffled 0.486, random 0.474, none
0.474. Boundary AP at 15 µm: the subspace scores lead (0.305–0.308), the
method's own score is last (0.221) — localization and partition quality order
the fields oppositely.

Multi-slice: cross-slice label agreement 69.8 % (63.1–78.5 %) against a
self-alignment bound of 91.5 %.

Choosing k (five slices, k = 2–16): the silhouette is maximal at k = 10, 8, 9,
8, 9 against the annotated 8, and within 0.001 of its maximum at k = 8 on four
slices; grid-phase stability is not usable on a single section (upper bound
16, 15, 8, 12, 9; four of its ten lowest values coincide with drops in the
seed control). At k = 9 the BST divides the same way on slices 0.19 and 0.24
(r = 0.84 between the per-gene differences) and the MPN on 0.04 and 0.09
(r = 0.88); the 26-division record is `k_split_events.csv`.

## Files

`run_all_slices.py` (pipeline), `loo_cv.py` (grid), `_cache.py` +
`sheaf_ablation.py` + `boundary_eval.py` + `k_sensitivity.py` (revision
experiments), `reference_field.py`, `multislice_alignment.py` /
`multislice_panels.py`, `segfree_panels.py`, `k_selection.py` (the k
criteria on one slice), `k_split.py` + `k_split_analysis.py` +
`k_split_table.py` (divisions at k = 9, 10; the manuscript's division table),
`method_validation.py` (synthetic check, `make method-validation`; not in the
paper). Tracked copies of every
table are under `../../reported/merfish/`.
