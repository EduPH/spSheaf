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
make merfish            # five slices at the configuration of Table 3 -> data/results/merfish/output_merfish/
make merfish-seeds      # 20 k-means seeds x 5 slices: Table 1 (means), ED 2c
make merfish-loo-cv     # the 27-configuration grid (sigma_s, alpha_b, alpha_s) x 5 slices: Fig 2b/c, Methods
make cache SLICE=MERFISH_0.19.h5ad   # shared cache (fibers, stalks, ground truth) for the three below; once per slice
make sheaf-ablation SLICE=...        # ten edge-score arms, 5 seeds: Fig 2d
make boundary-eval SLICE=...         # the fields scored against annotated boundaries: Fig 2e
make k-sensitivity SLICE=...         # k = 4..16
make revision SLICE=...              # the three above; loop over the five slices
make merfish-reference-field         # local vs tissue reference at interfaces (Sec 2.9)
make panels-segfree                  # per-cell maps of slice 0.19 for Fig 2a
make panels-multislice               # gene-centroid registration of the five slices: ED 3
make merfish-section-all             # everything above, in order (hours)
```

## Results (in the paper)

| | ARI | NMI |
|---|---|---|
| Table 1, 20-seed mean over 5 slices | 0.571 ± 0.000 | 0.655 ± 0.000 |
| per slice (0.04 … 0.24) | 0.570, 0.632, 0.478, 0.656, 0.518 | |
| 27-grid: per-slice argmax / one configuration / leave-one-slice-out | 0.582 / 0.571 / 0.552 ± 0.052 | |

Ablation (grid-level ARI, 5 slices × 5 seeds): sheaf local 0.576, sheaf
tissue 0.536, neighbour 0.530, identity (tissue-referenced) 0.526, Procrustes
/ projection / Grassmann 0.491–0.495, shuffled 0.486, random 0.474, none
0.474. Boundary AP at 15 µm: the subspace scores lead (0.305–0.308), the
method's own score is last (0.221) — localization and partition quality order
the fields oppositely.

Multi-slice: cross-slice label agreement 69.8 % (63.1–78.5 %) against a
self-alignment bound of 91.5 %.

## Files

`run_all_slices.py` (pipeline), `loo_cv.py` (grid), `_cache.py` +
`sheaf_ablation.py` + `boundary_eval.py` + `k_sensitivity.py` (revision
experiments), `reference_field.py`, `multislice_alignment.py` /
`multislice_panels.py`, `segfree_panels.py`, `method_validation.py` (synthetic
check, `make method-validation`; not in the paper). Tracked copies of every
table are under `../../reported/merfish/`.
