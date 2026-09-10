# MERFISH — Moffitt et al. 2018 mouse hypothalamic preoptic region

The primary quantitative benchmark. 135 genes, ~20.5 M molecules, 5 coronal
slices, 8 annotated anatomical domains, ~14,200 grid points per slice.

## Run

```bash
make merfish                       # all five slices at the reference settings

make cache SLICE=MERFISH_0.19.h5ad # build the shared cache once, then:
make sheaf-ablation SLICE=MERFISH_0.19.h5ad
make boundary-eval  SLICE=MERFISH_0.19.h5ad
make k-sensitivity  SLICE=MERFISH_0.19.h5ad
make revision                      # all three
```

Needs `data/raw/merfish/moffitt2018_export_full/molecules_full_add.csv.gz`.
Reference settings: δ = 15 µm, h = 25 µm, k_graph = 6, k_nbrs = 50, d = 6,
σ_s = 60 µm, α_b = 0.5, α_s = 0.5, k = 8, τ = 0.01.

## Results

**Domain inference**, means over 20 k-means seeds × 5 slices:

| ARI | NMI | HOM | COM | CHAOS | PAS | ASW | time |
|---|---|---|---|---|---|---|---|
| 0.559 ± 0.008 | 0.641 ± 0.004 | 0.647 | 0.635 | 0.029 | 0.032 | −0.051 | 15.4 s |

Per-slice ARI 0.511, 0.525, 0.451, 0.660, 0.652. The only stochastic step is
the k-means initialization, and it moves ARI by ±0.008.

**The domain maps of Fig. 2a** (`segfree_panels.py`, `make panels-segfree`):
slice 0.19, per annotated cell, the annotation and every method's labels
Hungarian-matched to it. spSheaf is re-clustered from the `_cache` fibers and
boundary field at the defaults (ARI 0.62 on this slice for that run; the
20-seed pipeline mean is 0.660); the competitors are the tracked per-slice-best
label tables under `benchmarks/segfree/results/labels/`, i.e. each at its most
favorable in-sample configuration (FICTURE 0.28, Points2Regions 0.42, SSAM
0.22 on this slice).

**Multi-slice alignment, ED Fig. 4** (`multislice_panels.py`, `make
panels-multislice`): the five slices through the pipeline at the defaults
(seed 42), gene-centroid registration and the self-alignment control, dumped
to `panels/multislice.npz`. Mean adjacent-pair agreement 0.675 (0.633–0.722),
self-alignment upper bound 0.911. Needs ~6 GB: the loader keeps only the
annotated slices and nine columns of the molecule table.

**What the sheaf term contributes** (`sheaf_ablation.py`; ten arms differing
only in the edge score, everything else identical, means over 5 slices × 5
seeds):

| arm | ARI | NMI |
|---|---|---|
| sheaf | **0.571** | 0.641 |
| sheaf, local ξ | 0.557 | 0.633 |
| identity restrictions | 0.544 | 0.629 |
| plain kernel neighbour | 0.541 | 0.626 |
| *control:* shuffled field | 0.526 | 0.613 |
| procrustes / grassmann / projection | 0.519 / 0.518 / 0.518 | ≈0.609 |
| *control:* random field | 0.518 | 0.608 |
| *control:* no weighting (α_b = 0) | 0.496 | 0.583 |

The full score is worth **+0.075 ARI** over no weighting and **+0.045** over
its own spatially shuffled field, so the gain is not a generic smoothing
effect. Replacing the restriction maps with the identity costs 0.027.

**The boundary field on its own** (`boundary_eval.py`, average precision at a
15 µm tolerance) tells the opposite story: projection 0.331, grassmann 0.329,
procrustes 0.328, identity 0.271, neighbour 0.259, **sheaf 0.251**. The
subspace-only scores localize borders better (mean error 53 µm against 116 µm)
while producing worse partitions. This is reported rather than buried — the
coordinate residual is what the downstream clustering needs, and it is not the
sharpest edge detector.

**Multi-slice.** Gene-centroid registration across the five serial sections
gives 64.1 % mean cross-slice label agreement against a 91.1 % self-alignment
bound. Clustering all slices jointly was tried and rejected (mean ARI ≈ 0.29).

**k-sensitivity.** ARI peaks at k = 5 on two slices and k = 7 on three; NMI
peaks at the annotated k = 8 on two of five. Coherence rises monotonically
with k, so it cannot be used to select it.

Feeds Fig. 2a,c,d, Table 1, and Extended Data Figs. 1, 2 and 3 (ED numbering of 5 Sep 2026: 1 merfishslices, 2 bench, 3 methodval, 4 multislice, 5 osmmodes, 6 spacetx, 7 scale, 8 dnaterr, 9 ifheat, 10 pecab).
