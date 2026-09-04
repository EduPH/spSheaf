# spSheaf

Sheaf-theoretic boundary detection for segmentation-free spatial domain
inference in spatial transcriptomics.

spSheaf takes raw molecule detections — coordinates and gene identities, with
no cell segmentation — and returns a partition of the tissue into spatial
domains, delineated by an explicit boundary field. It builds a cellular sheaf
over a spatial graph and targets the *discontinuities* of that sheaf directly;
it never assembles or diagonalizes a sheaf Laplacian.

The construction transfers unchanged across platform, dimension and modality:
the same pipeline runs on 2D transcript maps, on 3D subnuclear chromatin
imaging, and on multiplexed protein panels.

## Install

```bash
pip install -e .                  # or: make install
pip install -e ".[benchmark]"     # adds ssam, points2regions, anndata
pip install -e ".[figures]"       # adds what the figure checks need
```

Python ≥ 3.9. `make test` runs the suite.

## How it works

Eight steps, each in its own module under `src/spsheaf/`:

1. **Expression fibers** — a regular grid over the tissue; at each grid point,
   Gaussian kernel smoothing of nearby molecules gives a gene-expression fiber.
2. **Spatial graph** — a symmetric k-nearest-neighbour graph on the grid.
3. **Stalks** — at each vertex, a local PCA basis of its neighbours' fibers.
4. **Restriction maps** — per edge, the SVD of the stalk overlap
   `M_uv = U_uᵀU_v = LΣRᵀ` gives maps `Lᵀ`, `Rᵀ` into a shared,
   principal-angle-aligned frame. The singular values are the cosines of the
   principal angles between the two stalks.
5. **Sheaf disagreement** — `D(e) = ‖F_{u◁e}ξ_u − F_{v◁e}ξ_v‖²`, the residual
   that survives the best alignment.
6. **Boundary field** — per-vertex mean over incident edges, Gaussian-smoothed.
7. **Boundary-aware clustering** — expression attenuated by
   `w(v) = exp(−α_b·B̃(v)/median B̃)`, concatenated with scaled coordinates,
   k-means.
8. **Connectivity cleanup** — small disconnected fragments are relabelled to
   their surrounding majority.

`docs/package_api.md` documents the modules; `docs/method_demo/` has small
synthetic constructions that isolate what the sheaf term contributes.

## Layout

```
src/spsheaf/       the method
experiments/       one folder per dataset, each with its own README
benchmarks/segfree/ comparison against SSAM, Points2Regions and FICTURE
figures/           the figure pipeline and the spec it enforces
                   assets/ inputs, out/ generated (untracked)
tests/             unit tests, including metric-parity between implementations
data/              not tracked; see data/README.md for availability
```

Run `make help` for every target.

## Results

Two annotated benchmarks carry the quantitative claims.

| Dataset | ARI | NMI | Note |
|---|---|---|---|
| MERFISH mouse hypothalamus, 5 slices | **0.559 ± 0.008** | **0.641 ± 0.004** | 20 k-means seeds; 8 annotated domains |
| osmFISH somatosensory cortex | 0.538 | 0.639 | 11 annotated regions, single section |

Against the SDMBench comparison of Yuan et al. 2024 on MERFISH, spSheaf ranks
**first of twelve on NMI and homogeneity** and second on completeness, and
mid-field on the spatial-continuity metrics.

Against the three closest segmentation-free methods, scored on the same
molecule tables under one leave-one-slice-out selection rule applied
identically to all four:

| Method | MERFISH ARI (held-out) |
|---|---|
| **spSheaf** | **0.557** |
| FICTURE | 0.266 |
| Points2Regions | 0.258 |
| SSAM (de novo) | 0.208 |

On osmFISH, SSAM reaches 0.566 against spSheaf's 0.538 — osmFISH is one of
SSAM's own published analyses and its released settings were tuned on it.

Ten ablation arms differing only in the edge score show the sheaf term is
worth **+0.075 ARI** over no boundary weighting and **+0.045** over its own
spatially shuffled field, so the gain is not generic smoothing.

Each experiment's README gives the full numbers and the protocol they were
measured under. **Results reported under different protocols are not
interchangeable** — `experiments/README.md` explains which is which.

## Reproducing

Obtain the raw data (`data/README.md`), then:

```bash
make merfish        # the headline benchmark
make osmfish
make revision       # ablation, boundary evaluation, k-sensitivity
make figures        # rebuild every figure from the stored results
make figures-verify # assert the figure specification on the output
```

Figure scripts write into `figures/out/`, which is created on first run and
not tracked. `figures/assets/` holds the legacy rasters that the Extended Data
composites place; those are inputs, not outputs.

## Citation

> E. Paluzo-Hidalgo and Y. Imoto. *Sheaf-theoretic boundary detection for
> segmentation-free spatial domain inference in spatial transcriptomics.*
> Manuscript in preparation.

## Licence

MIT. See `LICENSE`.
