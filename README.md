# spSheaf

Sheaf-theoretic boundary detection for segmentation-free spatial domain
inference in spatial transcriptomics.

spSheaf takes raw molecule detections — coordinates and gene identities, with
no cell segmentation — and returns a partition of the tissue into spatial
domains, delineated by an explicit boundary field. It builds a cellular sheaf
over a spatial graph and targets the *discontinuities* of that sheaf directly;
it never assembles or diagonalizes a sheaf Laplacian.

The geometric core — fibers, stalks, restriction maps and the disagreement
score — is the same across platform, dimension and modality: it runs on 2D
transcript maps, on 3D subnuclear chromatin imaging, and on multiplexed protein
panels. What is set per dataset is the fiber contents, the spatial
discretization, the stalk dimension and the number of output domains.

## Install

```bash
pip install -r requirements.txt        # runtime + the figure checks
pip install -r requirements-lock.txt   # or: the exact versions we ran on
pip install -e ".[benchmark]"          # optional: ssam, points2regions
```

`requirements.txt` is generated from `pyproject.toml`, which is the source of
truth; `make requirements` regenerates it and a test fails if it goes stale.
`requirements-lock.txt` pins every version the published results were produced
on, for exact reproduction.

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
| osmFISH somatosensory cortex | 0.620 | 0.694 | 11 annotated regions, single section; annotation registered label-free |

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

On osmFISH spSheaf reaches ARI 0.620 / NMI 0.694 against SSAM's 0.566 / 0.721
(SSAM leads on NMI) — osmFISH is one of SSAM's own published analyses and its
released settings were tuned on it.

Ten ablation arms differing only in the edge score show the sheaf term is
worth **+0.075 ARI** over no boundary weighting and **+0.045** over its own
spatially shuffled field, so the gain is not generic smoothing.

Each experiment's README gives the full numbers and the protocol they were
measured under. **Results reported under different protocols are not
interchangeable** — `experiments/README.md` explains which is which.

## Reproducing

### 1. Point at the data

The data tree is not part of this repository (see `data/README.md`). Tell the
code where it is, by either:

```bash
export SPSHEAF_DATA=/path/to/data          # per shell, or
cp paths.example.toml paths.toml           # and set data_root in it
```

With neither set, `<repo>/data` is used. Check it resolved:

```bash
python -c "from spsheaf import paths; print(paths.data_root())"
```

### 2. Rebuild the figures from stored results

Fast — this reads the metric tables and panel dumps, and recomputes nothing:

```bash
make figures         # every figure, into figures/out/
make figures-verify  # assert the specification on the finished PDFs
make figures-test    # unit tests for the figure pipeline itself
```

### 3. Re-run the experiments

Slow, and needs the raw inputs rather than the stored results. One exception:
the segmentation-free comparison (SSAM, Points2Regions, FICTURE) is
reproducible from the per-cell label tables and per-configuration metric
tables tracked under `benchmarks/segfree/results/` (`make matched-selection`),
not from raw competitor inputs, which are not redistributed;
`python -m benchmarks.segfree.inputs` reports what is missing.

```bash
make merfish         # the headline benchmark, all five slices
make osmfish
make revision        # ablation, boundary evaluation, k-sensitivity
make matched-selection
```

Each experiment's README gives its own commands and inputs. `make help` lists
every target.

Figure scripts write into `figures/out/`, which is created on first run and
not tracked. `figures/assets/` holds the legacy rasters that the Extended Data
composites place; those are inputs, not outputs.

## Citation

> E. Paluzo-Hidalgo and Y. Imoto. *Sheaf-theoretic boundary detection for
> segmentation-free spatial domain inference in spatial transcriptomics.*
> Manuscript in preparation.

## Licence

MIT. See `LICENSE`.
