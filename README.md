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
                   results/ the competitors' tables and per-cell labels (tracked)
figures/           the figure pipeline and the spec it enforces
                   assets/ inputs, out/ generated (untracked)
reported/          spSheaf's own result tables, one row per configuration
                   (tracked; see its README, and tools/collect_reported.py)
tools/             maintenance scripts (requirements, provenance)
tests/             unit tests, including metric-parity between implementations
data/              not tracked; see data/README.md for availability
```

Run `make help` for every target.

## Results

Four annotated datasets carry the quantitative claims: five MERFISH sections of
mouse hypothalamus, and three single cortical sections — osmFISH
somatosensory, and the Allen prototype MERFISH and BaristaSeq sections of
SpaceTx mouse VISp.

Search budget is the thing to watch when reading any of this. All three
comparators are distributed with settings their authors tuned, so each is given
its own full grid here: 108 settings for SSAM on osmFISH and 287–288 on the
SpaceTx sections, 66 for FICTURE, 24 for Points2Regions. On MERFISH, where
there are five sections, all four methods are selected leave-one-slice-out, so
no method sees the section it is scored on. On the three single sections there
is nothing to hold out, so every method is at its in-sample argmax, and spSheaf
is reported twice:

- **transferred** — every parameter fixed on MERFISH by leave-one-slice-out and
  carried to the other three datasets unchanged. Nothing is fitted on the
  section being scored. This is the number that speaks to whether the method
  travels.
- **in-sample** — the argmax of a grid no larger than the competitor's on that
  same section, which is the rule the competitors are already under.

ARI against the published annotation, Hungarian-matched:

| Method | MERFISH (5 sections, held out) | osmFISH | Allen prototype | BaristaSeq |
|---|---|---|---|---|
| **spSheaf** (transferred) | **0.571** | 0.658 | 0.541 | 0.713 |
| **spSheaf** (in-sample) | — | 0.658 | **0.754** | **0.757** |
| Points2Regions | 0.388 | 0.432 | 0.684 | 0.733 |
| SSAM | 0.343 | **0.729** | 0.410 | 0.738 |
| FICTURE | 0.326 | 0.463 | 0.490 | 0.631 |

Under the one rule that applies to all four datasets — each method selected on
its own grid, held out where holding out is possible — spSheaf leads on three
and trails on one. On MERFISH the margin survives the harder comparison too:
its held-out 0.571 is above the best any competitor reaches when allowed to
pick per slice on the slice itself (Points2Regions, 0.510).

On the two SpaceTx sections the in-sample gain is real, and the transferred
deficit is the price of carrying the smoothing radius and the coordinate
reference over rather than fitting them. On osmFISH it is not: the argmax over
the matched grid *is* the transferred configuration, so the gap to SSAM does not
narrow with tuning. `experiments/osmfish/` documents why — a supervised
classifier on the same fibers recovers the confusable layers, so the
information is present in the panel, but the disagreement field at those
interfaces is 0.82× its value at the interfaces that are found, and the domains
are narrower than the kernel's mixing extent. That is a resolution limit of the
33-gene panel and the geometry rather than a tuning failure, and it is reported
as a limitation. osmFISH is also one of SSAM's own published analyses.

Against the SDMBench comparison of Yuan et al. 2024 on MERFISH, spSheaf ranks
**first of twelve on NMI and homogeneity** and second on completeness, and
mid-field on the spatial-continuity metrics.

Ten ablation arms differing only in the edge score, over five slices at five
seeds each, show the sheaf term is worth **+0.102 ARI** over no boundary
weighting, **+0.102** over an independent random field and **+0.090** over its
own spatially shuffled field. That last margin is the informative one: the
shuffle keeps the field's distribution and destroys only its spatial
arrangement, so the gain is not generic smoothing.

Which parameters the accuracy actually depends on is measured rather than
asserted. Over the three in-sample grids, the coordinate reference moves mean
ARI by up to 0.13 and the kernel bandwidth and smoothing radius by up to 0.08,
while the stalk dimension moves it by at most 0.02 over 4 ≤ d ≤ 20 — a null
result, and the answer to whether the subspace is short of capacity. A
parameter's marginal is only read over settings run at every level of it;
`figures/ed_sensitivity.py` enforces that and draws the panel from `reported/`.

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

Two Extended Data panels need a run of their own before they can be drawn, and
each script says so and prints the exact command when its input is missing —
`make spacetx-insample-panels` for the in-sample maps of ED Fig. 6, and
`make osmfish-insample` plus `make spacetx-insample-grid` for the grids behind
ED Fig. 5. ED Fig. 5 falls back to the archived copies in `reported/` when the
grids are not on this machine, so it builds from a bare clone.

### 3. Check a number without re-running anything

Every figure and table in the manuscript is backed by a small table under
`reported/`, which is versioned precisely because `data/` is not.
`reported/README.md` maps each file to what it supports and to the command that
produced it; the competitors' tables sit separately under
`benchmarks/segfree/results/`, with the per-cell label CSVs the figures
Hungarian-match. Together the two are the provenance of Table 2.

```bash
make reported         # refresh reported/ from the results root, and its README
make reported-check   # fail if the two have drifted
```

### 4. Re-run the experiments

Slow, and needs the raw inputs rather than the stored results. One exception:
the segmentation-free comparison (SSAM, Points2Regions, FICTURE) is
reproducible from the per-cell label tables and per-configuration metric
tables tracked under `benchmarks/segfree/results/` (`make matched-selection`),
not from raw competitor inputs, which are not redistributed;
`python -m benchmarks.segfree.inputs` reports what is missing.

```bash
make merfish              # the headline benchmark, all five slices
make merfish-seeds        # 20 k-means seeds x 5 slices: Table 1
make merfish-loo-cv       # the leave-one-slice-out selection itself
make osmfish              # the transferred configuration
make osmfish-all          # every osmFISH analysis the manuscript reports
make spacetx-all          # both SpaceTx sections, end to end
make revision             # ablation, boundary evaluation, k-sensitivity
make method-validation    # the synthetic check on what the sheaf term sees
make matched-selection    # one selection rule across all four methods
```

The parameter grids are the slow part, they are matched to the competitors'
budgets by design, and both are resumable — interrupt and re-run, and only the
configurations still missing from the CSV are computed:

```bash
make osmfish-insample        # osmFISH, matched to SSAM's 108 settings
make spacetx-insample-grid   # 162 configurations per SpaceTx section
make spacetx-insample-panels # re-run each section at its grid argmax, for ED Fig. 6
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
