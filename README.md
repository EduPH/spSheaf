# spSheaf

Sheaf-theoretic spatial domain inference from raw molecules, without cell
segmentation. spSheaf places kernel-smoothed expression fibers on a grid,
gives each vertex a local principal-subspace stalk, scores every edge by its
*sheaf disagreement* — the residual mismatch of neighbouring local coordinates
after the stalks are aligned — and attenuates those transitions during
clustering. Paluzo-Hidalgo & Imoto, Nature Methods (submitted).

## Install

```bash
pip install -e .                     # Python >= 3.9; numpy, scipy, scikit-learn, pandas
pip install -r requirements-lock.txt # exact versions the results were produced on
pip install -e ".[benchmark]"        # optional: ssam, points2regions
make test
```

## Use

```python
from spsheaf import io, fibers, graph, sheaf, clustering

grid, raw, _ = fibers.build_kernel_fibers(mol, grid_spacing=15, bandwidth=25,
                                        x_col="x", y_col="y")     # mol: DataFrame with x, y, gene
fib    = fibers.normalize_fibers(raw)
edges  = graph.build_spatial_graph(grid, k=6)
stalks = sheaf.compute_stalks(fib, grid, uniform_dim=6, k_nbrs=50)
rs     = sheaf.compute_restriction_maps(stalks, edges, 6)
b_raw, b_smooth = sheaf.compute_boundary_scores(fib, grid, stalks, edges, rs, smooth_radius=60)
labels = clustering.boundary_aware_clustering(fib, grid, b_smooth, edges, n_clusters=8)
labels, _ = clustering.spatial_cleanup(labels, edges, grid_coords=grid)
```

`spsheaf.pipeline.run_single_slice` wraps the eight steps and the evaluation.
Units are those of the input coordinates. Parameters: grid spacing `δ`,
kernel bandwidth `h`, stalk dimension `d`, smoothing radius `σ_s`, boundary
weight `α_b`, spatial weight `α_s`, and the coordinate reference
(`SPSHEAF_REFERENCE=local|tissue`, default local). `docs/package_api.md`
lists the modules.

## Reproduce

Point the code at the data (`data/README.md` lists every input and where it
comes from):

```bash
export SPSHEAF_DATA=/path/to/data      # or copy paths.example.toml to paths.toml
```

Rebuild the figures from the stored tables (minutes, no raw data needed
except for the panel dumps under `data/results`):

```bash
make figures && make figures-verify   # 13 PDFs into figures/out/, checked against the 180 mm / 6-8 pt spec
```

Check a number without running anything: every spSheaf table in the paper is
under `reported/` (`reported/README.md` names the claim and the command behind
each file); the competitors' tables and per-cell labels are under
`benchmarks/segfree/results/`. `make reported-check` fails if `reported/`
drifts from the results tree.

Re-run the experiments (hours; raw inputs required):

```bash
make merfish merfish-seeds merfish-loo-cv   # Table 1, Fig 2b/c: five slices, 20 seeds, the 27-configuration grid
make revision                               # ablation, boundary evaluation, k sweep (Fig 2d/e)
make osmfish-all osmfish-insample osmfish-insample-tissue   # Fig 3, ED 4, the 2 x 108 grid
make spacetx-all                            # both SpaceTx sections, 162-configuration grids, ED 6
make matched-selection benchmark-table      # one selection rule for all four methods (Table 2)
make help                                   # everything else
```

The grids are resumable: interrupt and re-run, only missing configurations
are computed. The competitor sweeps (SSAM, Points2Regions, FICTURE) are
reproducible from the tracked metric tables; re-running them needs the
competitors' inputs, which `python -m benchmarks.segfree.inputs` names.

## Main results

ARI against the published annotation, every method tuned on its own grid on
the section it is scored on (Table 2 of the paper). MERFISH: one
configuration per method over the five slices.

| | MERFISH (5 slices) | osmFISH | Allen prototype MERFISH | BaristaSeq |
|---|---|---|---|---|
| **spSheaf** (local / tissue reference) | **0.571** / 0.548 | 0.658 / 0.673 | 0.665 / **0.754** | 0.749 / **0.757** |
| Points2Regions | 0.457 | 0.432 | 0.684 | 0.733 |
| SSAM | 0.401 | **0.729** | 0.410 | 0.738 |
| FICTURE | 0.342 | 0.465 | 0.475 | 0.631 |

On MERFISH the ranking holds under all three selection rules the five slices
allow — per-slice argmax 0.582 vs 0.510, one configuration 0.571 vs 0.457,
leave-one-slice-out 0.552 vs 0.388 — and spSheaf ranks first of twelve on
NMI against the SDMBench baselines. On osmFISH the two missed layers are not
separable in the 33-gene panel even under supervision; see
`experiments/osmfish/`. Ten ablation arms attribute +0.102 ARI to the sheaf
term over no weighting and +0.090 over its own spatial shuffle.

## Layout

```
src/spsheaf/         the method (fibers, graph, sheaf, clustering, evaluation, pipeline)
experiments/         one folder per dataset; each README gives commands, inputs and outputs
benchmarks/segfree/  SSAM, Points2Regions and FICTURE under a matched protocol; results/ tracked
figures/             figure pipeline; out/ is generated, assets/ holds the few placed rasters
reported/            spSheaf's own result tables, one row per configuration (tracked)
tests/  tools/       unit tests; provenance and requirements scripts
docs/                package API; dev-notes/ are working notes
data/                untracked; data/README.md
```

## Licence

MIT.
