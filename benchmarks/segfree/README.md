# Segmentation-free comparison: SSAM, Points2Regions, FICTURE

Head-to-head against the three closest segmentation-free methods, on the same
raw molecule coordinates spSheaf consumes. Results are in `results/` and are
tracked in git — they are the provenance for the comparison table in the
manuscript.

## Install

The comparators are deliberately **not** core dependencies. A broken install
here must never stop the main pipeline from running.

```bash
pip install -e ".[benchmark]"     # ssam, points2regions, anndata
```

Two dependencies are not pip-installable:

* **FICTURE** — external CLI, v0.0.5.7 was used for the published numbers.
* **R + sctransform + feather** — SSAM's preferred normalisation shells out to
  R. It was *not* available when these results were produced; SSAM's built-in
  `normalize_vectors()` was substituted. SSAM still scores well on osmFISH
  under the same substitution, so this is unlikely to explain its lower MERFISH
  score, but it is a stated caveat in the manuscript Methods.

## Inputs

The comparison was originally run on the collaborator's machine against files
whose names differ from the copies in this repository. Check what is available:

```bash
python -m benchmarks.segfree.inputs
```

Anything reported missing must be obtained or converted before the benchmark
can be rerun end to end. Do not substitute a similarly-named file blindly —
see `inputs.py` for what each entry actually is.

## Metrics

There is exactly one implementation of the SDMBench panel, in
`spsheaf.evaluation`. `common.py` imports it. An earlier duplicate copy lived
here; it was verified equivalent (`tests/test_metric_parity.py`, 25 cases,
agreement to 1e-9) and then removed, so spSheaf and its comparators are scored
by the same ruler.

## Known asymmetry — and what was done about it

Competitor hyperparameters are selected by a search scored **against the
ground-truth annotation**. That is how these methods are used in practice and
how their own papers report them, so the numbers here are the ones a reader
would meet; but it means a competitor value is an upper bound, and a comparison
is only fair if spSheaf is given the same latitude.

Two things follow, and both are now done rather than noted:

1. **Matched search budgets.** The comparison used to select spSheaf from a grid
   an order of magnitude smaller than the competitors'. The budgets are:

   | method | settings, per single section | how they arise |
   |---|---|---|
   | SSAM | 144 (osmFISH), 287–288 (SpaceTx) | `find_domains` sweep × resolution × normalisation |
   | FICTURE | 18 configurations, 66 runs | aggregation strategy × model × seed |
   | Points2Regions | 24–32 | pixel width × smoothing |
   | spSheaf | 132 (osmFISH), 162 (SpaceTx) | `make osmfish-insample`, `make spacetx-insample-grid` |

   SSAM's resolution is a hyperparameter of its clustering, not a fixed property
   of the method, and treating it as one is what moved its osmFISH score from
   0.566 to **0.729** — it now leads that dataset. `make benchmark-table` reads
   every resolution and reports the argmax over all of them, which is the number
   the manuscript compares against.

2. **One selection rule per comparison.** Where there are five sections
   (MERFISH) every method is selected leave-one-slice-out, so no method sees the
   section it is scored on (`make matched-selection`). Where there is one
   section there is nothing to hold out, so every method is at its in-sample
   argmax and spSheaf is reported twice — transferred and in-sample — in Table 2.

Still open: a label-free rule that picks the coordinate reference and $\sigma_s$
without an annotation. Until there is one, the sensitivity to both is reported
(ED Fig. 5) rather than used, and the transferred row is the honest headline.

## Tables

```bash
make benchmark-table        # rebuild results/benchmark_table.csv from the metric files
make benchmark-table-check  # fail if it has drifted
make matched-selection      # the leave-one-slice-out rule across all four methods
```

`results/benchmark_table.csv` is derived, not maintained by hand — it used to be
the latter, which is how it came to carry SSAM's superseded osmFISH score.
`results/metrics/` holds the per-run JSON and CSV it is built from, and
`results/labels/` the per-cell label tables the figures Hungarian-match.
spSheaf's own tables are not here; they are under `reported/`.

**One rule for what counts as a configuration**, for every method and every
dataset: a single run at a fixed seed, then the argmax over configurations.
Points2Regions' grid is one run per (pixel width, smoothing), SSAM's
`find_domains` sweep one run per setting, and spSheaf's own grids one run per
configuration at k-means seed 42. FICTURE is the only method with seed
replicates — its `spatial` and `kmeans` aggregations were run at five seeds each,
its `argmax` strategy is deterministic — so its seeded configurations are taken
at their first seed and the deterministic ones as run.

The alternatives were both rejected, and both are in `benchmark_table.csv`
labelled as reference only: averaging FICTURE's seeds scores it at an expected
value while every other method sits at one draw, and taking the best of all 66
runs gives it a wider search than anyone else was given. The three rules differ
by up to 0.046 ARI, so the choice is not cosmetic — it just has to be the same
choice for everybody.
