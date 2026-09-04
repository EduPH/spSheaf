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

## Known asymmetry

Competitor hyperparameters were selected by a search scored **against the
ground-truth annotation**; spSheaf's were fixed by leave-one-out
cross-validation. The comparison is therefore biased in the competitors'
favour, and the reported competitor values are upper bounds. Resolving this
with one identical, annotation-free selection rule for all four methods is the
open item that most affects what the manuscript can claim.
