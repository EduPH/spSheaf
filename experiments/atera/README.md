# 10x Atera — breast and cervical FFPE

> **Not part of the manuscript.** This analysis was cut during the scope
> reduction and no figure or number from it appears in the paper. It is kept
> because the runs are real and the code is the largest-scale exercise of the
> pipeline in the repository. Treat everything here as exploratory.

Whole-transcriptome in situ, ~18,000 genes. Breast: 740,442,119 transcripts
over ~9.9 × 5.9 mm. Cervical: 1,235,087,993 transcripts over ~8.9 × 8.9 mm.
No ground truth of any kind.

## Data

- Breast: <https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer>
- Cervical: <https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-cervical-cancer>

Download each Output Bundle and extract `transcripts.parquet`.
`README_run_notes.md` keeps the original working notes, including the full
parameter rationale.

## Run

No `make` target.

```bash
python -m experiments.atera.run_atera --input <transcripts.parquet> --inspect

python -m experiments.atera.run_atera --input <transcripts.parquet> \
  --subsample 0.01 --top-genes 500 --grid-spacing 50 --bandwidth 100 \
  --smooth-radius 150 --n-clusters 12 --min-molecules 15 --block-size 1000 \
  --output-dir data/results/atera/output_atera_breast

# the marker-aware variant, which is the one that recovers cell types
python -m experiments.atera.atera_cache ...
python -m experiments.atera.atera_markeraware ...
```

QV ≥ 20 and `is_gene` filtering are on by default.

## Results

Two families of runs, with different numbers. Say which one you mean.

| Run | Domains | Coherence | Grid points |
|---|---|---|---|
| Breast, top-500 by count | 12 | 0.934 | 22,471 |
| Cervical, top-500 by count | 12 | 0.948 | 29,012 |
| Breast, marker-aware (1 %) | 12 | 0.949 | 23,285 |
| Breast, full density (740 M) | 12 | 0.953 | — |
| Cervical, marker-aware | 12 | 0.942 | — |

Subsampling to 1 % agrees with the full-density run at ARI 0.848 / NMI 0.853,
and 84.2 % of grid points take the same cell-type label.

**The count filter was the mistake worth recording.** Selecting the top 500
genes by raw count drops lineage markers that are informative but not
abundant, and an entire basal/squamous compartment with them. Keeping 100 % of
curated-marker molecules and 1 % of the rest recovers it at signature score
+1.9, and it persists at +1.94 in the full-density run. With that fix the
breast section resolves luminal tumour (ESR1/PGR/GATA3/FOXA1/XBP1),
basal/squamous, stroma/CAF, immune, vascular and proliferative compartments;
three of twelve domains remain low-signal and unassigned.

**Why it was cut.** The analysis forces the curated markers into the fiber and
then reports recovering a compartment defined by those same markers, which is
circular. There is no ground truth, silhouette is negative, and the signature
auto-assignment is imperfect — in the original run only 5 of 12 breast domains
cleared the confidence margin, and one clearly stromal POSTN/COL3A1 domain was
auto-called "Immune". `ATERA_SUMMARY.md` and `ATERA_MARKERAWARE_NOTE.md` carry
the full record.
