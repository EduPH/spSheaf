# Experiments

One folder per dataset. Each has its own README with the command, the inputs
it needs, and what it found.

Two datasets carry the quantitative claims; the rest demonstrate that the same
pipeline transfers across platform, dimension and modality.

| Folder | Dataset | Status | Headline |
|---|---|---|---|
| [`merfish/`](merfish) | Moffitt 2018 MERFISH hypothalamus | **main text** | ARI 0.559, NMI 0.641 over 5 slices (20 seeds) |
| [`osmfish/`](osmfish) | osmFISH somatosensory cortex | **main text** | ARI 0.538, NMI 0.639; two domains lost at the resolution limit |
| [`benchmark_sdmbench/`](benchmark_sdmbench) | SDMBench ranking (Yuan 2024) | **main text** | 1st of twelve on NMI and homogeneity, 2nd on completeness |
| [`stereoseq/`](stereoseq) | Stereo-seq E16.5 embryo | Extended Data | 8 organ-level compartments over ~552 M transcripts |
| [`brain_receptor/`](brain_receptor) | Vizgen MERSCOPE brain | Extended Data | 15 bilaterally symmetric domains, coherence 0.923 |
| [`takei/`](takei) | DNA seqFISH+ and sequential-IF | Extended Data | radial reproducibility r = 0.91; peripheral marks r = +0.72 |
| [`pecab/`](pecab) | PECAb protein panel | Extended Data | coherence 0.968/0.973; cross-replicate r = 0.94 |
| [`atera/`](atera) | 10x Atera breast and cervical | **not in the manuscript** | exploratory; kept for the record |

`common/` holds helpers shared across datasets (cross-validation, shape and
width measures). The comparison against SSAM, Points2Regions and FICTURE lives
in [`../benchmarks/segfree`](../benchmarks/segfree), not here.

## A note on which number to quote

Several results exist under more than one protocol, and they are not
interchangeable. For MERFISH in particular:

- **0.559 / 0.641** — mean over 20 k-means seeds × 5 slices. This is the
  headline and what Table 1 reports.
- **0.557 / 0.631** — a single representative seed, used where a competitor is
  scored under the identical protocol (Table 2).
- **0.628** — pooled *median* NMI over all seeds and slices, used for the
  SDMBench ranking because that is how the benchmark pools.

Each experiment README says which protocol its numbers come from.
