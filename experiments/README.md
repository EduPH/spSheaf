# Experiments

One folder per dataset; each README gives the commands, the inputs and the
numbers that reach the paper. All read the data root set by `SPSHEAF_DATA`
(see `../data/README.md`) and write under `data/results/<dataset>/`.

| Folder | Dataset | In the paper | Headline |
|---|---|---|---|
| `merfish/` | Moffitt 2018 MERFISH hypothalamus, 5 slices | Table 1, Fig. 2, ED 1, ED 3 | ARI 0.571 / NMI 0.655 (20 seeds); ablation +0.102 ARI |
| `benchmark_sdmbench/` | SDMBench ranking (Yuan 2024) | ED 2 | 1st of 12 on NMI (0.651 vs 0.565) |
| `osmfish/` | osmFISH somatosensory cortex | Fig. 3, ED 4, ED 5 | ARI 0.658 / NMI 0.706; two layers not separable in the panel |
| `spacetx/` | SpaceTx VISp: Allen prototype MERFISH, BaristaSeq | Table 2, ED 5, ED 6 | ARI 0.754 and 0.757 over 162-configuration grids |
| `stereoseq/` | Stereo-seq E16.5 embryo | ED 7a | 8 organ-level compartments over 552 M transcripts |
| `brain_receptor/` | Vizgen MERSCOPE brain hemisection | ED 7b–d | 15 domains; ARI 0.264 vs the Allen CCF (rigid) |
| `takei/` | DNA seqFISH+ and sequential-IF nuclei (3D) | ED 8, ED 9 | radial order r = 0.91 across replicates; r = +0.72 in 46/46 nuclei |
| `pecab/` | PECAb protein panel | ED 10 | 5 domains; cross-replicate marker r = 0.94 |
| `atera/` | 10x Atera breast and cervical | not in the paper | exploratory, kept for the record |

`common/` holds shared helpers (block cross-validation, shape and width
measures). The comparison against SSAM, Points2Regions and FICTURE is in
`../benchmarks/segfree/`.

Which number to quote: spSheaf on MERFISH is the 20-seed mean (0.571; seed
spread ≤ 0.001); on the three single sections it is a single run at seed 42,
the rule every method in Table 2 is under. Numbers under different selection
rules (per-slice argmax 0.582, one configuration 0.571, leave-one-slice-out
0.552) are not interchangeable; `../benchmarks/segfree/README.md` defines them.
