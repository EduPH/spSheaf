# SpaceTx mouse VISp — the two blind sections

Two annotated sections not used in development, from the SpaceTx consortium
(Long et al. 2023): the Allen Institute prototype MERFISH section (268 genes,
900,678 molecules, 2,150 annotated cells in L1–L6b) and the Cold Spring Harbor
BaristaSeq section (81 genes, 319,358 molecules, 4,432 cells in L1–L6 and
white matter). Molecule and cell tables share one frame (checked label-free,
composition correlation 0.94 → 0.97 and 0.89 → 0.90 after ≤ 8 µm shifts).

## Run

```bash
bash experiments/spacetx/download.sh              # spot + cell tables into data/raw/spacetx/ (all, or one name)
python -m experiments.spacetx.inspect MERFISH_Allen_Prototype
make spacetx SPACETX=MERFISH_Allen_Prototype      # one run at the MERFISH values (h 25, sigma_s 60, d 6; SPSHEAF_REFERENCE=local for the transferred row)
make spacetx SPACETX=BARISTASeq
make spacetx-baselines SPACETX=...                # naive k-means baselines
make spacetx-insample-grid                        # 162 configurations per section: reference x h x alpha_s x sigma_s x d (hours, resumable)
make spacetx-insample-panels                      # re-run each section at its argmax; dumps the ED 3 maps
make spacetx-all                                  # all of the above
make spacetx-segfree SPACETX=...                  # SSAM + Points2Regions on the same spot table
make spacetx-ficture SPACETX=... ; bash benchmarks/segfree/run_ficture.sh ... ; python -m benchmarks.segfree.eval_ficture ...
make spacetx-ficture-labels SPACETX=...           # FICTURE's per-cell labels at the Table 1 configuration
make spacetx-ablation                             # the ten-arm edge-score ablation at each section's argmax (Supplementary Fig. 3)
```

`run_spacetx.py` exposes `--spatial-weight`, `--boundary-alpha`,
`--reference`, `--smooth-radius`, `--bandwidth`, `--uniform-dim`.

## Results (Table 1, in-sample argmax of 162 per section)

| Section | spSheaf local / tissue | Points2Regions | SSAM | FICTURE |
|---|---|---|---|---|
| Allen prototype | 0.665 / **0.754** (h 40, σ_s 120, α_s 0.25, d 6) | 0.684 (0.661 ± 0.052 over seeds) | 0.410 | 0.475 |
| BaristaSeq | 0.749 / **0.757** (h 40, σ_s 180, α_s 0.5, d 9) | 0.733 (0.685 ± 0.098) | 0.738 | 0.631 |

50 (Allen) and 34 (BaristaSeq) of the 162 configurations beat the best
competitor. The coordinate reference is the largest marginal effect on both
sections (mean ARI 0.132 and 0.102 apart), the parameters interact (on Allen
the reference alone gives 0.541 → 0.577 at h 25; h 40 given the tissue
reference gives 0.579 → 0.753), and neither the reference nor σ_s has a
label-free rule; the tissue reference, better on both sections and on
osmFISH, is the package default, and Table 1 reports the argmax over both.
BaristaSeq's 0.019 over SSAM is within Points2Regions' seed spread and is not
a separation.

Ablation at the argmax (mean over 5 seeds): on Allen the sheaf score reaches
0.719 against 0.721 for the kernel baseline; on BaristaSeq 0.734 against
0.736, with Procrustes at 0.748. The weighting itself contributes 0.01–0.02
ARI over no weighting on these sections and no edge score separates from the
kernel baseline (`reported/tissue/spacetx/`).

BaristaSeq at k = 6: L1 (32 cells) is recovered by no method; spSheaf's sixth
cluster is a Synpr-high patch within L6 (201 cells, identical over seeds), and
at k = 5 the five remaining layers are recovered at ARI 0.782. In the figure
a cluster that shares no cell with its Hungarian partner is drawn grey.

Naive k-means (coordinates / expression / both): 0.198 / 0.351 / 0.432 on
Allen, 0.257 / 0.114 / 0.255 on BaristaSeq.

## Files

`run_spacetx.py` (`--n-clusters`, `--seed`, `--panel-name` for extra
runs), `sweep.py` (writes `sweep.csv`, resumable), `insample_panels.py`,
`ablation_cache.py` (caches a section in the MERFISH ablation's format, so
`experiments.merfish.sheaf_ablation --cache-path` runs on it), `baselines.py`,
`inspect.py`, `download.sh`. Tracked tables: `../../reported/spacetx/` and
`../../reported/tissue/spacetx/`.
