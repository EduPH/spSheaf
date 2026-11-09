# SDMBench — placing the MERFISH result in context

Rebuilds the comparison from the source data of Fig. 2d of Yuan et al. 2024,
which reports 13 spatial-domain methods; 11 have MERFISH results, and this
adds ours to make twelve.

## Run

```bash
python -m experiments.benchmark_sdmbench.seed_sensitivity \
  --mol-path data/raw/merfish/moffitt2018_export_full/molecules_full_add.csv.gz \
  --seeds 0-19 --output-dir data/results/benchmark_sdmbench/output_seed_sensitivity

python -m experiments.benchmark_sdmbench.gt_calibration --mol-path <same>
python -m experiments.benchmark_sdmbench.benchmark_fig2d \
  --feather "figure2b&2d/figure2.feather" --biotech MERFISH --metrics all
```

Needs the MERFISH molecule table and the benchmark's own
`figure2.feather`, which is small enough to keep in the repository.

## Results

**Rank of twelve: 1st on NMI, 1st on homogeneity, 2nd on completeness**;
mid-field (≈5th) on the three spatial-continuity metrics CHAOS, PAS and ASW.

Pooled medians over all seeds and slices (section input, 9 Sep 2026): NMI 0.651; next best on NMI is 0.565.

> The ranking uses **pooled medians** because that is how the benchmark pools;
> Table 1 of the paper reports the **20-seed mean**, 0.655. Same runs, two
> summary statistics.

**Seed sensitivity** over 20 seeds × 5 slices: ARI 0.571 ± 0.000, NMI
0.655 ± 0.000 (per-slice s.d. ≤ 0.001), about 30 s per slice.

**Two metrics are excluded from the ranking, and `gt_calibration.py` is why.**
Recomputed from molecule-level data through this code path, the *ground-truth
partition itself* scores Moran's I 0.109 against 0.153–0.245 for the
baselines — it ranks last on its own annotation. The absolute values reflect
the benchmark's internal cell-expression preprocessing, which cannot be
reproduced from molecules. Our domains score 0.180 against the ground truth's 0.109, so Moran's I and Geary's C are reported for reference and not
ranked. This is a property of the metric, not of any method.

Feeds Extended Data Fig. 1 (the SDMBench ranking and seed sensitivity).
