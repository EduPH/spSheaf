# Stereo-seq — E16.5 mouse embryo (MOSTA)

Scalability on whole-transcriptome, whole-section data: ~552 M transcripts
across 28,633 genes over ~115 mm² at ~0.5 µm resolution.

## Run

No `make` target; call the module directly.

```bash
# fine, organ-level (δ = 15 µm)
python -m experiments.stereoseq.run_stereoseq_masked \
  --gem data/raw/stereoseq/E16.5_E1S3_GEM_bin1.tsv.gz \
  --grid 30 --bandwidth 50 --smooth 120 --n-clusters 8 \
  --top-genes 300 --subsample 0.5

# coarse, whole embryo (δ = 100 µm)
python -m experiments.stereoseq.run_stereoseq_masked \
  --gem <same> --grid 200 --n-clusters 48 --top-genes 300 --subsample 0.1
```

Coordinates are bin1 units (1 unit ≈ 0.5 µm), so `--grid 30` is 15 µm. Note
that `--subsample` defaults to 0.1: the fine run needs `--subsample 0.5`
passed explicitly.

The loader streams the GEM file in two passes and subsamples per chunk, so the
section never has to be held in memory. An Otsu tissue mask and largest
connected component strip the background before clustering — without that step
the boundary field is dominated by the tissue edge.

## Results

**Qualitative; there is no spatial-domain ground truth for this section.** The
companion file annotates 25 *cell types*, which is a different partition from
tissue domains, so ARI against it would not mean what it appears to mean. No
metrics table is written.

- Coarse run: 48 tissue-level clusters over ~7,000 grid points.
- Fine run: **8 organ-level compartments** resolved, plus non-tissue as a
  single background domain.
- The ROI path reaches ~443,000 grid points, which is the largest grid the
  pipeline has been run on.

The top-300 gene filter retains 49.1 % of all molecules.

Feeds Extended Data Fig. 5a (as the placed raster figures/assets/ed_stereoseq.png). Settings as run: h = 50 units (25 µm), σ_s = 120 units (60 µm), d = 6, k_nbrs = 30, α_s = 0.3, tissue coordinate reference (the run predates the local reference).
