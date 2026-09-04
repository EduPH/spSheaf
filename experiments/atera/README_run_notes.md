# 10x Atera — sheaf spatial domains (WS3)

Apply the molecule-level pipeline to 10x **Atera** WTA FFPE whole-transcriptome
in-situ data (18,028 genes, subcellular resolution) on human **breast cancer**
and **cervical cancer**. `transcripts.parquet` in the Output Bundle is a cloud of
`(x, y, gene)` detections — a direct fit for the core pipeline (same input
contract as MERFISH / Stereo-seq). No ground-truth annotation, so evaluation is
unsupervised + a **per-domain marker-gene** readout for the biological discussion
(cancer-specific spatial patterns: tumour epithelium / stroma / immune / etc.).

## Data

- Breast: https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer
- Cervical: https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-cervical-cancer

Download each **Output Bundle** and extract `transcripts.parquet`:

```bash
# after downloading <bundle>.zip from the dataset page
unzip <bundle>.zip -d breast_bundle    # -> breast_bundle/.../transcripts.parquet
```

## Run

```bash
# 1) confirm the schema (column names may differ from the Xenium defaults)
python run_atera.py --input breast_bundle/transcripts.parquet --inspect

# 2) run (tune grid/bandwidth to the section; n-clusters is a free parameter)
python run_atera.py --input breast_bundle/transcripts.parquet \
    --grid-spacing 40 --bandwidth 80 --smooth-radius 150 \
    --top-genes 1000 --n-clusters 12 --output-dir output_atera_breast
```

Defaults assume 10x column names (`feature_name`, `x_location`, `y_location`,
`qv`); override with `--gene-col/--x-col/--y-col/--qv-col` if `--inspect` shows
otherwise. QV filtering (`--min-qv 20`) and control/blank-probe removal are on by
default.

## Outputs (in --output-dir)

- `domain_labels.csv` — per-grid-point (x, y, domain)
- `boundary_field.csv` — per-grid-point boundary scores
- `domain_markers.csv` — top enriched genes per domain (for the biology)
- `metrics.csv` — unsupervised quality (silhouette, CH, DB, spatial coherence)
- `domain_map.png` — domains (legend shows each domain's top markers) + boundary field

## Notes / tuning

- **Whole transcriptome (~18k genes):** `--top-genes` bounds the fiber dimension
  (stalk memory ~ N_grid × N_genes × d). 1,000 is a reasonable start; raise for a
  finer signal if memory allows, or coarsen `--grid-spacing`.
- **Scale:** Atera is subcellular; for *domain*-level structure use a coarse
  bandwidth (δ ≈ 40 µm, h ≈ 80 µm here). Shrink for finer structure.
- **Large files:** the parquet is **streamed by row-group** (`--chunk-rows`,
  default 5M) with QV filtering + control removal applied per batch and only the
  compact `(gene, x, y)` kept, so a hundreds-of-millions-of-rows `transcripts.parquet`
  never loads whole. The fiber build is also block-processed (`--block-size`) so
  the radius query never materialises all neighbours at once; use `--subsample`
  if a section is extremely dense.
- **Stalk memory:** scales as N_grid × N_genes × d. For a very large section keep
  `--top-genes` ≤ ~1000 and/or coarsen `--grid-spacing`; e.g. 100k grid points ×
  1000 genes × 6 ≈ 4.8 GB.
- Validated end-to-end on a synthetic 3-region section (domains recovered, marker
  genes correct); pending a run on the real bundles.
