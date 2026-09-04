# Data

Nothing in this folder is tracked except this file. Raw inputs are third-party
and large; everything under `cache/` and `results/` is produced by the
experiment scripts.

## Layout the code expects

```
data/
  raw/<dataset>/      inputs you download (see the table below)
  cache/<dataset>/    intermediates written by the two-stage runs
  results/<dataset>/  metrics, labels and panel data
```

The root is resolved by `spsheaf.paths`, which checks, in order: the
`SPSHEAF_DATA` environment variable, a `paths.toml` beside `pyproject.toml`
(copy `paths.example.toml`), and finally `<repo>/data`. No script contains an
absolute path — if one fails to find its input it says which file it wanted.

Check what is present before running anything:

```bash
python -m benchmarks.segfree.inputs     # the benchmark's declared inputs
```

## Data availability

| Dataset | Source | Scale | Expected under `data/raw/` |
|---|---|---|---|
| **MERFISH** mouse hypothalamic preoptic region (Moffitt et al. 2018) | Dryad [`10.5061/dryad.8t8s248`](https://doi.org/10.5061/dryad.8t8s248). The processed molecule tables used here were prepared by Y. Imoto and are available on reasonable request. | 135 genes, ~20.5 M molecules, 5 slices, 8 annotated domains | `merfish/moffitt2018_export_full/molecules_full_add.csv.gz` |
| **osmFISH** mouse somatosensory cortex (Codeluppi et al. 2018) | <http://linnarssonlab.org/osmFISH> | 33 genes, 4,839 cells, 11 annotated regions | `osmfish/mRNA_coords_raw_counting.hdf5`, `osmfish/osmfish_20260512084934.h5ad`, `osmfish/osmFISH_SScortex_mouse_all_cells.loom` |
| **Stereo-seq** E16.5 mouse embryo (Chen et al. 2022) | MOSTA <https://db.cngb.org/stomics/mosta>, accession `STDS0000058` | ~552 M transcripts, 28,633 genes, ~115 mm² | `stereoseq/E16.5_E1S3_GEM_bin1.tsv.gz` |
| **MERSCOPE** mouse brain receptor showcase | Vizgen public showcase data | 54.7 M molecules, 649 genes, 7 focal planes | `brain_receptor/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_detected_transcripts_S1R1.csv` |
| **DNA seqFISH+** mouse ES cells (Takei et al. 2021) | Zenodo [`10.5281/zenodo.3735329`](https://doi.org/10.5281/zenodo.3735329) (~144 MB) | ~3,600 loci per nucleus, 197 + 200 nuclei | `takei/dna_seqfish/DNAseqFISH+/DNAseqFISH+25kbloci-E14-replicate1.csv` |
| **Sequential immunofluorescence** (Takei et al. 2021) | Same Zenodo record (~3.3 GB) | 21 chromatin marks, 46 nuclei | `takei/IF-binned1x1x1-E14-replicate1-Pos0.zip` **and** `takei/IF_supp_table_hyb_targets.xlsx` |
| **PECAb** uterine carcinosarcoma (Fujii et al. 2024) | Zenodo [`10.5281/zenodo.8348965`](https://doi.org/10.5281/zenodo.8348965) | 35-antibody panel + 220-gene panel, 7,657 cells, 2 replicates | `pecab/PECAb_dataset/6_Carcinosarcoma_33x220plex/Raw/6_raw_SiPECAb_Rep{1,2}.csv` |
| **10x Atera** breast and cervical FFPE | 10x Genomics: [breast](https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-breast-cancer), [cervical](https://www.10xgenomics.com/datasets/atera-wta-ffpe-human-cervical-cancer) | ~18,000 genes; 740 M and 1.24 B transcripts | `atera/{breast,cervical}/.../transcripts.parquet` |

The SDMBench comparison additionally uses the Fig. 2d source data of Yuan et
al. 2024, which is small enough to ship: it is tracked at
`data/results/benchmark_sdmbench/figure2b&2d/figure2.feather` in the working
repository and must be copied in alongside the raw data.

## Reproducing the comparison against SSAM, Points2Regions and FICTURE

`benchmarks/segfree/` scores the three closest segmentation-free methods on
the same molecule tables. Its stored label and metric tables are the
provenance for the published comparison and are tracked; the sweeps that
produced them are **not** re-runnable from this repository alone, because
three declared inputs are absent or in a different format
(`benchmarks/segfree/inputs.py` names them). Treat those tables as data, not
as a pipeline.
