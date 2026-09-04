# Takei et al. 2021 — two subnuclear experiments in 3D

Both use the **continuous feature-fiber** variant: the fiber at a grid point is
a smoothed vector of measured intensities rather than a molecule count. Same
grid, graph, stalks, restriction maps, disagreement and clustering as
everywhere else. δ = 0.55 µm, h = 0.8 µm, 5 zones. No `make` targets.

`README_phase_notes.md` keeps the development history, including a caveat
about sample size worth reading before quoting anything.

---

## DNA seqFISH+ — chromosome territories

~3,600 genomic loci per nucleus at 103 nm × 103 nm × 250 nm; nuclei ~19 × 14 ×
3.5 µm. 197 nuclei in replicate 1, 200 in replicate 2.

```bash
python -m experiments.takei.run_takei_features \
  --input data/raw/takei/dna_seqfish/DNAseqFISH+/DNAseqFISH+25kbloci-E14-replicate1.csv \
  --features chrom --grid-nm 550 --bandwidth-nm 800 --n-zones 5 --max-cells 200 \
  --output-dir data/results/takei/output_takei_feat_full1

python -m experiments.takei.certify_territories \
  --data <same csv> --zones-dir <that dir> --out-dir <that dir>
```

**Results.** Zones carved from chromosome-intensity fibers recover chromosome
territories: zone↔chromosome NMI **0.569** (rep 1) and **0.571** (rep 2), zone
purity 0.29. Per-chromosome mean radial position reproduces across the two
replicates at **Pearson r = 0.91** (p = 2.7 × 10⁻⁸, Spearman ρ = 0.85).
chr19 is the most interior (radial 0.484, and the most gene-dense at 12.9);
chr1 and chr2 the most peripheral (0.637, 0.611).

> The r = 0.91 needs the full sample. At 22 nuclei the same correlation was
> 0.37 and not significant. `README_phase_notes.md` records this.

Feeds Extended Data Figs. 7 and 8.

---

## Sequential immunofluorescence — chromatin-mark zones

21 chromatin marks over 46 nuclei, 150k–700k voxels each (median ~340k),
giving 230 zones.

```bash
python -m experiments.takei.run_takei_if \
  --zip data/raw/takei/IF-binned1x1x1-E14-replicate1-Pos0.zip \
  --excel data/raw/takei/IF_supp_table_hyb_targets.xlsx \
  --max-cells 46 --n-zones 5 --normalize-voxel \
  --output-dir data/results/takei/output_takei_if

python -m experiments.takei.certify_if --dir <that dir>
```

Both files are needed — the Excel sheet maps marks to hyb/channel. The zip's
inner folder is labelled "replicate2" (an upstream naming quirk), which is why
`--sheet` defaults to `"E14 replicate 2"`.

**Results.** The zones are radially organized, which is what the biology
predicts and what nothing in the pipeline was told:

- peripheral-mark score against radius, **r = +0.716 ± 0.158**, positive in
  **46 of 46** nuclei; pooled over all 230 zones, r = +0.701
- the peripheral-most zone is lamina- or heterochromatin-associated in
  **45 of 46 nuclei (98 %)**
- interior marks correlate at −0.162 and are negative in only 31 of 46, which
  is expected: the nucleolus sits interior in some nuclei and against the
  periphery in others

Pooling all nuclei gives 5 zone archetypes: constitutive heterochromatin,
repressive, baseline interior, lamina-associated, and active / nuclear
speckle.

Feeds Extended Data Figs. 9 and 10.
