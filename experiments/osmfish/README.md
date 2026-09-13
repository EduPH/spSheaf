# osmFISH — Codeluppi et al. 2018 mouse somatosensory cortex

33-gene panel, 4,839 annotated cells, 11 cortical regions. The raw molecule
table carries 1,976,659 molecules under 35 gene labels (the panel plus Klk6
and Lum) from all 39 hybridization rounds; every method reads the same table.
Molecule coordinates are camera pixels at 15.3846 px/µm and are flipped and
translated relative to the annotation; `register.py` recovers the transform
without the labels (`Methods 4.11`).

Inputs: `data/raw/osmfish/mRNA_coords_raw_counting.hdf5`,
`osmfish_20260512084934.h5ad` (SDMBench annotation),
`osmFISH_SScortex_mouse_all_cells.loom`.

## Run

```bash
make osmfish                 # the reported run (seed 42): Fig 3 and its panel dump
make osmfish-seeds           # 20 initializations at the same configuration
make osmfish-diagnose        # are the two missed layers separable at all? (supervised test, Methods 4.10)
make osmfish-elongation      # alpha_s sweep, per-domain recall, elongation: ED 4
make osmfish-resolution      # domain width vs the kernel's mixing extent: ED 4b
make osmfish-attribute       # eight-corner grid: cost of each departure from the MERFISH values
make osmfish-all             # all of the above in order
make osmfish-insample        # 108-configuration grid, local reference (hours, resumable)
make osmfish-insample-tissue # the same grid with the tissue reference
python -m experiments.osmfish.field_at_boundaries   # the field at layer interfaces vs interiors
```

Configuration (Table 4): δ 150 px (9.8 µm), h 500 px (32.5 µm), σ_s 300 px
(19.5 µm), d = 9, α_s 0.5, local reference — the argmax of the 108 grid.

## Results (in the paper)

- ARI **0.658**, NMI **0.706** (3,716 / 4,839 cells); 20 seeds 0.648 ± 0.012,
  median 0.653. Tissue-reference argmax 0.673 / 0.709. SSAM reaches 0.729 at
  the best of its 144; none of spSheaf's 216 configurations does.
- Two layers are missed at every setting. Layer 3–4 is 73 µm wide against
  ±32.5 µm of kernel mixing, but narrowing h to 19.5 µm leaves its recall at 0
  (ARI 0.658 → 0.576); a supervised classifier on the cell-by-gene matrix
  separates it from Layer 4 at balanced accuracy 0.773 ± 0.081 and Layer 2–3
  lateral from medial at 0.707 ± 0.071. The panel, not the resolution, binds.
- α_s ∈ {0, 0.25, 0.5, 0.75}: ARI 0.609, 0.619, 0.658, 0.593; Layer 2–3
  lateral appears only at 0.75, where Layer 4 and Layer 2–3 medial lose 0.17
  and 0.23 of recall.
- The smoothed field at layer interfaces is 0.72× its within-domain mean; the
  two interfaces bounding the missed layers sit at 0.82×.
- Naive k-means (11 groups, 5 seeds): coordinates 0.223 ± 0.006, expression
  0.185 ± 0.027, both 0.222 ± 0.016.

## Files

`run_osmFISH.py` (run and `--sweep` grid), `register.py`, `seeds.py`,
`tune.py` (`--stage diagnose|rule|cv|attribute`), `elongation.py`,
`resolution.py`, `field_at_boundaries.py`; `ablation.py`, `shape_variants.py`
and `osmFISH-dimensionality.py` are development scripts not in the paper.
Tracked tables: `../../reported/osmfish/`.
