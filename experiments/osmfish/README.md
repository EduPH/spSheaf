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
make osmfish                 # the local-reference run (seed 42): Fig 3 and its panel dump
make osmfish-seeds           # 20 initializations at the same configuration
make osmfish-diagnose        # are the two missed layers separable at all? (supervised test, Methods 4.10)
make osmfish-elongation      # alpha_s sweep, per-domain recall, elongation (not in the submitted version)
make osmfish-resolution      # domain width vs the kernel mixing extent (not in the submitted version)
make osmfish-attribute       # eight-corner grid: cost of each departure from the MERFISH values
make osmfish-all             # all of the above in order
make osmfish-insample        # 108-configuration grid, local reference (hours, resumable)
make osmfish-insample-tissue # the same grid with the tissue reference
python -m experiments.osmfish.field_at_boundaries   # the field at layer interfaces vs interiors
```

Configuration in the paper (Supplementary Table 2): the argmax of the 2 × 108
grid, which is at the tissue reference — δ 150 px (9.8 µm), h 300 px
(19.5 µm), σ_s 900 px (58.5 µm), d = 9, α_s 0.3 — ARI 0.673 ± 0.001 over
seeds. The local-reference argmax (h 32.5 µm, σ_s 19.5 µm, α_s 0.5) is 0.658
± 0.013. `make osmfish-seeds` and `make osmfish-elongation` take the same
scale flags as `run_osmFISH.py` (`--bandwidth`, `--smooth-radius`,
`--spatial-weight`, `--uniform-dim`), so either configuration can be re-run;
the tissue-reference runs need `SPSHEAF_REFERENCE=tissue` (the default) and
their tables are under `../../reported/tissue/osmfish/`.

## Results (in the paper)

- ARI **0.673**, NMI **0.709** at the tissue-reference argmax (0.658 / 0.706
  at the local one). SSAM reaches 0.729 at the best of its 144; none of
  spSheaf's 216 configurations does.
- Two layers (Layer 3–4 and Layer 2–3 lateral) are missed at every
  configuration and initialisation. The diagnostics below were run and are
  kept here, but since 29 Sep 2026 the paper reports only the miss (the
  private repository's `docs/notes/osmfish_diagnostics_unreported.md` has the
  numbers): supervised separability of the two layers (balanced accuracy
  0.773 ± 0.081 and 0.707 ± 0.071), the resolution test (h 32.5 → 19.5 µm
  leaves recall at 0, ARI 0.658 → 0.576), the α_s sweep (0.609, 0.619, 0.658,
  0.593 at the local scales; 0.665, 0.679, 0.620, 0.594 at the tissue ones)
  and the field at the layer interfaces (0.72× its within-domain mean, 0.82×
  at the two missed interfaces).
- Naive k-means (11 groups, 5 seeds): coordinates 0.223 ± 0.006, expression
  0.185 ± 0.027, both 0.222 ± 0.016.

## Files

`run_osmFISH.py` (run and `--sweep` grid), `register.py`, `seeds.py`,
`tune.py` (`--stage diagnose|rule|cv|attribute`), `elongation.py`,
`resolution.py`, `field_at_boundaries.py`; `ablation.py`, `shape_variants.py`
and `osmFISH-dimensionality.py` are development scripts not in the paper.
Tracked tables: `../../reported/osmfish/`.
