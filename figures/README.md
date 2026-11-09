# Figure pipeline

Every figure is drawn from the result tables at its final physical size —
180 mm canvas (Nature double column), 6/7/8 pt type (tick labels / text /
panel letters), no frames on spatial panels, rasters ≥ 300 dpi — and the
finished PDFs are checked against that spec, so it holds by construction.
Place figures in the manuscript at 180 mm, never at `\linewidth`.

```bash
make figures          # 17 PDFs into out/ (untracked)
make figures-verify   # asserts canvas, type sizes and raster dpi on the PDFs; fails the build
make figures-test     # unit tests of the pipeline itself
make figures-panels   # every panel as a separate PDF, for hand composition
```

## Script → figure

Numbering is that of the submitted version (`main_spSheaf_nm.tex`; the longer
`main_spSheaf_revised.tex` numbers them differently).

| Script | Figure | Reads |
|---|---|---|
| `fig1_pipeline.py` | Fig. 1 schematic | nothing measured |
| `fig2_manuscript.py` | Fig. 2 | `benchmarks/segfree/results_union/matched_selection_*.csv` (`SPSHEAF_MS_DIR`), `reported/merfish/sheaf_ablation_*`, `boundary_eval_*`, `data/results/merfish/panels/segfree_maps_s0.19` |
| `fig3_osmfish.py` | Fig. 3 | `data/results/osmfish/output_osmFISH_boundary/panels/osmfish.npz` |
| `fig5_ksplit.py` | Fig. 4 (`fig5.pdf`) | `data/results/merfish/k_split_MERFISH_*.npz`, `reported/merfish/k_split_celltypes.csv` |
| `ed_benchmark.py` | ED 1 | `data/results/benchmark_sdmbench/output_benchmark/*_ranking.csv`, `output_seed_sensitivity/` |
| `ed_sensitivity.py` | ED 3 | the three in-sample grids (`reported/` fallback) |
| `ed_spacetx.py` | ED 2 | SpaceTx panel dumps (`make spacetx-insample-panels`; `spsheaf_k5.*` for the k = 5 column), competitor labels |
| `ed_kselection.py` | ED 4 | `reported/synthetic/k_selection_*.csv`, `reported/merfish/k_selection_MERFISH_*.csv` |
| `ed_scale.py` | ED 5 | `data/results/brain_receptor/output_brain_receptor/*.csv` + `assets/ed_stereoseq.png` |
| `ed_composites.py` | ED 6, ED 7 | Takei results (`ed_ifheat_panels.py`, `ed_nucleus_panels.py`) + `assets/dna_*.png` |
| `ed_pecab.py` | ED 8 | `data/results/pecab/output_pecab_signaling_Rep{1,2}/panels/pecab.npz` |
| `ed_merfishslices.py` | Supplementary Fig. 1 | the MERFISH cache (`make cache`) |
| `ed_multislice.py` | Supplementary Fig. 2 | `data/results/merfish/panels/multislice.*` |
| `sup_ablation_spacetx.py` | Supplementary Fig. 3 | `reported/tissue/spacetx/sheaf_ablation_*.csv` (`SPSHEAF_DATA` at the tissue root) |
| `sup_ksplit.py` | Supplementary Table 4's figure companion (not placed) | `reported/merfish/k_split_events.csv` |
| `ed_osmfish_modes.py` | not in the submitted version | `data/results/osmfish/elongation.csv` |

`out_local/` and `out_tissue/` are untracked snapshots of `out/` built with
the results of each coordinate reference.

Each script prints the exact command for a missing input. `assets/` holds
the four rasters still placed rather than drawn (`_unused/` the retired
ones).

## Writing a panel

```python
import sheafstyle as ss
ss.apply()
fig, ax = ss.panel(52, 42)                  # millimetres, exact; never bbox_inches="tight"
fig.subplots_adjust(left=0.26, right=0.98, top=0.94, bottom=0.32)
ax.bar(names, values, color=[ss.METHOD_COLORS[n] for n in names])
ss.save_panel(fig, "panels/x.pdf")          # checks spines on spatial axes before writing

fig, ax = ss.panel(60, 46); ss.spatial(ax)  # spatial: no frame, no ticks, equal aspect
sc = ax.scatter(x, y, c=colors, s=0.35, linewidths=0); ss.rasterize(sc)   # marks raster, text vector
```

`layout.py` composes panels into a figure declared in mm (`Layout`, `Panel`;
`y` from the top edge); it refuses overlapping or off-canvas panels and draws
the panel letters itself. `verify_figure.py` reads type sizes from the PDF's
font operators, not glyph boxes. `sheafstyle.METHOD_COLORS` and
`arm_style()` fix the colours for methods and ablation arms.
