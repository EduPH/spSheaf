# SpaceTx mouse VISp — candidate third annotated benchmark

Why: the manuscript's quantitative evidence is MERFISH (five slices of one
animal) and osmFISH (one section). Memo point 9 asks for an independent,
quantitatively annotated dataset. The SpaceTx consortium profiled the *same*
mouse primary visual cortex with five imaging platforms — BaristaSeq, ExSeq,
an Allen-prototype MERFISH (254 genes), ISS and multiplexed smFISH — and
publishes a per-molecule `spot_table.csv` for each, alongside cell tables.
Cortical layers are the natural domain annotation, and SDMBench's Data22–24
are SpaceTx datasets, so published baseline scores exist.

## Current state (11 Sep 2026) — read this before the dated entries below

Both sections are in the manuscript as Table 2 columns and as ED Fig. 6. The
dated sections further down are a working log and several of their numbers have
been superseded; where they disagree with this section, this section is right.

What changed after the entries below were written: the grids were re-matched to
the competitors' search budgets, and the coordinate reference was added to the
sweep. Both mattered. SSAM's own in-sample grid here is 287–288 settings
(resolution treated as the hyperparameter it is, plus the sctransform variants),
against the 71 recorded below, and spSheaf's was 12–18. Under that asymmetry
spSheaf was being selected from a grid an order of magnitude smaller than the
method it was compared with, which is not a fair comparison in either direction.

The matched grid is 162 configurations per section, run by
`make spacetx-insample-grid`:

| parameter | levels |
|---|---|
| coordinate reference | local, tissue |
| kernel bandwidth `h` | 25, 40, 60 µm |
| smoothing radius `σ_s` | 60, 120, 180 µm |
| spatial weight `α_s` | 0.25, 0.5, 0.75 |
| stalk dimension `d` | 4, 6, 9 |
| boundary strength `α_b` | 0.5 (fixed) |

ARI against the layer annotation, Hungarian-matched:

| section | spSheaf transferred | spSheaf in-sample (of 162) | best competitor, in-sample |
|---|---|---|---|
| Allen prototype MERFISH | 0.541 | **0.754** | 0.684 (Points2Regions, of 24) |
| BaristaSeq | 0.713 | **0.757** | 0.738 (SSAM, of 288) |

The in-sample argmax is the tissue reference on both sections, at `h` 40 µm:
σ_s 120, d 6, α_s 0.25 on Allen; σ_s 180, d 9, α_s 0.5 on BaristaSeq.

The surprise is which parameter dominates. It is not the bandwidth, as the
"Bandwidth rule" section below concluded from the 12-setting grid — it is the
coordinate reference, which moves mean ARI by 0.132 on Allen and 0.102 on
BaristaSeq against 0.036 and 0.081 for the bandwidth. The stalk dimension moves
it by 0.010 and 0.004, i.e. not at all. ED Fig. 5 draws all of this
(`figures/ed_sensitivity.py`); read its marginals only over balanced blocks,
which that script enforces, because an unbalanced marginal on a resumable grid
reads whatever order the configurations happened to finish in.

We have no label-free rule that picks the reference or σ_s across all four
datasets, so both are fixed on MERFISH and transferred, and the sensitivity is
reported rather than used. That is why there are two spSheaf rows in Table 2.

Reproducing, in order:

```bash
make spacetx SPACETX=MERFISH_Allen_Prototype     # the transferred configuration
make spacetx SPACETX=BARISTASeq
make spacetx-segfree SPACETX=...                 # SSAM + Points2Regions
make spacetx-insample-grid                       # the 162 configurations, resumable
make spacetx-insample-panels                     # re-run each at its own argmax, for the figure
cd figures && python ed_spacetx.py
```

`spacetx-insample-panels` reads the argmax out of `sweep.csv` rather than
taking it as an argument, so the figure and Table 2 cannot disagree; it writes
`spacetx/<ds>/panels/spsheaf_insample.{npz,json}`. The grid tables are archived
under `reported/spacetx/` so ED Fig. 5 builds without them.

## Step 1 — download (needs a network; run from a terminal on your Mac)

```bash
bash experiments/spacetx/download.sh            # all five, ~minutes
bash experiments/spacetx/download.sh BARISTASeq # or one
```

The script prints the first lines of every table so the next two questions
can be answered before any code is written.

## Step 2 — inspect, then run

```bash
python -m experiments.spacetx.inspect MERFISH_Allen_Prototype
```
prints the columns, extents, unit guesses and candidate annotation columns of
the three tables. Then, with the column names it shows:

```bash
python -m experiments.spacetx.run_spacetx MERFISH_Allen_Prototype \
    --spot-x <x col> --spot-y <y col> --spot-gene <gene col> \
    --cell-id <id col> --cell-x <x col> --cell-y <y col> --label <layer col> \
    --unit-um <um per unit> --grid-spacing 10 --bandwidth 25 --uniform-dim 6
```
`run_spacetx.py` runs the pipeline blind at the MERFISH sheaf defaults,
registers the annotation onto the grid label-free (composition correlation,
as for osmFISH) and scores at cell level; it writes
`data/results/spacetx/<dataset>/metrics.csv` and a panel dump. If the layer
labels are not in the mapped cell table, pass `--label-csv` with a two-column
file `<cell id>,<layer>` exported from the SDMBench h5ad.

## What to check before committing effort

1. **Is there a layer label per cell?** Look in `s3_mapped_cell_table.csv`
   and the Readme for a `layer` / `region` / `cortical_depth` column. If only
   a depth-from-pia is given, layers can be assigned from depth with the
   standard VISp layer boundaries, but that is a derived annotation and must
   be stated as such. Alternatively take the ground truth from SDMBench's
   Data22–24 `.h5ad` files (Yuan et al. 2024, Zenodo), which is what the
   published baselines were scored against — preferable, because then the
   ranking is on the same labels.
2. **Are the spot table and the cell table in the same frame and unit?**
   Compare spot-cloud extent with cell-centroid extent. osmFISH needed a
   flip-plus-translation and a px→µm factor (`benchmarks/segfree/osm_register.py`);
   expect the same here and reuse that composition-correlation check.

## Step 3 — which one first

`MERFISH_Allen_Prototype` (254 genes; closest to the headline benchmark) and
`BARISTASeq` (most independent chemistry). ExSeq is ~1 mm² and 42 genes —
small; smFISH is 22 genes — probably below the stalk dimension needed.

## Step 4 — run under the matched protocol

Reuse `experiments/osmfish/run_osmFISH.py` as the template (single section,
cell-level scoring, leave-one-spatial-block-out CV in `experiments/common/cv.py`)
and `benchmarks/segfree/` for SSAM / Points2Regions / FICTURE on the same spot
table. Parameters carried over from MERFISH unchanged, as for osmFISH; only
δ, h, d and k set from the platform (Methods 4.13).

## First result — MERFISH_Allen_Prototype (5 Sep 2026)

Unpublished Allen Institute prototype MERFISH of mouse VISp, Zhuang MOp gene
panel (268 genes in the spot table, 253 in the cell table), 900,678 spots,
2,150 segmented cells annotated into six cortical layers (L1 9 cells, L2/3 585,
L4 414, L5 504, L6a 498, L6b 140), 1.3 × 1.1 mm. Spots `xc`/`yc` and cells
`rotated_x`/`rotated_y` share one micrometre frame (label-free registration:
composition r 0.94 at centroid alignment, 0.97 after an 8 µm shift).

spSheaf at the MERFISH sheaf defaults, δ = 10 µm, h = 25 µm, d = 6, k = 6,
blind (`run_spacetx.py`): **ARI 0.578, NMI 0.702**, identical over seeds
0, 1, 42. Per-layer recall: L4 0.90, L6a 0.76, L5 0.67, L2/3 0.54, L1 0.00
(9 cells), L6b 0.00 (140 cells, a thin band under L6a).

Naive baselines, same protocol, 20 initializations (`baselines.py`):
coordinates only 0.198 ± 0.002, expression only 0.347 ± 0.028, concatenated
0.432 ± 0.002 (NMI 0.347 / 0.370 / 0.471).

### Competitors on the same spot table (5 Sep 2026, run in the cloud sandbox)

| method | rule | ARI | NMI |
|---|---|---|---|
| spSheaf, MERFISH defaults (h 25 µm) | transfer, blind | 0.578 | 0.702 |
| spSheaf, best of a 12-setting grid (h 40 µm, α_s 0.5, σ_s 120) | in-sample argmax | **0.753** | **0.778** |
| Points2Regions, best of 24 (pw 5, ps 15) | in-sample argmax | 0.684 | 0.726 |
| Points2Regions, 5 seeds at that setting | | 0.661 ± 0.052 | |
| SSAM, best of 71 `find_domains` settings | in-sample argmax | 0.410 | 0.492 |

Points2Regions is strong on this clean layer stack: 11 of its 24 settings beat
spSheaf's transferred default, and its grid median is 0.510. spSheaf's grid
(`sweep.py` as it then stood; h ∈ {25, 40}, α_s ∈ {0.25, 0.5, 0.75},
σ_s ∈ {60, 120} — 12 settings, superseded by the 162 above) is bimodal in the bandwidth: every h = 25 setting is 0.55–0.58,
every h = 40 setting with α_s ≤ 0.5 is 0.740–0.753, so the in-sample bound is
not a lucky corner but the bandwidth. Under the symmetric in-sample rule
spSheaf leads Points2Regions by 0.07 ARI / 0.05 NMI; under transfer-vs-tuned it
trails by 0.11. SSAM (v1.1.3: outlier filter off, resolution 1.0 → 13 cell
types; `norm_thres` 300–1000) is far behind either.

Installing SSAM (its build needs numpy present and a current setuptools):
`python -m venv env && . env/bin/activate && pip install -U pip setuptools wheel && pip install numpy cython && pip install --no-build-isolation ssam louvain "setuptools<81"`.

Not yet done, as of that entry: FICTURE; h = 60 rows of the grid; BaristaSeq.
All three are done — see the current-state section at the top. Still open: a
label-free rule for the reference and σ_s, and the other three SpaceTx
platforms. (An earlier note here said the
per-gene density is ~4x lower than Moffitt MERFISH; it is the reverse — see
"Bandwidth rule" below.)

## Competitors on the same spot table

```bash
pip install -e ".[benchmark]"          # ssam, points2regions
make spacetx-segfree                   # SSAM + Points2Regions, matched protocol
make spacetx-ficture                   # writes FICTURE's input under data/cache/segfree/ficture_input/spacetx_<ds>/
bash benchmarks/segfree/run_ficture.sh data/cache/segfree/ficture_input/spacetx_MERFISH_Allen_Prototype \
     data/results/segfree/ficture/spacetx_MERFISH_Allen_Prototype 6,12,20 12,18
python -m benchmarks.segfree.eval_ficture data/results/segfree/ficture/spacetx_MERFISH_Allen_Prototype \
     data/cache/segfree/ficture_input/spacetx_MERFISH_Allen_Prototype
```

`benchmarks/segfree/run_spacetx.py` places the cells with the label-free
translation stored by `run_spacetx.py`, so all methods are scored against the
same registration; SSAM uses its authors' MERFISH-notebook settings plus the
MERFISH-benchmark `find_domains` sweep (best in-sample reported, as for
osmFISH), Points2Regions a 4 x 6 grid over pixel width x smoothing with k fixed
to the annotated number. Results: `data/results/segfree/spacetx_<ds>/`.

## Second result — BARISTASeq (5 Sep 2026)

Cold Spring Harbor (Zador lab) BaristaSeq of mouse VISp: 319,358 spots, 81
genes, 4,432 cells in six classes (L1 32, L2/3 1,065, L4 997, L5 906, L6 1,165,
white matter 267), 0.8 × 1.3 mm; same table layout as the Allen prototype, so
the default column flags apply. Frames aligned (composition r 0.89 → 0.90 at a
4 µm shift).

| method | rule | ARI | NMI |
|---|---|---|---|
| spSheaf, MERFISH defaults (h 25 µm, d 6) | transfer, blind | **0.724** | 0.747 |
| spSheaf, best of 12 settings (h 25, α_s 0.25, σ_s 120) | in-sample argmax | 0.745 | 0.756 |
| spSheaf, best NMI (h 40, α_s 0.5, σ_s 120) | in-sample | 0.740 | **0.779** |
| Points2Regions, best of 24 (pw 10, ps 6) | in-sample argmax | 0.733 | 0.771 |
| Points2Regions, 5 seeds at that setting | | 0.685 ± 0.098 | |
| SSAM, published MERFISH settings | | 0.435 | 0.583 |
| SSAM, best of 71 `find_domains` settings | in-sample argmax | 0.703 | 0.741 |
| naive: coordinates / expression / both | | 0.257 / 0.114 / 0.255 | |

Per-layer recall at the default: white matter 0.98, L4 0.94, L2/3 0.93, L6
0.74, L5 0.71, L1 0.00 (32 cells). Here the transferred default is within 0.01
of Points2Regions' tuned best and above SSAM's; only 1 of 24 Points2Regions
settings beats it, and its five seeds span 0.55–0.82. spSheaf's grid is flat
in the bandwidth on this dataset (h 25 and 40 both 0.72–0.75 for α_s ≤ 0.5),
unlike the Allen prototype, where h = 40 was needed.

## FICTURE on both (5 Sep 2026, Eduardo's Mac; `make spacetx-ficture` + `run_ficture.sh` + `eval_ficture.py`)

Six models per dataset (6/12/20 factors × 12/18 µm training width), three
cell-label mappings, five seeds; best in-sample row reported, as for the other
competitors. Metric tables: `data/results/segfree/ficture/metrics_spacetx_<ds>.csv`.

| dataset | FICTURE best ARI / NMI | model | at the annotated k |
|---|---|---|---|
| MERFISH_Allen_Prototype | 0.490 / 0.504 | 20 factors, k-means mapping | same (k = 6) |
| BARISTASeq | 0.631 / 0.632 | 20 factors, argmax mapping (15 domains) | 0.621 / 0.661 (k = 6) |

### Summary across the two independent datasets

| | spSheaf blind | spSheaf in-sample | Points2Regions | SSAM | FICTURE |
|---|---|---|---|---|---|
| Allen prototype MERFISH | 0.578 / 0.702 | 0.753 / 0.778 | 0.684 / 0.726 | 0.410 / 0.492 | 0.490 / 0.504 |
| BaristaSeq | 0.724 / 0.747 | 0.745 / 0.779 | 0.733 / 0.771 | 0.703 / 0.741 | 0.631 / 0.632 |

Competitor columns are their best in-sample setting (the permissive rule).

## Bandwidth rule — what the numbers say (5 Sep 2026, evening)

Per-gene molecule density does NOT explain which h works. Using the exact
inputs spSheaf sees (Moffitt = the single z-plane kept by `_cache.py`; convex
hull areas):

| dataset | molecules | genes | area mm² | mol / gene / mm² | cell spacing (1/√cell density) | h used → ARI |
|---|---|---|---|---|---|---|
| Moffitt MERFISH s0.19 | 263,046 | 135 | 3.23 | 604 | 15.1 µm | 25 (h/s 1.66) |
| osmFISH | 1,976,659 | 33 | 3.49 | 17,200 | 26.9 µm | 32.5 (1.21); 16.2 costs −0.053 |
| Allen prototype | 900,678 | 268 | 1.25 | 2,694 | 24.1 µm | 25 → 0.58 (1.04); 40 → 0.75 (1.66) |
| BaristaSeq | 319,358 | 81 | 0.91 | 4,314 | 14.3 µm | 25 → 0.72 (1.75); 40 → 0.74 (2.8) |

Moffitt has the lowest per-gene density and the smallest good h, so a
counts-per-kernel rule points the wrong way. What the three well-tested
datasets share is h ≈ 1.7 × mean cell spacing (Moffitt 1.66, Allen 1.66 at
the good h, BaristaSeq 1.75 at the default, flat above). Prediction to test:
osmFISH at h ≈ 45 µm (1.7 × 26.9) should score ≥ the 32.5 µm default. A
segmentation-free proxy for cell spacing is the mean nearest-neighbour
distance between local maxima of the total molecule-density KDE (SSAM's
local-max step), which needs no cells and no labels. Untested; not in the
manuscript. The manuscript currently says only that no rule fits all four.

## Figure — Extended Data ed_spacetx (5 Sep 2026)

`figures/ed_spacetx.py` draws both sections per cell (annotation, ranked
sheaf field, spSheaf transferred, competitors at their best in-sample setting,
Hungarian-matched) plus the Allen in-sample run at h = 40. It needs
`make spacetx-ficture-labels SPACETX=<ds>` (writes `labels_ficture_best.csv`
by reproducing the best metric row exactly) and the h = 40 panel dump, which
`make spacetx-insample-panels` writes `spacetx/<ds>/panels/spsheaf_insample.{npz,json}`;
copy its `panels/spsheaf.{npz,json}` to `panels/spsheaf_h40.{npz,json}` of the
main results folder (that is where the figure reads them; done once by hand
on 5 Sep 2026, reproduced 0.753/0.778).
