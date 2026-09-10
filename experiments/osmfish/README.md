# osmFISH — mouse somatosensory cortex

Second annotated benchmark, and the one that shows the method's resolution
limit. 33 genes, 4,839 cells, 11 annotated cortical regions, 44,410 grid
points.

## Run

```bash
python -m experiments.osmfish.register   # ONCE: label-free registration of the annotation (writes data/results/osmfish/registration.json)
make osmfish            # metrics
make panels-osmfish     # same run, also dumps the data Figure 3 is drawn from

make osmfish-diagnose | osmfish-rule | osmfish-cv | osmfish-attribute
make osmfish-resolution | osmfish-elongation
```

Needs `data/raw/osmfish/mRNA_coords_raw_counting.hdf5` and
`osmfish_20260512084934.h5ad`. Settings: δ = 150 px (9.8 µm), h = 500 px
(32.5 µm), σ_s = 300 px, d = 9, k = 11.

Three corrections are load-bearing and easy to get wrong: the annotation lives
in the loom coordinate frame, which is **axis-flipped** relative to the imaging
frame (recovered without labels — mean per-cell composition correlation 0.83
flipped against 0.07 unflipped); the scale is 15.3846 px/µm; and the residual
**translation** between the annotation and the molecule grid must be fitted,
not assumed. `run_osmFISH.py` used to match the two centroids, which is biased
by (109, −74) µm because the molecule cloud extends beyond the annotated cells
asymmetrically; `register.py` fits the translation label-free (composition
correlation 0.53 vs 0.17 for centroid matching), stores it, and every osmFISH
experiment reads it. That single fix moved the same partition from ARI 0.538 to
0.620 (5 Sep 2026). Every number below is with the fitted registration.

## Results

**ARI 0.620, NMI 0.694** at cell level (default initialization); 3,678 of 4,839 cells correct.

Naive baselines under the same protocol: coordinates only 0.223 ± 0.006,
expression only 0.185 ± 0.027, the two concatenated 0.222 ± 0.016.

SSAM reaches ARI 0.566 / NMI 0.721 here: spSheaf leads on ARI, SSAM on NMI.
osmFISH is one of SSAM's own primary published analyses and its released
settings were tuned on it; spSheaf runs at the MERFISH-selected defaults.

**Per-domain recall** — the interesting part:

| domain | recall | | domain | recall |
|---|---|---|---|---|
| Internal Capsule / CPu | 1.00 | | Layer 5 | 0.71 |
| Pia Layer 1 | 0.97 | | White matter | 0.67 |
| Hippocampus | 0.96 | | **Layer 2-3 lateral** | **0.00** |
| Ventricle | 0.93 | | **Layer 3-4** | **0.00** |
| Layer 4 | 0.89 | | | |
| Layer 6 | 0.85 | | | |
| Layer 2-3 medial | 0.79 | | | |

Two domains are missed at the default α_s = 0.5, for different reasons, and
the spatial weight separates them (`resolution.py`, `elongation.py`):

- **Layer 3-4 is at the resolution limit.** It is 73.1 µm wide against a
  65 µm smoothing diameter: W − 2h = 8 µm; every recovered domain is at least
  64 µm wide. Its fibers are contaminated, so its expression margin is small:
  at the default it is lost in every initialization, and at α_s ≤ 0.25 it is
  recovered (0.95) in only 1–2 of 10 initializations, Layer 4 taking the
  cluster otherwise.
- **Layer 2-3 lateral is partly a labelling distinction.** A supervised
  classifier separates it from Layer 2-3 medial at only 0.707 balanced
  accuracy [0.64, 0.78]. It is recovered (recall 0.61) only at α_s = 0.75 — but
  that setting collapses Pia Layer 1, whose aspect ratio is 10.3, from 0.97 to
  0.00. Elongation correlates with recall at ρ = −0.27, −0.27, −0.13 for
  α_s = 0, 0.25, 0.5 and −0.61 at 0.75.

α_s sweep: ARI 0.580, 0.590, 0.620, 0.482 and mean per-domain recall 0.692,
0.694, 0.707, 0.608 at α_s = 0, 0.25, 0.5, 0.75 (default initialization).

**The field is not a boundary map here** (`field_at_boundaries.py`): averaged
over interface edges the smoothed disagreement is 0.72× its within-domain mean,
and every cortical-layer interface is below 1; it is highest inside the pia
(3.4), ventricle (2.0) and white matter (1.6) against Layer 5 at 0.7. The
weighting works on osmFISH by attenuating the sparse, heterogeneous domains.

**Partition alternatives tried and rejected** (`shape_variants.py`): see its
docstring — connectivity-constrained Ward helps in-sample on osmFISH and
collapses on MERFISH; over-cluster + merge and diffusion coordinates are worse.

**Seed sensitivity** (`data/results/osmfish/seed_sensitivity.csv`, 20 k-means
initializations at the defaults): ARI 0.597 ± 0.038, median 0.621 — bimodal,
14/20 within 0.003 of the reported run and 5/20 at a second optimum near 0.535.
At α_s = 0.25 the spread is 0.609 ± 0.010. Layer 3-4 is recovered (0.95) in
1–2 of 10 initializations for α_s ≤ 0.25 and never at 0.5: a marginal
cluster, consistent with the resolution-limit reading.

**Transfer, not tuning** (with the fitted registration). The MERFISH settings
carried over unchanged (`make osmfish-rule`: h 250 px, σ_s 600 px, d 4) reach
ARI 0.599 with a rule that never sees a label. Attribution from the default
corner (`make osmfish-attribute`, eight-corner grid): bandwidth 250 −0.053,
smoothing 600 −0.035, d = 4 −0.031. Leave-one-spatial-block-out
cross-validation (`make osmfish-cv`, 27 configurations × 5 folds) gives
0.699 ± 0.067 per-block ARI for the selected configuration against 0.670 for
the carried-over defaults, and the same configuration wins every fold
(σ_s 600, α_b 0.25, α_s 0.3; whole-section ARI 0.676 / NMI 0.728). The
paper keeps the defaults for every reported number and quotes the CV value as
the bound on what in-sample selection would add.

Feeds Fig. 3 and Extended Data Fig. 2.
