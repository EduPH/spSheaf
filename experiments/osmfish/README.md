# osmFISH — mouse somatosensory cortex

Second annotated benchmark, and the one that shows the method's resolution
limit. 33 genes, 4,839 cells, 11 annotated cortical regions, 44,410 grid
points.

## Run

```bash
make osmfish            # metrics
make panels-osmfish     # same run, also dumps the data Figure 3 is drawn from

make osmfish-diagnose | osmfish-rule | osmfish-cv | osmfish-attribute
make osmfish-resolution | osmfish-elongation
```

Needs `data/raw/osmfish/mRNA_coords_raw_counting.hdf5` and
`osmfish_20260512084934.h5ad`. Settings: δ = 150 px (9.8 µm), h = 500 px
(32.5 µm), σ_s = 300 px, d = 9, k = 11.

Two corrections are load-bearing and easy to get wrong: the annotation lives in
the loom coordinate frame, which is **axis-flipped** relative to the imaging
frame (recovered without labels — mean per-cell composition correlation 0.83
flipped against 0.07 unflipped), and the scale is 15.3846 px/µm.

## Results

**ARI 0.538, NMI 0.639** at cell level; 3,228 of 4,839 cells correct.

Naive baselines under the same protocol: coordinates only 0.223 ± 0.006,
expression only 0.185 ± 0.027, the two concatenated 0.222 ± 0.016.

SSAM reaches ARI 0.566 / NMI 0.721 here and so beats spSheaf on this dataset.
osmFISH is one of SSAM's own primary published analyses and its released
settings were tuned on it; the comparison is reported as it stands.

**Per-domain recall** — the interesting part:

| domain | recall | | domain | recall |
|---|---|---|---|---|
| Pia Layer 1 | 0.89 | | Internal Capsule / CPu | 0.64 |
| Layer 6 | 0.83 | | White matter | 0.48 |
| Hippocampus | 0.82 | | Ventricle | 0.45 |
| Layer 2-3 medial | 0.79 | | **Layer 2-3 lateral** | **0.00** |
| Layer 4 | 0.78 | | **Layer 3-4** | **0.00** |
| Layer 5 | 0.75 | | | |

Two domains are missed entirely, and `resolution.py` and `elongation.py` show
they are missed for two different reasons.

- **Layer 3-4 is below the resolution limit.** It is 73.1 µm wide against a
  65 µm smoothing diameter: W − 2h = 8 µm. Every domain that is recovered is
  at least 64 µm wide. This is a stated property of a kernel method, not a
  tuning failure.
- **Layer 2-3 lateral is partly a labelling distinction.** A supervised
  classifier separates it from Layer 2-3 medial at only 0.707 balanced
  accuracy [0.64, 0.78]. It is recovered (recall 0.73) at α_s = 0.75 — but
  that setting collapses Pia Layer 1, whose aspect ratio is 10.3, from 0.89 to
  0.00. Elongation correlates with recall at ρ = +0.36, +0.35, +0.34 for
  α_s = 0, 0.25, 0.5 and inverts to −0.21 at 0.75.

**Transfer, not tuning.** The MERFISH settings carried over unchanged reach
ARI 0.428 with a rule that never sees a label. Leave-one-spatial-block-out
cross-validation gives 0.580 ± 0.081 for a selected configuration against
0.590 for the carried-over defaults, and picks a different winner every fold —
so selection here is noise. Attribution of the transferred settings:
bandwidth −0.082 ARI, stalk dimension −0.017, smoothing +0.015.

Feeds Fig. 3 and Extended Data Fig. 3.
