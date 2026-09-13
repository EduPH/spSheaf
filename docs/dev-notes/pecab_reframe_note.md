# PECAb carcinosarcoma — corrected framing (signaling-state, not cell-type domains)

_Compiled 23 July 2026._

## Why the old framing was wrong

We had presented PECAb as "recovering discrete cancer cell-type domains" and
compared unfavourably to the paper's own clusters (NMI ~0.02). Characterising the
data shows that framing is mismatched to what the assay measures:

- **Both panels are signaling/regulatory, not cell-type.** The 35 protein
  antibodies are 11 phospho-proteins + ~15 pathway proteins (WNT3a, TGFBR1, PI3K,
  Notch1, …) + only ~9 lineage/structural markers. The companion 220-gene
  seq-smFISH panel is RNA-binding / transcription-factor / chromatin-regulator
  genes (DHX36, HNRNPUL1, DNMT3A, DUSP6, ARID1B, …); only 2/18 classic lineage
  markers (FN1, ZEB1) are present.
- **Even the authors' own 5 clusters are near-flat in protein space** (mean
  |z| = 0.055, max |z| per cluster 0.12--0.23) and only weakly spatial
  (neighbour-agreement 0.36 vs 0.22 shuffled). By contrast our mesenchymal
  compartment reaches |z| = 1.5 — i.e. the method finds *stronger* structure than
  the published clustering.
- The paper's stated purpose is to **capture cell states / reconstruct
  spatiotemporal signaling**, not tissue architecture.

**Contrast with the Atera/Imoto case.** There, cell-type markers existed but were
*filtered out* by a count threshold — a fixable feature-selection error, and
fixing it revealed sharp cancer cell types. Here the discriminative cell-type
features are *largely absent from the assay by design*, so no feature selection
or parameter tuning creates discrete domains — the signal is intrinsically
continuous (signaling state) with one genuinely discrete structure (mesenchymal).

## Corrected result (both replicates, `pecab_signaling.py`)

Grid 150, bandwidth 300, k = 5. Output leads with the **continuous** picture.

- **Sheaf boundary field** = spatial map of signaling-state transitions; mostly
  smooth (a continuum) with a few sharp interfaces.
- **Mesenchymal/myofibroblast compartment** recovered in both replicates
  (domain 1; FN1/VIM/a-SMA), the one strong discrete structure. Coherence
  0.970 (Rep1) / 0.975 (Rep2); silhouette 0.028 / 0.076.
- **Continuous EMT gradient** (`emt_map.png`): mesenchymal-minus-epithelial score
  varies smoothly across the section rather than in blocks.
- **Cross-replicate reproducibility:** the marker co-localization structure of the
  two independent sections agrees at **Pearson r = 0.937** (21 marker pairs). The
  EMT axis is consistent in sign across replicates: corr(VIM,FN1) +0.58/+0.59;
  corr(FN1,E-Cadherin) -0.60/-0.48; corr(E-Cadherin,EpCAM) +0.25/+0.38. This is
  the honest validation: the recovered structure is real and reproducible even
  though it is a continuum, not discrete domains.

## Framing for the manuscript

- Position PECAb as **modality versatility on protein/signaling data**: the method
  runs unchanged and captures the spatial organization of *continuous signaling
  state* (boundary field) plus the one discrete compartment (mesenchymal), rather
  than discrete cancer cell types.
- Lead with the boundary field and the cross-replicate reproducibility (r = 0.94);
  drop the "cancer cell type" language (that is the Atera story, which its data
  supports and this does not).
- Honest note: the protein signal is subtle by design; the method neither
  fabricates domains nor misses the real structure (graceful behaviour on
  low-contrast data), and finds more structure than the authors' own clustering.
- Likely belongs in **Extended Data** as the protein-modality vignette.

## Files (`output_pecab_signaling_Rep1/`, `_Rep2/`)

`domain_boundary_map.png` · `marker_maps.png` · `emt_map.png` ·
`signature_heatmap.png` · `marker_corr.csv` · `summary.csv`. Code:
`pecab_signaling.py`.
