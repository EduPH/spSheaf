# Atera breast — marker-aware re-analysis (response to colleague)

Addresses the request to recover the **cancer cell-type spatial patterns** that
specific marker genes express, rather than generic top-count domains.

_Compiled 23 July 2026._

## What was wrong before

The original run selected the **top 500 genes by molecule count** for the fiber.
This dropped low-abundance but decisive cell-type markers: **PGR, KRT5, KRT14,
KRT17, TP63, CDKN2A** (the entire basal/squamous program), plus SCGB2A2, PLIN1,
GZMB. The domains therefore could not resolve the cancer cell types, and the
squamous compartment was invisible. A count threshold is exactly the wrong
filter for these genes — they are sparse and noisy in raw data (the motivation
for the colleague's denoising study).

## The fix (two changes; pipeline otherwise unchanged)

1. **Always retain the curated marker panel** (both cancer panels + canonical
   lineage genes), in addition to the top-count genes. Implemented by streaming
   the full 740M-molecule section keeping **100% of marker-gene molecules** and
   1% of the rest (`atera_cache.py`), so sparse markers are fully sampled.
2. **Per-gene z-standardization of the smoothed spatial field.** Each gene's
   kernel-smoothed density is z-scored across grid points before the sheaf
   pipeline, so every gene contributes **equally by its spatial pattern**,
   independent of abundance. Low-count markers (ESR1, PGR, KRT5, TP63, ...) now
   drive the domains instead of being drowned by high-count genes. This is the
   key methodological lever (`atera_markeraware.py`).

Run: grid 50 um, bandwidth 100 um, 290 genes (all 40 curated markers retained),
23,285 grid points, k = 12 domains. Marker counts recovered in the section
(100% sampled): ESR1 170k, PGR 40k, KRT5 33k, TP63 31k, CDKN2A 24k, KRT17 21k,
GATA3 984k, FOXA1 717k, XBP1 1.49M.

## Results

- **12 domains, spatial coherence 0.949.** Domains now map to cancer cell types
  (`domain_celltypes.csv`, `celltype_map.png`):
  - **Luminal tumour** (domains 8, 11) — ESR1/PGR/GATA3/FOXA1/XBP1, the central
    tumour mass.
  - **Basal/squamous** (domain 5, score +1.9) — KRT5/KRT14/KRT17/TP63/CDKN2A;
    a focal region that was entirely absent from the original analysis.
  - **Stroma/CAF** (domain 3, +1.4) — COL1A1/COL1A2/COL3A1/DCN/POSTN.
  - **Immune** (domain 4, +2.2) — PTPRC/CD68/CD3/MS4A1.
  - **Vascular** (domain 10) — PECAM1/VWF. **Proliferation** (domains 0, 9) —
    MKI67/TOP2A/CDK1.
  - Domains 1, 2, 7 are **flagged "Unassigned (low signal)"** (negative on every
    signature) rather than force-labelled, correcting the earlier tendency to
    default low-expression regions to "Immune".
- **Marker spatial-pattern maps** (`marker_maps.png`) reproduce the colleague's
  figure: the luminal markers (ESR1, PGR, GATA3, FOXA1) co-localize in the
  tumour nests, while COL1A2 (stroma), PLIN1 (adipocyte) and KRT5 (basal) show
  distinct complementary patterns — the "specific spatial patterns" requested.

## Honest caveats

- The per-domain cell-type label is a signature argmax and is **guidance**; the
  evidence is the marker maps and `domain_signature_scores.csv`. High-confidence
  calls (basal/squamous, stroma, immune, vascular, proliferation) have clear
  score margins; luminal domains are moderate; low-signal domains are flagged.
- Markers were sampled at 100% but non-markers at 1%; per-gene z-standardization
  removes the resulting magnitude difference, and markers simply carry less
  Poisson noise (a benefit). A full-density confirmatory run is possible.

## Cervical section (`output_atera_cervical_markeraware/`)

Same procedure with a squamous-tumour signature (KRT5/14/17/KRT6A/KRT15/TP63/
SOX2/CDKN2A/SERPINB3/4). Cached 23.9M molecules from the 1.24B-molecule section.
**12 domains, spatial coherence 0.942.** The panel-B marker maps
(`marker_maps.png`) show the squamous program (CDKN2A/KRT5/KRT17/TP63)
co-localizing in the tumour nests, with the immune markers (CD68/PTPRC/GZMB) and
stroma (DCN) forming distinct compartments. Domains resolve into a **squamous
tumour** compartment, a large proliferative tumour bulk (expected — squamous
carcinoma is highly proliferative, so keratin and proliferation programs
overlap), stroma/CAF, immune, and low-signal background.

## Full-density confirmatory (breast, `output_atera_breast_fulldensity/`)

To check that 1% subsampling did not distort the result, the breast section was
re-rasterised at **full density (all 740M molecules, no subsampling of any
gene)** on the same gene set and grid (`atera_fulldensity.py`), then re-clustered.

- **12 domains, spatial coherence 0.953** (vs 0.949 at 1%).
- Partition agreement with the 1% run: **ARI 0.848, NMI 0.853**; **84.2% of grid
  points receive the same cancer-cell-type label**.
- Every tumour-relevant cell type is recovered in both runs — importantly the
  **basal/squamous domain persists with a strong score (+1.94)**, confirming it
  is real and not a subsampling artifact. The only difference is a minor,
  diffuse immune compartment that the full-density run folds into stroma.

**Conclusion:** the marker-aware result is robust to subsampling; keeping markers
at 100% and z-standardising per gene is what matters, not the sampling rate of
the bulk genes.

## Files

Breast 1%: `output_atera_breast_markeraware/`; breast full-density:
`output_atera_breast_fulldensity/`; cervical: `output_atera_cervical_markeraware/`.
Each has `celltype_map.png`, `marker_maps.png`, `signature_heatmap.png`,
`domain_celltypes.csv`, `domain_signature_scores.csv`, `domain_labels.csv`.
Code: `atera_cache.py`, `atera_markeraware.py`, `atera_fulldensity.py`.

## Next

Update the manuscript Atera section with the cancer-cell-type result (both
sections) and the subsampling-robustness confirmation.
