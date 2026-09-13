# Atera application — reference summary (for manuscript writing)

Application of the segmentation-free sheaf boundary-detection method to **10x
Genomics Atera** whole-transcriptome in-situ data, on two human cancer sections
(breast, cervical). This is the "application to the latest data" workstream. No
ground-truth spatial annotation exists, so evaluation is qualitative +
marker-based; compartment labels are left to expert annotators.

_Compiled 18 July 2026. Author: Eduardo Paluzo-Hidalgo._

---

## 1. Platform and data

**Platform.** 10x Genomics **Atera** — an in-situ, imaging-based spatial platform
delivering **whole-transcriptome** (probe-based, ~18,000 genes) expression at
**subcellular resolution** on FFPE and fresh-frozen tissue. Launched April 2026.
Output is molecule-level: each detected RNA is a row with its gene identity,
`(x, y, z)` location in microns, and a quality value — the exact
`(x, y, gene)` input contract the method is built on (same as MERFISH /
Stereo-seq). The Output Bundle uses the 10x Xenium file format.

**Datasets** (10x preview data, FFPE human cancer):

| Section | Transcripts (total) | Section extent | Genes (WTA) |
|:--------|:-------------------:|:--------------:|:-----------:|
| Breast cancer   | 740,442,119   | ~9.9 × 5.9 mm | ~18,000 |
| Cervical cancer | 1,235,087,993 | ~8.9 × 8.9 mm | ~18,000 |

- Source: 10x Genomics dataset pages, *Atera WTA FFPE Human Breast Cancer* and
  *…Human Cervical Cancer* (Output Bundle → `transcripts.parquet`).
- `transcripts.parquet` schema (standard Xenium): `transcript_id, cell_id,
  overlaps_nucleus, feature_name, x_location, y_location, z_location, qv,
  fov_name, nucleus_distance, codeword_index, codeword_category, is_gene`.
- **No manual spatial-domain / cell-type ground truth is distributed** with the
  preview data.

**Preprocessing (per section).** The `transcripts.parquet` is streamed by
row-group; per batch we (i) keep detections with **QV ≥ 20** (10x default),
(ii) keep real genes only via the `is_gene` flag (drops negative-control,
blank, and codeword probes), and (iii) **subsample to 1%** of transcripts —
statistically indistinguishable at the domain grid spacing, where each grid
point still aggregates thousands of molecules, and required because the full
data is extremely dense (~2.7 transcripts/µm²; a k-d tree on all ~740M points
would need ~12 GB). The **top 500 genes by count** are then retained to bound
the local-PCA stalk memory (which scales as N_grid × N_genes × d). This mirrors
the Stereo-seq scalability adaptation (gene filtering + subsampling).

---

## 2. Method as applied to Atera

The core pipeline is unchanged (2D molecule-level). For each section:

1. **Kernel fibers.** A regular grid (spacing **δ = 50 µm**) over the tissue;
   at each grid point a 500-gene expression fiber is built by Gaussian kernel
   smoothing (bandwidth **h = 100 µm**) of nearby molecules, normalized to
   proportions and log-transformed. The radius query is block-processed so the
   neighbor lists never materialize all at once (memory-safe at this density).
2. **Spatial graph.** Symmetric k-NN graph (k = 6) on the grid.
3. **Stalks.** Local PCA basis (dimension d = 6) over the k_nbrs = 50 nearest
   fibers at each vertex.
4. **Restriction maps + sheaf disagreement.** Per edge, principal-angle SVD of
   the stalk overlap gives restriction maps; the disagreement is the residual
   mismatch after alignment — high where the expression program rotates sharply.
5. **Boundary field.** Per-vertex mean disagreement, Gaussian-smoothed
   (σ_s = 150 µm).
6. **Boundary-aware clustering.** Boundary-weighted expression features +
   scaled spatial coordinates, k-means into **k = 12 domains**.
7. **Spatial cleanup.** Small disconnected fragments reassigned.

**Downstream (per domain):**
- **Marker genes** — top genes by enrichment (domain mean − global mean of the
  log-normalized fibers). Data-driven, no interpretation.
- **Marker-signature scoring** — each domain scored against canonical
  compartment signatures (tumour epithelium [tissue-specific: breast=luminal,
  cervical=squamous], stroma/CAF, immune, vascular/endothelial, proliferation)
  by mean per-gene z-score across domains; an auto-suggested compartment +
  confidence margin is produced **as guidance only**.
- **Unsupervised metrics** — silhouette, Calinski–Harabasz, Davies–Bouldin,
  and spatial coherence (fraction of graph edges within one domain).
- **Annotation worksheet** — per-domain size + data-driven markers +
  signature scores + auto-suggestion + **blank columns for expert labels**.

**Run parameters (both sections):** `--subsample 0.01 --top-genes 500
--grid-spacing 50 --bandwidth 100 --smooth-radius 150 --n-clusters 12
--min-molecules 15 --block-size 1000`. QV ≥ 20, control probes removed.

**Code:** `experiments_pending/atera/run_atera.py` (pipeline + markers +
metrics + map), `signature_enrichment.py` (compartment signatures + heatmap +
compartment map), `make_annotation_worksheet.py` (expert worksheet).

---

## 3. Results

### 3.1 Breast cancer

- **12 spatial domains** over **22,471 grid points**, ~9.9 × 5.9 mm.
- **Spatial coherence 0.934** (domains spatially contiguous); silhouette −0.063
  (expected — whole-transcriptome domains overlap in expression space and the
  clustering blends spatial coordinates; not a meaningful discriminator here).
- The boundary field concentrates on tumour-nest edges (tumour–stroma
  interfaces).
- **Marker genes recover the two hallmark tumour compartments:**
  - *Luminal tumour epithelium* (multiple subdomains, e.g. 0, 1, 4, 5, 7, 10):
    CCND1, CDH1, GATA3, MUC1, NHERF1, TFF1, TFF3, XBP1, AGR3, CD24, STARD10 —
    the defining luminal breast-cancer programme. Several distinct epithelial
    subdomains = intratumoral heterogeneity / separate tumour nests (e.g. TFF1/
    TFF3-high vs GATA3/DSP-high).
  - *Cancer-associated fibroblast / desmoplastic stroma* (e.g. 2, 3, 8, 9, 11):
    POSTN, COL3A1, COL6A1/COL6A2, CXCL12, TAGLN, THBS1, CCN2/CTGF, AEBP1 —
    forming the connective bands separating epithelial nests.
  - Domain 6 (CXCL12, LAMB1, PIK3R1) reads as a mixed perivascular/stromal
    interface.

### 3.2 Cervical cancer

- **12 spatial domains** over **29,012 grid points**, ~8.9 × 8.9 mm.
- **Spatial coherence 0.948**; silhouette −0.047.
- Highly proliferative squamous programme; compartments span epithelial
  (squamous markers KRT5/KRT14/KRT17, TP63, SOX2, SERPINB3/4), stroma, immune,
  vascular, and proliferation-high regions.

### 3.3 Interpretation / evaluation stance

- **No ground truth** ⇒ no ARI/NMI. Evaluation is (a) unsupervised metrics and
  (b) biological interpretability of the domains' marker genes.
- The **signature auto-suggestion is guidance only and is imperfect**: on breast,
  a clearly stromal domain (POSTN/COL3A1) was auto-called "Immune", and only
  5/12 (breast) / 4/12 (cervical) domains cleared the confidence margin. This is
  a genuine limitation of simple signature-argmax on whole-transcriptome domains
  that contain mixed tumour–stroma interfaces and share genes across
  compartments — **not** a defect of the domains, whose data-driven markers are
  clean and consistent. Final compartment labels are therefore left to expert
  annotators via the worksheet.

---

## 4. Framing for the manuscript

- Position Atera as the **application to the newest platform**: segmentation-free,
  whole-transcriptome, at scale (>10⁹ molecules per section), recovering
  tumour–stroma architecture and intratumoral heterogeneity **directly from raw
  molecules with no cell segmentation**.
- Report it as **qualitative + marker-validated**, explicitly *not* benchmarked
  (no ground truth). State: "domains were characterized by their data-driven top
  marker genes; compartment annotation (tumour / stroma / immune / …) was
  performed by [collaborators] using canonical marker signatures as guidance."
- Emphasize the **boundary field** localizing tumour–stroma interfaces as the
  method's distinctive output.
- Honest limitations to state: no ground truth; unsupervised silhouette is low
  (dense whole-transcriptome expression); compartment labels are marker-based
  interpretations, not verified regions; subsampling to 1% (justified by grid
  aggregation).

## 5. Output files (per section, in `output_atera_<breast|cervical>/`)

`domain_labels.csv` (x, y, domain) · `boundary_field.csv` · `domain_markers.csv`
(top enriched genes/domain) · `metrics.csv` (unsupervised) · `domain_map.png`
(domains + boundary field) · `domain_signature_scores.csv` ·
`domain_compartments.csv` (auto-suggestion) · `signature_heatmap.png` ·
`compartment_map.png` · `domain_annotation_worksheet.csv` (for experts).
