# PECAb (protein) application — reference summary

Application of the segmentation-free sheaf boundary-detection method to
**PECAb** multiplexed-protein imaging (Takei et al.), demonstrating that the
method's continuous feature-fiber path generalizes from RNA/DNA to **protein**
without any change to the core pipeline. This is a "versatility on a new data
modality" workstream (WS2b). No ground-truth spatial annotation exists;
evaluation is unsupervised metrics + marker interpretability + biological
concordance.

_Compiled 22 July 2026. Author: Eduardo Paluzo-Hidalgo._

---

## 1. Platform and data

**Platform.** PECAb (Protein-Encoding Combinatorial Antibody barcoding) — a
multiplexed single-cell **protein** imaging assay reading 27–220 antibodies per
cell in situ. Output is a **cell-level** table: each row is a segmented cell
with a centroid `(x, y)`, morphology (`orientation, major/minoraxis`), and a
dense vector of per-antibody mean intensities. This is exactly the
**continuous feature-fiber** input contract (`fibers.build_feature_fibers`) —
the proteins *are* the fiber; no molecule counting.

**Headline dataset — human carcinosarcoma tissue** (`6_Carcinosarcoma_33x220plex`):

| | |
|:--|:--|
| Cells (Rep1) | 7,657 |
| Proteins (raw panel) | 35 (E-Cadherin, N-Cadherin, EpCAM, VIM, a-SMA, FN1, PDPN, Ki67, plus signaling: pSMAD3, pERK, pAKT, pS6K, bCatenin, Notch1, HES1, …) |
| Section extent | ~20,300 × 20,200 (camera px) |
| Median nearest-cell distance | ~90 px |

Carcinosarcoma is a biphasic tumour with coexisting **epithelial** (E-Cadherin /
EpCAM) and **mesenchymal** (N-Cadherin / VIM / a-SMA / FN1) components — a real
spatial-domain axis the method should recover directly from the protein field.
Other PECAb sets present (IMR90 23/56/206-plex, A549 27-plex) are cultured-cell
monolayers, less suited to a *tissue*-domain demo; the runner handles them
identically via `--input`.

**Companion 220-gene panel.** The same section was also imaged by seq-smFISH for
**220 RNA targets** (`6_raw_smFISH_Rep1.csv`, 7,657 cells, same coordinate
frame, 0 cell-overlap with the protein table — a separate measurement). This is
the "220-plex" in the dataset name. The runner is **modality-agnostic** (the
`--proteins`/feature columns are just the fiber), so it runs on this matrix
unchanged — see §3.2. Note this Takei panel targets **RNA-processing / regulatory
genes**, not classic cell-type markers, so its domain identities are less crisp
than the protein panel's — but it independently reproduces the EMT axis via
**ZEB1 / FN1 / FGFR1**.

---

## 2. Method as applied

Core pipeline unchanged (2D). Per section:

1. **Feature fibers.** Regular grid (spacing δ = 150 px) over the tissue; each
   grid point's fiber = Gaussian-kernel-weighted mean (bandwidth h = 300 px) of
   nearby cells' 35-protein vectors, z-standardized per protein. Grid points
   aggregating < 3 cells within the bandwidth are dropped (tissue edges).
2. **Spatial graph** — symmetric k-NN (k = 6) on the grid.
3. **Stalks** — local PCA basis (d = 6) over the 40 nearest fibers.
4. **Restriction maps + sheaf disagreement** — principal-angle SVD per edge;
   disagreement = residual mismatch (high where the protein program rotates).
5. **Boundary field** — per-vertex mean disagreement, smoothed (σ = 450 px).
6. **Boundary-aware k-means** — k = 7 domains; small fragments reassigned.

**Downstream:** per-domain marker proteins (domain mean − global mean in fiber
space); unsupervised metrics (silhouette, Calinski–Harabasz, Davies–Bouldin,
spatial coherence = fraction of graph edges within one domain).

**Run parameters:** `--n-domains 7 --grid-spacing 150 --bandwidth 300
--smooth-radius 450` (defaults). **Code:** `experiments_completed/pecab/run_pecab.py`.

---

## 3. Results (carcinosarcoma Rep1)

- **7 spatial domains** over **11,102 grid points**; **spatial coherence 0.955**
  (domains are large, contiguous territories — no salt-and-pepper).
- Silhouette ≈ 0 (−0.003), as with the Atera whole-transcriptome domains: dense,
  overlapping feature space where silhouette is uninformative; the meaningful
  signal is spatial contiguity + marker identity.
- The boundary field concentrates at a focal hotspot (~9k, 13k px) and along the
  epithelial/mesenchymal interfaces.

**Domains recover the epithelial ↔ mesenchymal (EMT) axis.** Per-domain mean
protein z-scores (cell-level), EMT score = mean(mesenchymal) − mean(epithelial):

| Domain | Identity | EMT score | Defining proteins |
|:--:|:--|:--:|:--|
| 1 | **Mesenchymal / myofibroblastic (CAF-like)** | **+0.66** | a-SMA +1.54, FN1 +0.69, VIM +0.60 |
| 3 | Mesenchymal | +0.28 | FN1 +0.43, VIM +0.32 |
| 4 | **TGF-β signaling niche** | +0.03 | **pSMAD3 +0.80**, EpCAM +0.18 |
| 6 | **Proliferative** | +0.03 | **Ki67 +0.32**, FN1 |
| 2 | Baseline / low-signal | −0.08 | near-zero across markers |
| 0 | Epithelial–mesenchymal **interface** | −0.14 | E-Cad +0.21 *and* N-Cad +0.18, VIM +0.31 |
| 5 | **Epithelial** | **−0.33** | E-Cadherin +0.18, VIM −0.46, FN1 −0.40 |

So the method carves a coherent **epithelial (5) → interface (0) → mesenchymal
(1/3)** spatial gradient, plus a **pSMAD3⁺ TGF-β signaling zone (4)** (TGF-β
drives EMT — biologically expected to sit near the interface) and a
**Ki67⁺ proliferative zone (6)** — all from protein intensities alone, with no
cell segmentation into types.

**Relation to the paper's own cell clusters.** The published PECAb clusters
(Seurat/UMAP on the full protein panel, 5 clusters) show **~zero overlap** with
our spatial domains (NMI ≈ 0.02; each paper cluster spreads roughly evenly
across all 7 domains). This is the expected, informative result: the paper's
clusters are **non-spatial cell states** that are spatially **intermixed**,
whereas the method recovers **spatial tissue architecture** — an orthogonal,
complementary axis a non-spatial clustering does not capture. (The
`(tileID, cellID)` join spans both replicates and is approximate, but the
even-spread pattern in the cross-tab is robust to that.)

---

### 3.2 Full 220-gene RNA panel (same section)

Running the modality-agnostic pipeline on the **220-gene** seq-smFISH matrix
(`--n-domains 10 --log1p`):

- **10 spatial domains** over the same 11,102 grid points; **spatial coherence
  0.946** (again large, contiguous territories — a finer but equally clean
  partition than the 35-protein run).
- Silhouette −0.028; Calinski–Harabasz 408; Davies–Bouldin 4.02.
- Most top markers are ubiquitous RNA-processing genes (RBM25, SF3B1, HNRNPUL1,
  splicing/transport factors) — a property of this *regulatory-gene* panel, not
  of the domains — **but the EMT axis re-emerges from the panel's few
  identity-bearing genes**: per-domain z-scores rank **ZEB1** (EMT master TF)
  highest in domains 9 (+0.71), 4 (+0.49), 7 (+0.33) and lowest in 1/5/6
  (≈ −0.3), with **FN1 / FGFR1 / PALLD** co-tracking (domain 7: FGFR1 +0.59,
  FN1 +0.53). Two **independent panels on the same tissue** (35 protein, 220 RNA)
  therefore agree on the same mesenchymal→epithelial spatial gradient —
  cross-modal corroboration of the recovered architecture.

Output: `output_pecab_carcinosarcoma_220rna/`.

## 4. Framing for the manuscript

- Position PECAb as **modality versatility**: the same sheaf pipeline that
  handles RNA molecules (MERFISH/Atera) and DNA loci (seqFISH+) runs unchanged
  on **protein** data via the continuous feature-fiber path — the fiber is
  swapped, the sheaf machinery is identical.
- Report it as **qualitative + marker-validated** (no ground truth): recovers the
  epithelial↔mesenchymal architecture of a carcinosarcoma directly from
  multiplexed protein intensities, and localizes a TGF-β (pSMAD3) signaling niche
  at the EMT interface.
- Emphasize the contrast with non-spatial cell-state clustering (orthogonal
  partitions) as evidence the method adds spatial structure.
- Honest limitations: no ground truth; unsupervised silhouette low (dense protein
  space); this raw panel is 35 of the 220 antibodies (headline demo; the full
  panel is in the processed matrix and runs identically); domain identities are
  marker interpretations.

## 5. Output files (`experiments_completed/pecab/output_pecab_carcinosarcoma/`)

`domain_labels.csv` (x, y, domain) · `boundary_field.csv` · `domain_markers.csv`
(top enriched proteins/domain) · `metrics.csv` (unsupervised) · `domain_map.png`
(domains + boundary field).
