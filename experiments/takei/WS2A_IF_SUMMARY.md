# WS2a — chromatin-mark nuclear zones (Takei sequential IF) reference summary

The Phase-2 completion of the Takei subnuclear application: nuclear zones
defined from **per-voxel immunofluorescence of ~21 chromatin marks / nuclear
bodies** — the signal the paper itself uses to define nuclear compartments —
rather than the Phase-1 chromosome-territory proxy. Demonstrates the method at
**subnuclear scale, in 3D, on a protein/epigenetic-mark modality**, and
certifies the recovered zones against the known radial architecture of the
nucleus. No manual per-voxel ground truth exists; certification is biological
concordance + cross-nucleus reproducibility.

_Compiled 22 July 2026. Author: Eduardo Paluzo-Hidalgo._

---

## 1. Platform and data

**Source.** Takei et al. 2021, *Integrated spatial genomics* (Zenodo
10.5281/zenodo.3735329), `IF-binned1x1x1-E14-replicate1-Pos0.zip` (3.3 GB,
md5 `113ef6da421bccee05b13036f13e9c13`; the archive's inner folder is labelled
`replicate2` — a Zenodo naming quirk, checksum-verified).

**Content.** 46 single mouse ES-cell (E14) nuclei imaged by **sequential
immunofluorescence**. Per cell and per imaging channel, a long table of
`hybID, avgint, xIdx, yIdx, zIdx` — the mean IF intensity in each 1×1×1 voxel at
each hybridization round. Each `(Channel, hybID)` pair is a **chromatin mark**,
mapped via `IF_supp_table_hyb_targets.xlsx` (sheet *E14 replicate 2*).

**21 marks used** (Ch1/Ch2; Ch4 = DAPI): constitutive/repressive —
`H3K9me2, H3K9me3, H3K27me2, H3K27me3, H4K20me2, H4K20me3, mH2A1, MinSat,
LINE1`; active/euchromatin — `H3K9ac, H3K27ac, H4K16ac, H4K20me1, RNAPII Ser5-P`;
nuclear bodies / structure — `Fibrillarin` (nucleolus), `SF3a66` (speckles),
`Lamin B1` (lamina), `Telomere`, `SINEB1`, `Geminin`, `H3pSer10`.

Nuclei carry **150k–700k voxels** each (median ≈ 340k).

---

## 2. Method as applied

Per nucleus:

1. **Per-voxel mark fiber.** Read the Ch1/Ch2 CSVs, keep the labelled
   `(Channel, hyb)` rows, and pivot `(xIdx, yIdx, zIdx)` voxels × marks into a
   dense per-voxel intensity vector (voxels imaged in every mark are kept).
   Each voxel's vector is **normalized by its own mean** (`--normalize-voxel`)
   so zones reflect mark **composition**, not staining-depth/brightness
   gradients, then `log1p`.
2. **3D feature fibers.** Voxel coords → nm (xy × 103, z × 250); anisotropic
   Gaussian-kernel smoothing onto a 3D grid (`fibers.build_feature_fibers`,
   grid 550 nm xy / 450 nm z, bandwidth 800 nm xy / 700 nm z).
3. **Sheaf pipeline** (unchanged): k-NN graph → local-PCA stalks →
   principal-angle restriction maps → boundary field → boundary-aware k-means
   (`--n-zones 5`) → spatial cleanup.
4. **Zone identity.** Per zone, mean z-scored mark profile → top marks +
   heterochromatin/active balance.

**Run:** `run_takei_if.py --zip ... --excel ... --max-cells 46 --n-zones 5
--normalize-voxel --output-dir output_takei_if`. **Certification:**
`certify_if.py`.

---

## 3. Results

**46 nuclei, 5 zones each**, spatially contiguous 3D territories. Recurring zone
identities (representative nucleus): **lamina-associated constitutive
heterochromatin** (Lamin B1 + MinSat + mH2A1), **repressive/facultative**
(H3K27me2 + H4K20me3 + H3K27me3), **nucleolar/speckle** (Fibrillarin + SF3a66),
and **active interior** (H3K27ac + H3K9ac + RNAPII Ser5-P).

**Certification — the zones reproduce the nucleus's known radial architecture:**

- A zone's mean **radial position** (0 = centre, 1 = surface) vs its
  **peripheral-mark score** (Lamin B1, MinSat, mH2A1, H3K9me2/3, H4K20me3):
  per-nucleus Pearson **r = +0.72 ± 0.16, positive in 46/46 nuclei**; pooled over
  all 230 zones **r = +0.70**. Heterochromatin/lamina marks are peripheral,
  robustly and unanimously.
- The **peripheral-most zone is a lamina/heterochromatin zone in 45/46 nuclei
  (98%)**.
- Interior (active/nucleolar) marks anti-correlate more weakly with radius
  (mean r = −0.16, negative in 31/46) — expected, since the nucleolus can sit
  centrally or perinucleolarly and active chromatin is not strictly central;
  the strong, clean signal is the peripheral heterochromatin compartment.

So the method recovers the canonical peripheral-heterochromatin / interior-
euchromatin organization of the nucleus **unsupervised, in 3D, reproducibly
across 46 nuclei**, directly from raw chromatin-mark IF voxels.

---

## 4. Framing for the manuscript

- This **completes the subnuclear application** begun with DNA seqFISH+
  chromosome territories: the same sheaf pipeline, now on the paper's own
  **chromatin-mark IF** signal, carves nuclear zones whose identities are
  interpretable epigenetic compartments (lamina heterochromatin, polycomb-
  repressed, nucleolar/speckle, active).
- Report as **qualitative + reproducibility-certified** (no per-voxel GT):
  the recovered zones reproduce the known radial nuclear architecture
  (peripheral-mark↔radius r ≈ 0.7, 46/46 nuclei; peripheral-most zone is
  lamina/heterochromatin in 98%).
- Emphasize modality + scale span: the **same method** runs from mm tissue
  (MERFISH/Atera) to µm nuclei, on RNA, DNA, and now **protein/epigenetic
  marks**.
- Honest limitations: no manual per-voxel ground truth; interior compartments
  less sharply radial than the periphery; per-voxel mean normalization chosen to
  suppress staining-depth gradients; single position (Pos0), one replicate.

## 5. Output files (`output_takei_if/`)

Per nucleus: `nucleus_<c>_zones.csv` (x, y, z, zone) ·
`nucleus_<c>_zone_marks.csv` (per-zone mark z-scores + hetero/active balance) ·
`nucleus_<c>_slices.png` (optical sections + orthogonal slabs). Aggregate:
`certification_summary.csv`, `certification_zone_table.csv`,
`certification_if.png`, `summary.csv`.
