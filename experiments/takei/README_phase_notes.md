# Takei nuclear zones — 3D sheaf boundary detection inside single nuclei

Applying the segmentation-free sheaf pipeline at **subnuclear scale** to recover
**nuclear zones** (subnuclear compartments) from DNA seqFISH+ data. This is the
intended demonstration that the method generalizes across modality
(transcriptome → genome), scale (mm tissue → µm nucleus), and dimension
(2D → 3D). The N-D code path in `spsheaf` (2D/3D grid, kernel, graph,
smoothing) is already in place; this experiment drives it per nucleus.

## Dataset (Phase-1 target: Takei 2021 mESC prototype)

Takei et al., *Nature* 2021, "Integrated spatial genomics reveals global
architecture of single nuclei" (doi:10.1038/s41586-020-03126-2).

- Zenodo record **3735329** (CC-BY-4.0, 46 GB total).
- **`DNAseqFISH+.zip` — 144 MB** — 3D pixel locations of ~3,600 genomic loci in
  single cells. Voxel size **103 nm (x, y)**, **250 nm (z)** — natively
  anisotropic, handled by `z_spacing` / `z_bandwidth`. **This is the Phase-1
  file.**
- `IF-binned1x1x1-*.zip` — 2–5 GB each — per-voxel intensities for ~17 chromatin
  marks / subnuclear structures. The dense "clean fiber" for **Phase 2**.
- `RNAseqFISH.zip` (1 MB); `IF_supp_table_hyb_targets.xlsx` (mark → hyb/channel).

Download (Phase 1 only):

```bash
cd experiments_pending/takei_nuclear_zones
curl -L -o DNAseqFISH+.zip \
  "https://zenodo.org/records/3735329/files/DNAseqFISH+.zip?download=1"
unzip DNAseqFISH+.zip -d dna_seqfish
```

## How the pipeline maps onto a nucleus

| Tissue pipeline | Nuclear-zone version |
|:----------------|:---------------------|
| molecules `(x, y, gene)` | locus dots `(x, y, z, locus_id)` per cell |
| one tissue-wide grid (δ≈15–50 µm) | one 3D grid **per nucleus** (δ≈100–150 nm) |
| fiber = local gene composition | fiber = local **locus composition** (Phase 1) or **chromatin-mark intensities** (Phase 2) |
| stalk = local PCA of fibers | unchanged |
| disagreement = expression-program rotation | disagreement = **chromatin-state rotation** |
| domains = tissue regions | domains = **nuclear zones** |
| ground truth = manual annotation | **published zones** (themselves clustering ⇒ *concordance*, not accuracy) |

Everything except grid/kernel scale and the fiber definition is unchanged, and
those spatial parts are already N-D.

## Phased plan

1. **Inspect** the `DNAseqFISH+` files — confirm columns (cell id, x/y/z, locus
   id) with `run_takei_nucleus.py --inspect`. *(schema TBD until the file is in hand)*
2. **Phase 1 — locus-composition fiber.** Per nucleus: pixels → nm, treat
   `locus_id` as "gene", build a 3D fiber field, run the pipeline → zones.
   Report unsupervised metrics + 3D plots. Proves the modality/scale/dimension
   generalization on a 144 MB download.
3. **Phase 2 — chromatin-mark IF fiber.** Swap the fiber for the dense ~17-mark
   IF field (or concatenate with locus features). Cleaner zones.
4. **Scoring.** If per-cell published zone labels are available, compute
   ARI/NMI as *concordance with published zones*; else qualitative.
5. **(Later) 2025 headline** — in-tissue cerebellum, 100k loci; needs
   Stereo-seq-style scaling.

## Suggested starting parameters (subnuclear, nm units)

| Symbol | Start | Note |
|:-------|:------|:-----|
| δ (x,y grid) | 120 nm | ~1 grid pt per ~1.2 voxels |
| h (x,y bandwidth) | 250 nm | ~2.5× δ |
| z grid spacing | 250 nm | native z voxel |
| z bandwidth | 400 nm | blends adjacent z |
| k (graph) | 6 | as tissue |
| k_nbrs (stalk PCA) | 30–50 | fewer points per nucleus |
| d (stalk dim) | 6 | |
| n_zones | 4–8 | per nucleus; free parameter |
| min loci / cell | ~500 | drop poorly-detected nuclei |

## Performance / dimensionality note (validated)

The per-nucleus 3D path is wired and validated on a synthetic nucleus
(`run_takei_nucleus.py` → grid, graph, stalks, boundary, zones in <1 s).
**Key constraint:** the locus-composition fiber has one axis per locus
(~3,660), and the stalk bases scale as `N_grid × 3,660 × d`. Keep the
per-nucleus grid coarse (~2–5k grid points, i.e. `--grid-nm ≥ ~500`) or the
stalks blow up (20k points × 3,660 → ~3.5 GB). Two better long-term options:
bin loci (e.g. to chromosome arms / Mb bins) to shrink the fiber dimension, or
move to the **17-dim IF chromatin-mark fiber** (Phase 2), which is both
lower-dimensional and dense — the loci are near single-copy, so the raw
locus fiber is very sparse and the recovered zones will be coarse.

## Confirmed schema (1Mb file, replicate 1)

Columns: `fov, cellID, channel, geneID, x, y, z, seeds, intensity`.

- A **nucleus is `(fov, cellID)`** — cellID repeats across FOVs. 201 nuclei,
  median **3,486 dots/nucleus** (min 702, max 6,359).
- `geneID` mixes **chromosome loci** (`chr1-#1`, ~1,200) with **named-gene loci**
  (`0610010F05Rik`, ~1,200). `--chrom-level` maps the former to chr1..chrX and
  buckets the latter into `other` → **21 dense fiber dimensions** (fast). The
  full per-locus fiber is ~2,460-dim (sparse, slow — coarse grid only).
- Units: **x, y in pixels (×103 nm)**, **z in slices (×250 nm)**. A nucleus is
  ~19 × 14 × 3.5 µm (flat/oblate). Anisotropy handled by `z_spacing`/`z_bandwidth`.
- Validated: chrom-level, `--grid-nm 550`, 5,727 grid points, 5 zones, ~1.8 s;
  zones are genuine 3D compartments (not columns).

## Certification (how we validate zones)

The paper defines nuclear zones by **combinatorial chromatin marks** (IF), not
by DNA loci. So:

- The 25kb file's `labelID` is **not** a zone reference — it is ~binary (0/1
  dominate), radially flat, and our chromosome-territory zones score ARI≈0
  against it (expected). Do not treat it as ground truth.
- **Self-certification by identity.** Each inferred zone is characterized by its
  dominant feature. With the chromosome-intensity fiber (Phase 2a), every zone
  is dominated by a distinct chromosome ⇒ the zones are **chromosome
  territories**, each labeled by its chromosome (see `run_takei_features.py`
  output). With the chromatin-mark IF fiber (Phase 2b), each zone's dominant
  marks give a compartment identity (LaminB1→lamina, nucleophosmin→nucleolus,
  SF3→speckle, H3K27me3→Polycomb, H3K9me3→heterochromatin, H3K4me3/RNAPII→active).
- **Reproducibility across replicates (certified).** On the full dataset
  (rep1 = 197 nuclei, rep2 = 200): zone↔chromosome NMI = 0.569 / 0.571
  (identical), and per-chromosome radial position reproduces at
  **Pearson r = 0.91 (p = 2.7e-8)**, Spearman ρ = 0.85. chr19 is most interior
  (matches its known gene-rich central positioning), chr1/chr2 most peripheral.
  Note: at only 22 nuclei this correlation was 0.37 (n.s.) — the signal is real
  but needs the full sample. See `certification_cross_replicate.png`.
- **External concordance** (optional): Hi-C A/B compartments per locus.

## Phase 2 status

- **2a (done, no download):** dense chromosome-intensity fiber via
  `fibers.build_feature_fibers` + `run_takei_features.py --features chrom`.
  Recovers chromosome territories with per-zone chromosome identity; runs ~2 s/nucleus.
- **2b (in progress — IF chromatin-mark fiber):** the paper's actual zone signal,
  certifiable by marker identity.

  **Current blockers (2 files):**
  1. `IF-binned1x1x1-E14-replicate1-Pos0.zip` — **download incomplete** (~416 MB
     of the 3.3 GB Zenodo file; the zip won't open until finished).
  2. `IF_supp_table_hyb_targets.xlsx` (17 kB) — **not yet downloaded**; it maps
     each IF target (mark / subnuclear structure) to its (hyb round, fluorescent
     channel).

  **Ready now:** `inspect_if.py` reports the IF structure the moment the download
  completes (`python inspect_if.py --zip <...>Pos0.zip --excel IF_supp_table_hyb_targets.xlsx`).
  It confirms the current zip is incomplete.

  **Plan once the two files are in hand:**
  1. Run `inspect_if.py` to lock the real schema (expected: per-voxel intensity
     keyed by hyb/channel, with a cell assignment).
  2. Write `if_to_wide.py`: join the IF long table with the Excel to name the
     ~17 marks, then pivot to a **wide per-voxel table** `(cellID, x, y, z,
     LaminB1, Nucleophosmin, SF3a66, H3K27me3, H3K9me3, H3K4me3, RNAPII, ...)`.
  3. Feed that straight into the existing feature-fiber runner:
     `run_takei_features.py --features cols:LaminB1,Nucleophosmin,... ` — the IF
     voxels are the points and the marks are the fiber features, so no new
     pipeline code is needed.
  4. **Certification by marker identity:** label each inferred zone by its
     dominant marks (LaminB1→lamina/periphery, nucleophosmin→nucleolus,
     SF3→speckle, H3K27me3→Polycomb, H3K9me3→heterochromatin, H3K4me3/RNAPII→
     active) and compare to the published zones.

## Open items

- Confirm exact `DNAseqFISH+` column schema (needs the file).
- Decide fiber for Phase 1 vs 2 (recommendation: locus first, IF next).
- Locate per-cell published zone labels for quantitative concordance.
