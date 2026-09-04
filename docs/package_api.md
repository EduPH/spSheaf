# Sheaf-Based Spatial Domain Inference

**Segmentation-free spatial domain inference for spatial transcriptomics via cellular sheaf boundary detection.**

This framework detects tissue domain boundaries by measuring disagreement between local gene expression bases (stalks of a cellular sheaf) across a spatial graph, then uses boundary-aware clustering to infer spatial domains — all without requiring cell segmentation.

## Overview

Traditional spatial domain methods rely on cell segmentation as a preprocessing step. This framework operates directly on raw molecule coordinates, constructing a smooth expression field via kernel smoothing and building a cellular sheaf whose *disagreement* localizes domain boundaries.

**Key idea:** Domain boundaries are where neighboring local PCA bases *rotate* relative to each other. The sheaf framework formalizes this: restriction maps between adjacent stalks encode subspace alignment via principal angles, and high sheaf disagreement flags boundary regions. Interior points retain their full expression signal during clustering, while boundary points are attenuated — producing clean segmentation.

### Results on Moffitt 2018 MERFISH (mouse hypothalamus)

| Slice | N points | ARI   | NMI   |
|:------|:--------:|:-----:|:-----:|
| 0.04  | 14,168   | 0.528 | 0.617 |
| 0.09  | 14,020   | 0.513 | 0.626 |
| 0.14  | 14,280   | 0.442 | 0.554 |
| 0.19  | 14,195   | 0.650 | 0.682 |
| 0.24  | 14,346   | 0.654 | 0.674 |
| **Mean** | **14,202** | **0.557** | **0.631** |

Mean NMI = **0.631** vs. best published benchmark (SCAN-IT, NMI = 0.554; Yuan et al., Nature Methods 2024, 13 methods compared). Results include spatial connectivity cleanup.

#### Best slices

<p align="center">
  <img src="Experiments/MERFISH/output_sheaf_boundary/sheaf_boundary_MERFISH_0_19_h5ad.png" width="90%" />
</p>

<p align="center">
  <img src="Experiments/MERFISH/output_sheaf_boundary/sheaf_boundary_MERFISH_0_24_h5ad.png" width="90%" />
</p>

Each figure shows: ground truth domains (left) vs. sheaf boundary prediction with Hungarian-matched colors (right), raw sheaf disagreement field (bottom left), and spatially smoothed boundary score (bottom right).

#### Summary across all slices

<p align="center">
  <img src="Experiments/MERFISH/output_sheaf_boundary/all_slices_bar_chart.png" width="80%" />
</p>

<p align="center">
  <img src="Experiments/MERFISH/output_sheaf_boundary/all_slices_scatter_comparison.png" width="80%" />
</p>


## Installation

```bash
git clone https://github.com/EduPH/Cellular-Sheaves-for-Spatial-Domain-Inference-in-Spatial-Transcriptomics.git
cd Cellular-Sheaves-for-Spatial-Domain-Inference-in-Spatial-Transcriptomics
pip install -e .
```

### Requirements

- Python ≥ 3.9
- NumPy, Pandas, SciPy, scikit-learn, Matplotlib

All dependencies are installed automatically via `pip install -e .`.


## Quick Start

### Single slice

```bash
python scripts/run_single_slice.py data/molecules_full_add.csv.gz --slice MERFISH_0.19.h5ad
```

### All slices benchmark

```bash
python scripts/run_all_slices.py data/molecules_full_add.csv.gz
```

### Multi-slice alignment

```bash
python scripts/multislice_alignment.py data/molecules_full_add.csv.gz
```

### Dimensionality analysis

```bash
python scripts/run_dimensionality.py data/molecules_full_add.csv.gz
```

### As a library

```python
from spsheaf import io, fibers, graph, sheaf, clustering, evaluation

# Load and select slice
mol = io.load_data("molecules_full_add.csv.gz")
mol_slice = io.select_slice(mol, "MERFISH_0.19.h5ad")

# Build fibers
grid_coords, raw_fibers, gene_names = fibers.build_kernel_fibers(mol_slice)
fibers_norm = fibers.normalize_fibers(raw_fibers)

# Sheaf construction
edges = graph.build_spatial_graph(grid_coords, k=6)
stalks = sheaf.compute_stalks(fibers_norm, grid_coords, uniform_dim=6)
restrictions = sheaf.compute_restriction_maps(stalks, edges, stalk_dim=6)

# Boundary detection
boundary_raw, boundary_smooth = sheaf.compute_boundary_scores(
    fibers_norm, grid_coords, stalks, edges, restrictions
)

# Clustering + cleanup
labels = clustering.boundary_aware_clustering(
    fibers_norm, grid_coords, boundary_smooth, edges, n_clusters=8
)
labels, n_cleaned = clustering.spatial_cleanup(labels, edges, grid_coords)
```


## Pipeline

The method proceeds in five stages:

### 1. Expression fiber field

A regular grid (spacing δ = 15 μm) is laid over the tissue. At each grid point *v*, a fiber **f**_v ∈ ℝ^G is computed by Gaussian kernel smoothing (bandwidth *h* = 25 μm) of nearby molecules, then normalized to proportions and log-transformed.

### 2. Cellular sheaf construction

A symmetric *k*-NN graph (k = 6) is built on the grid coordinates. At each vertex, a **stalk** F(v) ≅ ℝ^d (d = 6) is defined as the column space of the local PCA basis over the *k* = 50 nearest neighbors. For each edge, **restriction maps** are obtained from the SVD of the stalk overlap matrix (principal angle decomposition).

### 3. Boundary detection

The **sheaf disagreement** on each edge measures how much the projected fiber data disagrees when mapped into the shared edge subspace. Per-vertex boundary scores are computed as the mean disagreement over incident edges, then spatially smoothed (Gaussian, σ = 60 μm).

### 4. Boundary-aware clustering

PCA features are attenuated by an exponential boundary weight w(v) = exp(−α · B̃(v) / median(B̃)), concatenated with spatial coordinates, and clustered with *k*-means. Interior points cluster by expression; boundary points are assigned by spatial proximity.

### 5. Spatial connectivity cleanup

After clustering, small disconnected fragments (< 1% of total points) are detected via BFS on the spatial graph and reassigned to the label of their nearest large-component neighbor. This removes visual artifacts without changing the core algorithm.


## Multi-Slice Alignment

For datasets with multiple serial tissue slices, the `multislice_alignment.py` script registers slices to a common coordinate system using **gene centroid registration** and assesses cross-slice spatial coherence.

### Method

Each gene's spatial expression pattern defines a centroid (mean position of all its molecules). Matching centroids across slices determines the translation needed to align them. Specifically:

1. **Gene centroids.** For each slice and each gene, compute the centroid of all molecule positions.
2. **Stability filtering.** Keep only genes with stable proportions across slices (coefficient of variation < 0.5) and sufficient molecules (≥ 20 per slice). On the Moffitt dataset, all 135 genes pass this filter.
3. **Median consensus.** For each slice pair, compute the per-gene shift (reference centroid minus target centroid) and take the median across all stable genes. The median is robust to outlier genes whose spatial pattern genuinely differs between slices.
4. **Direct-to-reference.** The middle slice serves as the reference. Each slice is aligned directly to it — no pairwise chaining, which would accumulate errors.

### Validation

Two validation approaches confirm the alignment:

- **Perturbation recovery.** Known translations (30–90 μm) are applied to molecule coordinates, alignment is run on both original and perturbed data, and the difference in recovered transforms is compared to the ground truth. The method achieves exact recovery (0.0 μm error) with 135 gene votes.
- **Cross-slice label agreement.** After alignment, inter-slice edges connect nearby points across adjacent slices. Hungarian matching finds the best label correspondence, yielding a mean agreement rate of 63.7% — a measure of biological coherence across tissue depth.

### Results

| Pair | Label agreement |
|------|----------------|
| 0 → 1 | 0.644 |
| 1 → 2 | 0.560 |
| 2 → 3 | 0.684 |
| 3 → 4 | 0.658 |
| **Mean** | **0.637** |

The alignment does not affect per-slice clustering — each slice is still clustered independently. The alignment module serves as a registration and validation tool, confirming that the per-slice domain predictions are spatially coherent across serial sections.


## Repository Structure

```
├── spsheaf/                  # Main package
│   ├── __init__.py
│   ├── io.py                      # Data loading and slice selection
│   ├── fibers.py                  # Kernel-smoothed fiber construction
│   ├── graph.py                   # Spatial k-NN graph
│   ├── sheaf.py                   # Stalks, restriction maps, boundary detection
│   ├── clustering.py              # Boundary-aware clustering + spatial cleanup
│   ├── evaluation.py              # ARI, NMI, Hungarian label matching
│   ├── dimensionality.py          # Local/global intrinsic dimension estimation
│   ├── pipeline.py                # High-level orchestration
│   └── visualization/             # Plotting subpackage
│       ├── boundaries.py          # Single-slice diagnostic plots
│       ├── benchmark.py           # Multi-slice summary plots
│       └── dimensionality.py      # Dimension estimation plots
├── scripts/                       # Entry-point scripts
│   ├── run_single_slice.py
│   ├── run_all_slices.py
│   ├── run_dimensionality.py
│   └── multislice_alignment.py    # Multi-slice registration and validation
├── output_sheaf_boundary/         # Generated results and figures
├── pyproject.toml
├── LICENSE
├── .gitignore
└── README.md
```


## Hyperparameters

| Symbol | Default | Description |
|:-------|:--------|:------------|
| δ      | 15 μm   | Grid spacing |
| h      | 25 μm   | Gaussian kernel bandwidth |
| k      | 6       | Spatial graph neighbors |
| k_nbrs | 50      | Local PCA neighborhood size |
| d      | 6       | Uniform stalk dimension |
| σ_s    | 60 μm   | Boundary field smoothing radius |
| α      | 0.5     | Exponential weighting strength |
| α_s    | 0.5     | Spatial coordinate weight |

The stalk dimension *d* is empirically insensitive across [4, 50]. The remaining hyperparameters (σ_s, α, α_s) were selected by leave-one-out cross-validation across all 5 slices.


## Key Findings

- **Spatial scale is critical.** Domain-level structure emerges at coarse bandwidths (~150 μm), not fine scales (~25 μm) that resolve individual cells.
- **Manifold-aware methods underperform PCA for domain clustering.** Isomap, UMAP, and sheaf Laplacian eigenvectors encode smooth within-domain gradients rather than between-domain discontinuities.
- **The sheaf's value is in boundary detection, not spectral clustering.** Measuring rotation of local PCA bases is the key signal; boundary weighting improves clustering where spectral methods fail.
- **The intrinsic dimensionality is spatially non-uniform** (typically 2–5 per patch), motivating local/adaptive approaches.


## Data

This framework was developed and benchmarked on the **Moffitt et al. (2018)** MERFISH dataset (mouse hypothalamic preoptic region), exported by Yusuke Imoto as `molecules_full_add.csv.gz`:

- ~20.5 million molecules, 135 genes, 5 tissue slices
- Ground truth: 8 spatial domain labels
- No cell segmentation required

The data file is not included in this repository due to size. Place it in a `data/` directory or pass the path as a command-line argument.


## References

1. J. Hansen & R. Ghrist, *Toward a Spectral Theory of Cellular Sheaves*, J. Appl. Comput. Topol. (2019)
2. A. Singer & H.-T. Wu, *Vector diffusion maps and the connection Laplacian*, Comm. Pure Appl. Math. 65(8), 2012
3. Å. Björck & G.H. Golub, *Numerical methods for computing angles between linear subspaces*, Math. Comp. 27(123), 1973
4. Z. Yuan et al., *Benchmarking spatial clustering methods with spatially resolved transcriptomics data*, Nature Methods 21, 2024
5. Y. Si et al., *FICTURE: scalable segmentation-free analysis of submicron-resolution spatial transcriptomics*, Nature Methods 21, 2024
6. J.R. Moffitt et al., *Molecular, spatial, and functional single-cell profiling of the hypothalamic preoptic region*, Science 362, 2018


## License

MIT License. See [LICENSE](LICENSE).
