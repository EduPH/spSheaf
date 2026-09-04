import os
import gzip
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.neighbors import KNeighborsClassifier

# Import computational functions
from spsheaf import io, fibers
from spsheaf.dimensionality import (
    build_atlas_cover,
    estimate_dim_per_patch,
    estimate_dim_all_patches,
    local_dim_field_pca,
    local_dim_field_isomap,
    select_manifold_genes
)

# Import visualization module
from spsheaf.viz import dimensionality as dim_vis

# =========================================================
# 1. PARAMETERS & CONFIGURATION
# =========================================================
mol_path = "E16.5_E1S3_GEM_bin1.tsv.gz"
slice_name = "StereoSeq_E16.5_E1S3"
output_dir = "output_stereoseq_dimensionality"

# Scaled for Stereo-seq coordinate system
grid_spacing = 1500     # Resolution of the expression grid
bandwidth = 1500        # Smoothing of expression vectors
patch_size = 5000      # Large enough to cover tissue neighborhoods
overlap = 0.1          # Patch overlap for atlas cover
k_local = 20           # Neighbors for intrinsic dimensionality

os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 2. DATA LOADING & ALIGNMENT
# =========================================================
def load_stereoseq_raw(path):
    """
    Load raw molecule coordinates from the STOmics GEM file.
    Safely skips metadata lines starting with '#'.
    """
    skip_lines = 0
    open_func = gzip.open if path.endswith('.gz') else open
    
    with open_func(path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                skip_lines += 1
            else:
                break
                
    mol = pd.read_csv(path, sep='\t', skiprows=skip_lines)
    mol.rename(columns={"geneID": "gene"}, inplace=True)
    mol["matched_h5ad_file"] = slice_name
    return mol

print("--> Loading and aligning Stereo-seq data...")
mol = load_stereoseq_raw(mol_path)
mol_single = io.select_slice(mol, slice_name)

# =========================================================
# 3. FIBER CONSTRUCTION
# =========================================================
print(f"--> Building expression manifold (spacing={grid_spacing}, bw={bandwidth})...")
grid_coords, raw_fibers, gene_names = fibers.build_kernel_fibers(
    mol_single, grid_spacing=grid_spacing, bandwidth=bandwidth,
)
fibers_norm = fibers.normalize_fibers(raw_fibers)

# =========================================================
# 4. DIMENSIONALITY ESTIMATION
# =========================================================
print(f"--> Building atlas cover (patch_size={patch_size})...")
subset_assignments, patch_centers = build_atlas_cover(
    grid_coords, patch_size=patch_size, overlap=overlap,
)

print("--> Estimating per-patch and local dimensionality...")
# 4a. Per-patch PCA/Isomap analysis
# Capped at 50 or gene count - 1 to prevent LinAlgErrors on massive vocabularies
max_d = min(15, len(gene_names) - 1)

patch_dim = estimate_dim_per_patch(
    fibers_norm, subset_assignments,
    max_dim=max_d,
    n_patches=10,
)
dim_vis.plot_dim_per_patch(patch_dim, output_dir=output_dir)

# 4b. Global spatial distribution of dimensionality
pca_all, iso_all = estimate_dim_all_patches(
    fibers_norm, subset_assignments, patch_centers,
    max_dim=max_d,
)
dim_vis.plot_dim_all_patches(
    patch_centers, pca_all, iso_all, grid_coords, 
    output_dir=output_dir
)

# 4c. Local intrinsic dimension fields (point-by-point)
dim_pca = local_dim_field_pca(fibers_norm, grid_coords, k=k_local)
dim_iso = local_dim_field_isomap(fibers_norm, grid_coords, k=k_local)

dim_vis.plot_local_dim_fields(grid_coords, dim_pca, dim_iso, output_dir=output_dir)
dim_vis.plot_dim_histograms(dim_pca, dim_iso, output_dir=output_dir)

# =========================================================
# 5. GENE SELECTION
# =========================================================
print("--> Identifying manifold-relevant genes...")
gene_sel = select_manifold_genes(
    fibers_norm, grid_coords, gene_names, latent_dim=3, k=k_local,
)
dim_vis.plot_gene_selection(gene_sel, gene_names, output_dir=output_dir)

# =========================================================
# 6. SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("DIMENSIONALITY ANALYSIS SUMMARY")
print("=" * 60)
print(f"  Slice            : {slice_name}")
print(f"  Grid points      : {len(grid_coords):,}")
print(f"  Genes            : {len(gene_names):,}")
print(f"  Patches          : {len(subset_assignments)}")
print(f"  Local dim PCA    : mean={dim_pca.mean():.2f}, median={np.median(dim_pca):.0f}")
print(f"  Local dim Iso    : mean={dim_iso.mean():.2f}, median={np.median(dim_iso):.0f}")
print(f"  Top manifold genes : {', '.join(gene_names[gene_sel['indices'][:5]])}...")
print(f"\n  Outputs saved to: {output_dir}/")