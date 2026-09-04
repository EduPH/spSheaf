"""
Sheaf-based spatial domain inference for spatial transcriptomics.

A segmentation-free, topologically-motivated framework that detects
tissue domain boundaries by measuring disagreement between local PCA
bases (stalks of a cellular sheaf) and uses boundary-aware clustering
to infer spatial domains.

Reference pipeline:
    1. Gaussian kernel-smoothed expression fibers on a spatial grid
    2. Local PCA stalks → cellular sheaf over a spatial graph
    3. SVD-based restriction maps → sheaf disagreement → boundary scores
    4. Boundary-attenuated clustering (PCA + k-means)
    5. Spatial connectivity cleanup (remove small disconnected fragments)

Modules
-------
io : Data loading and slice selection
fibers : Kernel-smoothed fiber construction and normalization
graph : Spatial k-NN graph construction
sheaf : Stalk construction, restriction maps, boundary detection
clustering : Boundary-aware clustering and spatial connectivity cleanup
evaluation : ARI, NMI, Hungarian label matching
dimensionality : Local/global intrinsic dimension estimation
pipeline : High-level orchestration (single slice, all slices)
visualization : Diagnostic and summary plots
"""

__version__ = "0.1.0"
__author__ = "Eduardo Paluzo-Hidalgo, Yusuke Imoto"
