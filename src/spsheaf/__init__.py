"""spSheaf: sheaf-theoretic spatial domain inference from raw molecules.

The method is eight function calls (see ``examples/quickstart.py``):
``fibers.build_kernel_fibers`` and ``fibers.normalize_fibers`` put
kernel-smoothed expression fibers on a grid; ``graph.build_spatial_graph``
links the grid; ``sheaf.compute_stalks`` fits a local principal subspace at
every vertex; ``sheaf.compute_restriction_maps`` aligns neighbouring stalks;
``sheaf.compute_boundary_scores`` scores every edge by the residual
disagreement of the aligned coordinates and smooths it into a field;
``clustering.boundary_aware_clustering`` partitions the grid with that field
attenuating the transitions; ``clustering.spatial_cleanup`` removes
fragments. ``pipeline.run_single_slice`` wraps the eight steps and the
evaluation against an annotation.
"""
__version__ = "0.2.0"
__all__ = ["fibers", "graph", "sheaf", "clustering", "evaluation", "pipeline", "io", "paths"]
