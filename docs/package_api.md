# `spsheaf` — package API

Eight steps, one module each; every function takes and returns NumPy arrays
or pandas DataFrames. Coordinates are in the input's units.

| Step | Function | Notes |
|---|---|---|
| load | `io.load_data(path)`, `io.select_slice(mol, name)` | MERFISH molecule table; `SPSHEAF_MERFISH_INPUT=section` selects the whole physical section by bregma |
| 1 fibers | `fibers.build_kernel_fibers(mol, grid_spacing, bandwidth, min_molecules=15, x_col, y_col, z_col=None)` → `grid, fibers, gene_names` | Gaussian kernel, search radius 3h; drops grid points below `min_molecules` |
| | `fibers.build_feature_fibers(coords, features, grid_spacing, bandwidth)` | continuous feature fibers (3D, protein, chromatin marks) |
| | `fibers.normalize_fibers(f)` | `log(1 + 1000·f/‖f‖₁)` |
| 2 graph | `graph.build_spatial_graph(grid, k=6)` → `edges` | symmetric kNN |
| 3 stalks | `sheaf.compute_stalks(fibers, grid, uniform_dim=6, k_nbrs=50)` | local PCA of the `k_nbrs` nearest fibers (self included), locally centred |
| 4 restriction maps | `sheaf.compute_restriction_maps(stalks, edges, stalk_dim)` | SVD of `U_uᵀU_v = LΣRᵀ` per edge |
| 5–6 disagreement, field | `sheaf.compute_boundary_scores(fibers, grid, stalks, edges, rs, smooth_radius, reference=None)` → `raw, smooth` | `D(e) = ‖Lᵀξ_u − Rᵀξ_v‖²`; `reference` = `"tissue"` (default, `ξ_v = U_vᵀ(f_v − f̄)`, `f̄` the section mean) or `"local"` (`f_v − μ_v`, the neighbourhood mean); env `SPSHEAF_REFERENCE` |
| 7 clustering | `clustering.boundary_aware_clustering(fibers, grid, smooth, edges, n_clusters, spatial_weight=0.5, boundary_alpha=0.5, n_pca=10, n_init=20, random_state=42)` | `w = exp(−α_b B̃/median B̃)` on standardized PCA scores, `[Ẑ_w | α_s S]`, k-means |
| 8 cleanup | `clustering.spatial_cleanup(labels, edges, grid_coords, min_fragment_frac=0.01)` → `labels, n_reassigned` | fragments < τ of grid points take an adjacent label |
| all | `pipeline.run_single_slice(mol, slice_name, n_clusters=8, ...)`, `pipeline.run_all_slices(path, ...)` | the steps plus cell-level evaluation |

Evaluation: `evaluation.map_ground_truth`, `hungarian_match_labels`,
`compute_ari/nmi/homogeneity/completeness/chaos/pas/asw`,
`evaluate_clustering_full` (the SDMBench panel; the competitors are scored
through the same functions). `boundary_metrics.evaluate_field` scores a field
against annotated boundaries (AP, F1 at a tolerance, distances).
`edge_scores.compute_edge_score(kind, ...)` gives the ablation arms.
`paths.data_root()/raw()/cache()/results()` resolve the data tree
(`SPSHEAF_DATA`, then `paths.toml`, then `<repo>/data`).
