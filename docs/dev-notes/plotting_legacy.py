from spsheaf import io, fibers, graph, sheaf, clustering, evaluation
import random as random   
mol=io.load_data("Experiments/MERFISH/moffitt2018_export_full/molecules_full_add.csv.gz")

verbose= True
grid_spacing = 15
bandwidth= 25
k_graph = 6
k_nbrs= 50
uniform_dim = 6
spatial_weight= 0.5
boundary_alpha= 0.5
overcluster_factor= 1.0
smooth_radius=60.0
n_clusters= 8
slice_info = io.identify_slices(mol, verbose=verbose)
slice_names = sorted(slice_info["slice_name"].tolist())

slice_name= slice_names[0]
mol_single = io.select_slice(mol, slice_name, verbose=verbose)

# 2. Build fibers
grid_coords, raw_fibers, gene_names = fibers.build_kernel_fibers(
    mol_single, grid_spacing=grid_spacing, bandwidth=bandwidth,
    verbose=verbose,
)
fibers_norm = fibers.normalize_fibers(raw_fibers)

# 3. Spatial graph
edges = graph.build_spatial_graph(grid_coords, k=k_graph, verbose=verbose)

# 4. Stalks
stalks = sheaf.compute_stalks(
    fibers_norm, grid_coords,
    uniform_dim=uniform_dim, k_nbrs=k_nbrs, verbose=verbose,
)

# 5. Restriction maps
restrictions = sheaf.compute_restriction_maps(
    stalks, edges, uniform_dim, verbose=verbose,
)

# 6. Boundary detection
boundary_raw, boundary_smooth = sheaf.compute_boundary_scores(
    fibers_norm, grid_coords, stalks, edges, restrictions,
    smooth_radius=smooth_radius, verbose=verbose,
)

# 7. Boundary-aware clustering
labels = clustering.boundary_aware_clustering(
    fibers_norm, grid_coords, boundary_smooth, edges,
    n_clusters=n_clusters, spatial_weight=spatial_weight,
    boundary_alpha=boundary_alpha,
    overcluster_factor=overcluster_factor, verbose=verbose,
)


indices=random.sample(range(len(coords)),20000)
x=mol_single['x_local']
y=mol_single['y_local']
c = mol_single['gene_index']
coords =np.column_stack((x,y))
plt.scatter(coords[:,0][indices],coords[:,1][indices],c=colors[indices],s=1)
plt.axis('equal')
plt.axis('off')
plt.savefig('raw.png', dpi=600) 
plt.show()



sampled_edges = random.sample(edges, k=10000)
G = nx.Graph()
G.add_edges_from(edges)
nx.draw(G, grid_coords, with_labels=False, node_color='skyblue', edge_color='gray', node_size=0.1, width=0.1)
plt.axis('equal')
plt.axis('off')
plt.savefig('graph_subsample.png', dpi=600) 
plt.show()


c = boundary_smooth
plt.scatter(grid_coords[:,0],grid_coords[:,1],c=c,s=1)
plt.axis('equal')
plt.axis('off')
plt.savefig('boundary_smooth.png', dpi=600) 
plt.show()

c = boundary_raw
plt.scatter(grid_coords[:,0],grid_coords[:,1],c=c,s=1)
plt.axis('equal')
plt.axis('off')
plt.savefig('boundary_raw.png', dpi=600) 
plt.show()


c = np.sum(fibers_norm, axis=1)
plt.scatter(grid_coords[:,0],grid_coords[:,1],c=c,s=1)
plt.axis('equal')
plt.axis('off')
plt.savefig('fibers.png', dpi=600) 
plt.show()