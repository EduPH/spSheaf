# Synthetic tissue — calibrating the choice of k

Voronoi tissue with a known number of domains, used only to calibrate the
annotation-free criteria for k (Results, "Choosing the number of domains";
Methods, "Synthetic constructions"). K seed points are placed at random in a
square field, every molecule is assigned to the Voronoi cell of its nearest
seed, and each domain draws genes from its own multinomial over 60 genes (the
softmax of a random unit direction). 2.2 × 10⁵ molecules per construction;
δ 12, h 22, σ_s 50, d 6, α_s 0.5.

Two designs: *flat* (well separated; the K programs are independent
directions) and *nested* (near-sibling; two pairs of domains share a program up
to 0.22 of an independent direction). Two data seeds per design, K = 6,
k = 2–12. The generating domain of a grid point is its nearest seed.

```bash
make k-selection                                   # both designs, K = 6, two seeds -> data/results/synthetic/k_selection_<design>_k6_s<seed>.csv
python -m experiments.synthetic.k_selection --design flat --k-true 5   # any other K
```

Criteria per k: silhouette of the fibers, grid-phase stability (ARI between
the partitions at two lattice phases half a spacing apart, labels transferred
by nearest neighbour), a k-means-seed control, and ARI against the generating
partition. Read as an upper bound on k, stability recovers K on the flat
design (6–7, 6–8) and under-counts on the nested one (4–5, 4–4); ARI against
the generating partition is within 10 % of its maximum over k = 4–8 there.

`figures/ed_kselection.py` draws these two designs next to the five MERFISH
sections. Tracked tables: `../../reported/synthetic/`.
