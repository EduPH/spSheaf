# MERSCOPE — Vizgen mouse brain receptor showcase

A whole coronal hemisection, ~9.4 × 6.5 mm: 54.7 M molecules, a 649-gene
panel, 7 focal planes. Used for scale and for anatomical plausibility.

## Run

No `make` target.

```bash
python -m experiments.brain_receptor.run_brain_receptor_showcase \
  --input data/raw/brain_receptor/datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_detected_transcripts_S1R1.csv \
  --top-genes 300 --subsample 0.25 --grid-spacing 50 --bandwidth 100 \
  --smooth-radius 200 --n-clusters 15 \
  --output-dir data/results/brain_receptor/output_brain_receptor

# optional: score against the Allen reference atlas
pip install brainglobe-atlasapi
python -m experiments.brain_receptor.reference_allen \
  --labels .../domain_labels.csv --ap-um 6800
```

`--subsample` defaults to 0.3 in the script; the published run used 0.25, so
pass it. `reference_allen.py` downloads the Allen CCFv3 (~150–600 MB, cached
in `~/.brainglobe`) and therefore needs network access.

After the z-filter, the top-300 gene filter (which retains 98.8 % of
molecules) and a 25 % subsample, the run works on 1.85 M molecules over
19,455 grid points.

## Results

**15 domains, spatial coherence 0.923**, silhouette 0.108, boundary contrast
1.03, domain-size CV 0.377. The domains are bilaterally symmetric without
symmetry being imposed anywhere, and are anatomically recognizable: an
isocortical band, white matter, hippocampal formation, thalamus and ventral
hypothalamus.

No spatial-domain ground truth ships with this dataset, so bilateral symmetry
and correspondence with known anatomy are what can be checked.

**Allen CCF registration** (`metrics_vs_allen.csv`, 10 CCF divisions, 19,002
grid points, 96.8 % overlap): ARI 0.264, NMI 0.460; after merging to matched
divisions, accuracy 0.622, ARI 0.425.

> The manuscript reports these (ED Fig. 5 caption) as a reference rather than
> a benchmark: the registration is a rigid AP-plane match, so the scores
> confound domain quality with registration error.

**3D.** `run_brain_receptor_3d.py` runs the same code over all 7 focal planes
(117,183 grid points, coherence 0.987; a 2 × 2 mm crop gives 8 domains at
0.974). This exercises the N-D code path — the imaged slab is ~10 µm against a
~9 mm field, so the domains are near-columnar and 3D adds no biology here.
Not in the manuscript.

Feeds Extended Data Fig. 5b–d, at the tissue reference (Supplementary Table 3).
