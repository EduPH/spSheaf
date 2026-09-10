.PHONY: help install install-benchmark test merfish merfish-seeds merfish-section-all local-all reference-check osmfish-seeds osmfish-all spacetx-all osmfish segfree clean-cache \
        figures figures-verify figures-test figures-all figures-audit figures-panels panels-osmfish panels-pecab panels-segfree panels-multislice merfish-loo-cv matched-selection spacetx spacetx-baselines spacetx-segfree spacetx-ficture spacetx-ficture-labels spacetx-insample

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  %-20s %s\n", $$1, $$2}'

install:            ## install spsheaf in editable mode
	pip install -e .

install-pinned:     ## install the exact versions the results were produced on
	pip install -r requirements-lock.txt
	pip install -e .

requirements:       ## regenerate requirements.txt from pyproject.toml
	python tools/gen_requirements.py

install-benchmark:  ## also install the segfree comparators
	pip install -e ".[benchmark]"

install-figures:    ## also install what the figure checks need
	pip install -e ".[figures]"

test:               ## run the test suite
	pytest -q

merfish:            ## MERFISH benchmark, all five slices (input policy: SPSHEAF_MERFISH_INPUT, default 'section')
	python -m experiments.merfish.run_all_slices $(MERFISH_CSV) --output-dir $(RESULTS)/merfish/output_merfish

merfish-seeds:      ## 20 k-means seeds x 5 slices: Table 1 means and ED Fig. 2c
	python -m experiments.benchmark_sdmbench.seed_sensitivity --mol-path $(MERFISH_CSV) \
	  --output-dir $(RESULTS)/benchmark_sdmbench/output_seed_sensitivity

local-all:          ## every headline run for the local-centering variant (SPSHEAF_CENTERING=local; hours)
	$(MAKE) merfish-section-all
	$(MAKE) osmfish
	$(MAKE) spacetx SPACETX=MERFISH_Allen_Prototype
	$(MAKE) spacetx SPACETX=BARISTASeq
	$(MAKE) spacetx-insample

reference-check:    ## osmFISH and the SpaceTx sections with the tissue reference (the optional term), for the sensitivity table
	for l in 1; do \
	  SPSHEAF_REFERENCE=tissue python -m experiments.osmfish.run_osmFISH --no-plot --output-dir $(RESULTS)/osmfish/tissue_reference || exit 1; \
	  SPSHEAF_REFERENCE=tissue python -m experiments.spacetx.run_spacetx MERFISH_Allen_Prototype $(SPACETX_COLS) --grid-spacing 10 --bandwidth 25 --uniform-dim 6 --output-dir $(RESULTS)/spacetx/MERFISH_Allen_Prototype/tissue_reference || exit 1; \
	  SPSHEAF_REFERENCE=tissue python -m experiments.spacetx.run_spacetx BARISTASeq $(SPACETX_COLS) --grid-spacing 10 --bandwidth 25 --uniform-dim 6 --output-dir $(RESULTS)/spacetx/BARISTASeq/tissue_reference || exit 1; \
	done

osmfish-seeds:      ## 20 k-means seeds at the osmFISH defaults (+ alpha_s 0.25)
	python -m experiments.osmfish.seeds

osmfish-all:        ## every osmFISH analysis the manuscript reports, in order
	$(MAKE) osmfish
	$(MAKE) osmfish-seeds
	$(MAKE) osmfish-elongation
	$(MAKE) osmfish-resolution
	$(MAKE) osmfish-rule
	$(MAKE) osmfish-cv
	$(MAKE) osmfish-attribute
	python -m experiments.osmfish.field_at_boundaries

spacetx-all:        ## both SpaceTx sections: transferred run, naive baselines, in-sample grid
	$(MAKE) spacetx SPACETX=MERFISH_Allen_Prototype
	$(MAKE) spacetx SPACETX=BARISTASeq
	$(MAKE) spacetx-baselines SPACETX=MERFISH_Allen_Prototype
	$(MAKE) spacetx-baselines SPACETX=BARISTASeq
	python -m experiments.spacetx.sweep MERFISH_Allen_Prototype --budget-s 3000
	python -m experiments.spacetx.sweep BARISTASeq --budget-s 3000
	$(MAKE) spacetx-insample

merfish-section-all: ## everything the MERFISH input policy touches, in order (hours)
	$(MAKE) merfish
	$(MAKE) merfish-seeds
	for s in 0.04 0.09 0.14 0.19 0.24; do $(MAKE) revision SLICE=MERFISH_$$s.h5ad || exit 1; done
	$(MAKE) panels-segfree
	$(MAKE) panels-multislice
	$(MAKE) merfish-loo-cv

osmfish:            ## osmFISH cortex
	python -m experiments.osmfish.run_osmFISH

PECAB_RAW := data/raw/pecab/PECAb_dataset/6_Carcinosarcoma_33x220plex/Raw
RESULTS := $(shell python -c "from spsheaf import paths; print(paths.results())")
MERFISH_CSV := $(shell python -c "from spsheaf import paths; print(paths.raw('merfish'))")/moffitt2018_export_full/molecules_full_add.csv.gz

panels-osmfish:     ## re-run osmFISH and dump the data Fig. 3 is drawn from
	python -m experiments.osmfish.run_osmFISH

panels-segfree:     ## dump the per-cell domain maps Fig. 2a is drawn from (needs the MERFISH cache)
	python -m experiments.merfish.segfree_panels --slice MERFISH_0.19.h5ad

panels-multislice:  ## re-run the five slices + gene-centroid registration and dump ED Fig. 7's data
	python -m experiments.merfish.multislice_panels

panels-pecab:       ## re-run both PECAb replicates and dump their panel data
	python -m experiments.pecab.pecab_signaling \
	  --input $(PECAB_RAW)/6_raw_SiPECAb_Rep1.csv \
	  --output-dir data/results/pecab/output_pecab_signaling_Rep1
	python -m experiments.pecab.pecab_signaling \
	  --input $(PECAB_RAW)/6_raw_SiPECAb_Rep2.csv \
	  --output-dir data/results/pecab/output_pecab_signaling_Rep2

segfree-inputs:     ## report which benchmark inputs are available
	python -m benchmarks.segfree.inputs

clean-cache:        ## drop derived intermediates (raw data untouched)
	rm -rf data/cache/*

# --- revision experiments (reviewer memo points 2, 3, 5) ---------------
SLICE ?= MERFISH_0.19.h5ad

cache:              ## build the shared MERFISH cache for SLICE (slow, once)
	python -m experiments.merfish._cache --stage slice --slice $(SLICE)
	python -m experiments.merfish._cache --stage build --slice $(SLICE)

boundary-eval:      ## memo 3: score the boundary FIELD directly
	python -m experiments.merfish.boundary_eval --slice $(SLICE)

sheaf-ablation:     ## memo 2: sheaf vs simpler local-subspace controls
	python -m experiments.merfish.sheaf_ablation --slice $(SLICE)

k-sensitivity:      ## memo 5: sweep k, label-free vs label-dependent criteria
	python -m experiments.merfish.k_sensitivity --slice $(SLICE)

revision: cache boundary-eval sheaf-ablation k-sensitivity  ## all three

# --- osmFISH tuning, matched to the MERFISH protocol -------------------
osmfish-diagnose:   ## are the confusable domains separable at all? run first
	python -m experiments.osmfish.tune --stage diagnose

osmfish-rule:       ## MERFISH-transferred config; never sees the annotation
	python -m experiments.osmfish.tune --stage rule

osmfish-cv:         ## leave-one-spatial-block-out, matching MERFISH LOO-CV
	python -m experiments.osmfish.tune --stage cv

osmfish-attribute:  ## which transferred change costs the accuracy? (8 runs)
	python -m experiments.osmfish.tune --stage attribute

osmfish-resolution: ## does domain WIDTH predict which domains are lost?
	python -m experiments.osmfish.resolution

osmfish-elongation: ## are ELONGATED domains lost, and is alpha_s the cause?
	python -m experiments.osmfish.elongation

# --- figures -----------------------------------------------------------
# 180 mm canvas, 6/7/8 pt type (Nature: text 5-7 pt, panel letters 8 pt), no frames on spatial panels, >=300 dpi.
# The spec is enforced, not remembered: figures-verify fails the build.
FIGDIR := figures
#: The audit reads figure widths out of the manuscript, which lives elsewhere.
TEX ?= ../Cellular-Sheaves-for-Spatial-Domain-Inference-in-Spatial-Transcriptomics/docs/Manuscript-NatureMethods/main_spSheaf_revised.tex

figures:            ## build every manuscript figure from data
	cd $(FIGDIR) && python fig1_pipeline.py
	cd $(FIGDIR) && python fig2_manuscript.py
	cd $(FIGDIR) && python ed_osmfish_modes.py
	cd $(FIGDIR) && python ed_benchmark.py
	cd $(FIGDIR) && python fig3_osmfish.py
	cd $(FIGDIR) && python ed_pecab.py
	cd $(FIGDIR) && python ed_scale.py
	cd $(FIGDIR) && python ed_multislice.py
	cd $(FIGDIR) && python ed_spacetx.py
	cd $(FIGDIR) && python ed_composites.py

figures-verify:     ## assert the specification on the composed PDFs
	python $(FIGDIR)/verify_figure.py \
	  $(FIGDIR)/out/fig1.pdf \
	  $(FIGDIR)/out/fig2.pdf \
	  $(FIGDIR)/out/fig3.pdf \
	  $(FIGDIR)/out/ed_osmfish_modes.pdf \
	  $(FIGDIR)/out/ed_benchmark.pdf \
	  $(FIGDIR)/out/ed_pecab.pdf \
	  $(FIGDIR)/out/ed_scale.pdf \
	  $(FIGDIR)/out/ed_merfishslices.pdf \
	  $(FIGDIR)/out/ed_methodval.pdf \
	  $(FIGDIR)/out/ed_dnaterr.pdf \
	  $(FIGDIR)/out/ed_ifheat.pdf \
	  $(FIGDIR)/out/ed_multislice.pdf \
	  $(FIGDIR)/out/ed_spacetx.pdf --width-mm 180

figures-audit:      ## resolution of every included image (needs the manuscript)
	@test -f "$(TEX)" || { echo "set TEX=<path to the .tex>; the manuscript is not part of this repository"; exit 1; }
	python $(FIGDIR)/audit_figures.py "$(TEX)"

figures-test:       ## unit tests for the figure pipeline itself
	cd $(FIGDIR) && python -m pytest test_figstyle.py -q

figures-panels:     ## export every panel separately, for hand composition
	cd $(FIGDIR) && python export_panels.py

figures-all: figures figures-verify  ## build then check

# --- SpaceTx VISp (third annotated dataset) -----------------------------
SPACETX ?= MERFISH_Allen_Prototype
SPACETX_COLS ?= --spot-x xc --spot-y yc --spot-gene gene --cell-id sample_name --cell-x rotated_x --cell-y rotated_y --label layer

spacetx:            ## spSheaf on a SpaceTx dataset at the MERFISH defaults (SPACETX=..., SPACETX_COLS=...)
	python -m experiments.spacetx.run_spacetx $(SPACETX) $(SPACETX_COLS) --grid-spacing 10 --bandwidth 25 --uniform-dim 6

spacetx-baselines:  ## naive k-means baselines on the same dataset
	python -m experiments.spacetx.baselines $(SPACETX)

spacetx-segfree:    ## SSAM and Points2Regions on the same spot table, matched protocol
	python -m benchmarks.segfree.run_spacetx $(SPACETX) both

spacetx-ficture:    ## FICTURE input for the same dataset (then bash benchmarks/segfree/run_ficture.sh ...)
	python -m benchmarks.segfree.prep_ficture_input spacetx $(SPACETX)

spacetx-ficture-labels:  ## per-cell labels of FICTURE's best row (for figures/ed_spacetx.py)
	python -m benchmarks.segfree.ficture_labels spacetx_$(SPACETX)

spacetx-insample:   ## the Allen in-sample bandwidth run that figures/ed_spacetx.py also draws
	python -m experiments.spacetx.run_spacetx MERFISH_Allen_Prototype $(SPACETX_COLS) \
	  --bandwidth 40 --smooth-radius 120 --output-dir $$(python -c "from spsheaf import paths; print(paths.results('spacetx')/'MERFISH_Allen_Prototype'/'insample')")

merfish-loo-cv:     ## leave-one-slice-out selection of (sigma_s, alpha_b, alpha_s); the artifact behind Methods 4.12/4.14
	python -m experiments.merfish.loo_cv

matched-selection:  ## memo 5 / Y.I. 2.6: one selection rule for all four methods
	python -m benchmarks.segfree.matched_selection \
	  --csv benchmarks/segfree/results/matched_selection_ari.csv
