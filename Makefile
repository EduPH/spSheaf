.PHONY: help install install-benchmark test merfish osmfish segfree clean-cache \
        figures figures-verify figures-test figures-all figures-audit figures-panels panels-osmfish panels-pecab matched-selection

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

merfish:            ## MERFISH benchmark, all five slices
	python -m experiments.merfish.run_all_slices

osmfish:            ## osmFISH cortex
	python -m experiments.osmfish.run_osmFISH

PECAB_RAW := data/raw/pecab/PECAb_dataset/6_Carcinosarcoma_33x220plex/Raw

panels-osmfish:     ## re-run osmFISH and dump the data Fig. 3 is drawn from
	python -m experiments.osmfish.run_osmFISH

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
# 180 mm canvas, 6/8/10 pt type, no frames on spatial panels, >=300 dpi.
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
	  $(FIGDIR)/out/ed_ifheat.pdf --width-mm 180

figures-audit:      ## resolution of every included image (needs the manuscript)
	@test -f "$(TEX)" || { echo "set TEX=<path to the .tex>; the manuscript is not part of this repository"; exit 1; }
	python $(FIGDIR)/audit_figures.py "$(TEX)"

figures-test:       ## unit tests for the figure pipeline itself
	cd $(FIGDIR) && python -m pytest test_figstyle.py -q

figures-panels:     ## export every panel separately, for hand composition
	cd $(FIGDIR) && python export_panels.py

figures-all: figures figures-verify  ## build then check

matched-selection:  ## memo 5 / Y.I. 2.6: one selection rule for all four methods
	python -m benchmarks.segfree.matched_selection \
	  --csv benchmarks/segfree/results/matched_selection_ari.csv
