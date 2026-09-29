# PECAb — human uterine carcinosarcoma, multiplexed protein

The modality test: the fiber is a vector of antibody intensities, not
transcript counts. 35-antibody panel (11 phospho, ~15 pathway, ~9
lineage/structural) with a companion 220-gene panel, 7,657 cells, 2
independent sections.

## Run

```bash
make panels-pecab     # both replicates, and dumps the data the figure is drawn from
```

Needs `data/raw/pecab/PECAb_dataset/6_Carcinosarcoma_33x220plex/Raw/6_raw_SiPECAb_Rep{1,2}.csv`.
Settings: grid 150 px, bandwidth 300 px, boundary smoothing 100, k = 5.

## Results

**5 protein-state domains**, within-domain edge fraction **0.968** (Rep 1) and
**0.973** (Rep 2). Silhouette is low (0.02, 0.072) and that is the honest
signal: this tissue is a continuum, not a set of separable compartments.

**The output is a continuous axis plus one discrete compartment**, which is
what the panel supports. One domain is a mesenchymal / myofibroblast
compartment (α-SMA, FN1, vimentin) at |z| = 1.5, against the authors' own five
clusters at mean |z| = 0.055. The rest of the section is a continuous
epithelial-to-mesenchymal gradient rather than a partition.

**Reproducibility across sections.** The 21 lineage-marker pairs correlate
between replicates at **Pearson r = 0.94**. The EMT sign structure holds
independently in both: corr(VIM, FN1) +0.58 / +0.59; corr(FN1, E-cadherin)
−0.60 / −0.48; corr(E-cadherin, EpCAM) +0.25 / +0.38.

The boundary field tracks the EMT gradient magnitude at r = 0.277, against
0.043 for pSMAD3 — so the borders it finds are EMT-axis transitions rather
than the signalling interface reported by the original authors.

> The mesenchymal domain's *index* is run-dependent (0 in Rep 1, 3 in Rep 2).
> Read it from `summary.csv`; do not hard-code it.

`run_pecab.py` holds an earlier framing that treated the section as discrete
protein-state domains (7 domains, coherence 0.955 on the protein panel; 10
domains, 0.946 on the 220-gene panel). It was abandoned — it compared
unfavourably to the authors' own clusters at NMI ≈ 0.02 — and
`docs/dev-notes/pecab_reframe_note.md` records why. The continuous framing is the published
one.

Feeds Extended Data Fig. 8, at the tissue reference; k_nbrs = 40 (Supplementary Table 3).
