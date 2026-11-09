"""Per-chromosome radial position, replicate 1 against replicate 2.

Reads the two certification_radial.csv tables written by certify_territories
and reports Pearson and Spearman correlation of the mean radial position over
the 20 chromosomes, together with the per-replicate zone-chromosome NMI.
Writes dna_radial_replicates.csv (one row per chromosome, both replicates)
and dna_replicates_summary.csv next to the first table. Extended Data Fig. 6a.

    python -m experiments.takei.compare_replicates \
        --rep1 .../output_takei_feat_full1 --rep2 .../output_takei_feat_full2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--rep1", required=True); a.add_argument("--rep2", required=True)
    a.add_argument("--out-dir", default=None)
    a = a.parse_args()
    r1, r2 = Path(a.rep1), Path(a.rep2)
    out = Path(a.out_dir) if a.out_dir else r1
    t1 = pd.read_csv(r1 / "certification_radial.csv"); t2 = pd.read_csv(r2 / "certification_radial.csv")
    m = t1.merge(t2, on="chrom", suffixes=("_rep1", "_rep2"))
    pr, pp = pearsonr(m.mean_rep1, m.mean_rep2); sr, sp = spearmanr(m.mean_rep1, m.mean_rep2)
    n1 = pd.read_csv(r1 / "certification_per_nucleus.csv"); n2 = pd.read_csv(r2 / "certification_per_nucleus.csv")
    summ = dict(n_nuclei_rep1=len(n1), n_nuclei_rep2=len(n2),
                nmi_rep1=n1.nmi_zone_chrom.mean(), nmi_rep2=n2.nmi_zone_chrom.mean(),
                purity_rep1=n1.purity.mean(), purity_rep2=n2.purity.mean(),
                pearson_r=pr, pearson_p=pp, spearman_rho=sr, spearman_p=sp,
                most_interior_rep1=int(m.loc[m.mean_rep1.idxmin(), "chrom"]),
                most_interior_rep2=int(m.loc[m.mean_rep2.idxmin(), "chrom"]))
    m.to_csv(out / "dna_radial_replicates.csv", index=False)
    pd.DataFrame([summ]).to_csv(out / "dna_replicates_summary.csv", index=False)
    print(pd.Series(summ).to_string())


if __name__ == "__main__":
    main()
