#!/usr/bin/env python
"""
Certify that the inferred zones (chromosome-intensity fiber) recover
chromosome territories, and that the territory organization is reproducible
across nuclei.

For every nucleus with a saved zones CSV, each locus dot is assigned the zone
of its nearest grid point, then we measure:
  * NMI(zone, chromID)  — do zones capture chromosome identity? (0 = chance)
  * zone purity         — mean over zones of the dominant chromosome's fraction
  * per-chromosome radial position (0 = nuclear centre, 1 = periphery)

Reproducibility = how consistent each chromosome's radial position is across
nuclei (low spread ⇒ chromosomes sit in reproducible territories).

Usage
-----
    python certify_territories.py \
        --data dna_seqfish/DNAseqFISH+/DNAseqFISH+25kbloci-E14-replicate1.csv \
        --zones-dir output_takei_feat --out-dir output_takei_feat
"""
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.metrics import normalized_mutual_info_score


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="DNAseqFISH+ 25kb CSV.")
    p.add_argument("--zones-dir", default="output_takei_feat")
    p.add_argument("--out-dir", default="output_takei_feat")
    p.add_argument("--xy-nm", type=float, default=103.0)
    p.add_argument("--z-nm", type=float, default=250.0)
    return p


def main():
    a = build_parser().parse_args()
    df = pd.read_csv(a.data)
    df["_cell"] = df.fov.astype(str) + "_" + df.cellID.astype(str)
    zfiles = sorted(glob.glob(os.path.join(a.zones_dir, "nucleus_*_zones.csv")))
    print(f"{len(zfiles)} nuclei with saved zones")

    per_nuc, radial_rows = [], []
    for zf in zfiles:
        cid = re.search(r"nucleus_(.+)_zones\.csv", os.path.basename(zf)).group(1)
        sub = df[df._cell == cid]
        if len(sub) < 100:
            continue
        z = pd.read_csv(zf)
        grid = z[["x", "y", "z"]].values
        coords = np.column_stack([
            (sub.x.values - sub.x.min()) * a.xy_nm,
            (sub.y.values - sub.y.min()) * a.xy_nm,
            (sub.z.values - sub.z.min()) * a.z_nm])
        _, j = cKDTree(grid).query(coords, k=1)
        dot_zone = z["zone"].values[j]
        chrom = sub.chromID.values

        nmi = normalized_mutual_info_score(chrom, dot_zone)
        # purity: dominant chromosome fraction per zone, dot-weighted
        ct = pd.crosstab(dot_zone, chrom)
        purity = (ct.max(axis=1).sum()) / ct.values.sum()
        per_nuc.append(dict(cell=cid, dots=len(sub), zones=len(set(dot_zone)),
                            nmi_zone_chrom=round(nmi, 3), purity=round(purity, 3)))

        # per-chromosome radial position (centre=0, edge=1), this nucleus
        c = coords - coords.mean(0)
        r = np.sqrt((c ** 2).sum(1)); r = r / r.max()
        for ch in np.unique(chrom):
            radial_rows.append(dict(cell=cid, chrom=int(ch),
                                    radial=float(r[chrom == ch].mean())))

    nuc = pd.DataFrame(per_nuc)
    rad = pd.DataFrame(radial_rows)
    rad_stats = rad.groupby("chrom")["radial"].agg(["mean", "std"]).reset_index()

    nuc.to_csv(os.path.join(a.out_dir, "certification_per_nucleus.csv"), index=False)
    rad_stats.to_csv(os.path.join(a.out_dir, "certification_radial.csv"), index=False)

    print("\n=== zones recover chromosome territories? ===")
    print(f"  NMI(zone, chromID)  : mean {nuc.nmi_zone_chrom.mean():.3f} "
          f"(chance = 0)  [{nuc.nmi_zone_chrom.min():.2f}-{nuc.nmi_zone_chrom.max():.2f}]")
    print(f"  zone chromosome purity: mean {nuc.purity.mean():.3f}")
    print("\n=== reproducibility of chromosome radial position across nuclei ===")
    print(f"  mean within-chromosome std of radial position: "
          f"{rad_stats['std'].mean():.3f} (lower = more reproducible)")
    print("  most interior chroms:", rad_stats.nsmallest(3, 'mean').chrom.tolist(),
          " most peripheral:", rad_stats.nlargest(3, 'mean').chrom.tolist())

    # plot: per-chromosome radial position, mean +/- std across nuclei
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    rs = rad_stats.sort_values("mean")
    ax[0].errorbar(range(len(rs)), rs["mean"], yerr=rs["std"], fmt="o",
                   capsize=3)
    ax[0].set_xticks(range(len(rs)))
    ax[0].set_xticklabels([f"chr{c}" for c in rs.chrom], rotation=90, fontsize=8)
    ax[0].set_ylabel("radial position (0=centre, 1=edge)")
    ax[0].set_title(f"Chromosome radial position across {nuc.shape[0]} nuclei\n"
                    "(reproducible territories = tight error bars)")
    ax[1].hist(nuc.nmi_zone_chrom, bins=12, color="steelblue")
    ax[1].axvline(nuc.nmi_zone_chrom.mean(), color="crimson", ls="--",
                  label=f"mean {nuc.nmi_zone_chrom.mean():.2f}")
    ax[1].set_xlabel("NMI(zone, chromID) per nucleus")
    ax[1].set_ylabel("nuclei"); ax[1].set_title("Zones capture chromosome identity")
    ax[1].legend()
    fig.tight_layout()
    p = os.path.join(a.out_dir, "certification_territories.png")
    fig.savefig(p, dpi=400, bbox_inches="tight"); plt.close(fig)
    print(f"\n-> {p}")
    print(f"-> {os.path.join(a.out_dir, 'certification_per_nucleus.csv')}")


if __name__ == "__main__":
    main()
