#!/usr/bin/env python
"""
Reference-paper-style panels for the Takei nuclear datasets (cf. Takei et al.
2021, Nature, Fig. 3a: differential chromatin-mark enrichment per zone).

  --panel if     IF zone-archetype x chromatin-mark enrichment heatmap
                 (pools all 46 nuclei; zones clustered into recurring archetypes)
  --panel dna    per-chromosome radial position vs mouse-chromosome gene density
                 (validation of the chr19-interior / gene-rich claim), with a
                 per-chromosome mean +/- s.e.m. bar panel
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# biological ordering of the 21 IF marks (heterochromatin -> active -> structural)
MARK_ORDER = ["MinSat","mH2A1","LINE1","SINEB1","H3K9me3","H3K9me2","H3K27me3",
              "H3K27me2","H4K20me3","H4K20me2","Lamin B1","Telomere",
              "H3K9ac","H3K27ac","H4K16ac","H4K20me1","RNAPII Ser5-P",
              "Fibrillarin","SF3a66","Geminin","H3pSer10","DAPI"]
NON_MARK = {"zone","top_marks","hetero_z","active_z"}


def panel_if(a):
    files = sorted(glob.glob(os.path.join(a.if_dir, "nucleus_*_zone_marks.csv")))
    rows = []
    for f in files:
        d = pd.read_csv(f)
        marks = [c for c in d.columns if c not in NON_MARK]
        rows.append(d[marks])
    M = pd.concat(rows, ignore_index=True)
    marks = list(M.columns)
    X = M.values
    k = a.k
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
    prof = pd.DataFrame(X, columns=marks).groupby(lab).mean()
    # order marks biologically (keep those present, then any extras)
    order = [m for m in MARK_ORDER if m in marks] + [m for m in marks if m not in MARK_ORDER]
    prof = prof[order]
    # order archetypes by heterochromatin - active balance (peripheral first)
    het = [m for m in ["MinSat","mH2A1","LINE1","H3K9me3","H3K9me2","H3K27me3","H4K20me3","Lamin B1"] if m in order]
    act = [m for m in ["H3K9ac","H3K27ac","H4K16ac","RNAPII Ser5-P","Fibrillarin","SF3a66"] if m in order]
    balance = prof[het].mean(1) - prof[act].mean(1)
    prof = prof.loc[balance.sort_values(ascending=False).index]
    sizes = pd.Series(lab).value_counts()
    def tag(rowvals):
        top = [order[j] for j in np.argsort(-rowvals)[:3]]
        hetset = {"MinSat","mH2A1","LINE1","SINEB1","H3K9me3","H3K9me2","H3K27me3","H3K27me2","H4K20me3"}
        if "Lamin B1" in top: return "lamina-associated"
        if any(t in hetset for t in top): return "heterochromatin/repressive"
        if any(t in {"SF3a66","RNAPII Ser5-P","H3K27ac","H3K9ac","H4K16ac"} for t in top): return "active/speckle"
        if "Fibrillarin" in top: return "nucleolar"
        return "mixed"
    labels = [f"Zone type {i+1}: {tag(prof.iloc[i].values)}" for i in range(len(prof))]

    fig, ax = plt.subplots(figsize=(0.42*len(order)+4, 0.6*len(prof)+2))
    im = ax.imshow(prof.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=8)
    ax.set_yticks(range(len(prof))); ax.set_yticklabels(labels, fontsize=9)
    ax.set_ylim(len(prof)-0.5, -0.5)
    for i in range(len(prof)):
        ax.text(len(order)-0.35, i, f"n={sizes[prof.index[i]]}", ha="left", va="center", fontsize=7, color="0.4")
    fig.colorbar(im, ax=ax, shrink=0.6, pad=0.06, label="mean mark z-score")
    ax.set_title(f"IF nuclear zones: chromatin-mark enrichment per zone type\n"
                 f"({len(files)} nuclei, {len(M)} zones pooled into {k} archetypes)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(a.if_dir, "zone_mark_heatmap.png")
    fig.savefig(out, dpi=400, bbox_inches="tight"); plt.close(fig)
    prof.round(2).to_csv(os.path.join(a.if_dir, "zone_mark_heatmap.csv"))
    print(f"[if] -> {out}  ({k} zone archetypes x {len(order)} marks)")
    for lb in labels: print("   ", lb)


# mouse chromosome protein-coding gene density (genes per Mb, GRCm38/mm10; approx.)
GENE_DENSITY = {1:7.0,2:11.6,3:6.4,4:9.9,5:10.7,6:8.4,7:14.7,8:8.7,9:11.2,10:7.2,
                11:16.4,12:6.3,13:6.6,14:6.0,15:7.4,16:5.6,17:11.2,18:4.9,19:12.9}


def panel_dna(a):
    rad = pd.read_csv(a.dna_radial)  # chrom, mean, std
    rad = rad[rad.chrom.isin(GENE_DENSITY)].copy()
    rad["density"] = rad.chrom.map(GENE_DENSITY)
    r = np.corrcoef(rad.density, rad["mean"])[0, 1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # (left) radial vs gene density
    ax[0].errorbar(rad.density, rad["mean"], yerr=rad["std"]/np.sqrt(197), fmt="o",
                   color="steelblue", ecolor="0.7", capsize=2)
    for _, row in rad.iterrows():
        ax[0].annotate(f"chr{int(row.chrom)}", (row.density, row["mean"]),
                       fontsize=7, xytext=(3,3), textcoords="offset points")
    m, b = np.polyfit(rad.density, rad["mean"], 1)
    xs = np.linspace(rad.density.min(), rad.density.max(), 20)
    ax[0].plot(xs, m*xs+b, "--", color="firebrick", lw=1)
    ax[0].set_xlabel("gene density (protein-coding genes per Mb)")
    ax[0].set_ylabel("mean radial position (0 = center, 1 = periphery)")
    ax[0].set_title(f"Gene-rich chromosomes are more interior\nPearson r = {r:+.2f} (n=19 chromosomes)")
    # (right) per-chromosome radial, ordered, with s.e.m.
    ro = rad.sort_values("mean")
    ax[1].bar(range(len(ro)), ro["mean"], yerr=ro["std"]/np.sqrt(197),
              color=["firebrick" if c==19 else "steelblue" for c in ro.chrom], capsize=2)
    ax[1].set_xticks(range(len(ro))); ax[1].set_xticklabels([f"{int(c)}" for c in ro.chrom], fontsize=7)
    ax[1].set_xlabel("chromosome (ordered center -> periphery)")
    ax[1].set_ylabel("mean radial position +/- s.e.m.")
    ax[1].set_title("chr19 (gene-rich) is the most interior")
    ax[1].set_ylim(0.4, 0.7)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(a.dna_radial), "radial_vs_genedensity.png")
    fig.savefig(out, dpi=400, bbox_inches="tight"); plt.close(fig)
    rad.round(3).to_csv(os.path.join(os.path.dirname(a.dna_radial), "radial_vs_genedensity.csv"), index=False)
    print(f"[dna] -> {out}   radial~density r={r:+.3f}")


def panel_dnazone(a):
    """Representative-nucleus zone x chromosome enrichment heatmap: which
    chromosomes define each spatial zone (illustrates chromosome territories)."""
    from scipy.spatial import cKDTree
    fov, cell = a.dna_nucleus.split("_")
    z = pd.read_csv(os.path.join(a.dna_zones_dir, f"nucleus_{a.dna_nucleus}_zones.csv"))
    raw = pd.read_csv(a.dna_raw)
    sub = raw[(raw.fov == int(fov)) & (raw.cellID == int(cell))].copy()
    coords = np.column_stack([(sub.x - sub.x.min())*103.0,
                              (sub.y - sub.y.min())*103.0,
                              (sub.z - sub.z.min())*250.0])
    _, j = cKDTree(z[["x","y","z"]].values).query(coords, k=1)
    sub["zone"] = z.zone.values[j]
    chrs = [f"chr{k}_intensity" for k in list(range(1,20))+["X"]]
    prof = np.log1p(sub.groupby("zone")[chrs].mean())
    Z = (prof - prof.mean(0)) / (prof.std(0) + 1e-9)
    Z.columns = [c.replace("_intensity","") for c in Z.columns]
    fig, ax = plt.subplots(figsize=(0.5*Z.shape[1]+2, 0.6*Z.shape[0]+2))
    im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(Z.shape[1])); ax.set_xticklabels(Z.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(Z.shape[0])); ax.set_yticklabels([f"zone {int(i)}" for i in Z.index])
    for i in range(Z.shape[0]):
        ax.text(int(np.argmax(Z.values[i])), i, "*", ha="center", va="center")
    fig.colorbar(im, ax=ax, shrink=0.7, label="chromosome intensity (z)")
    ax.set_title(f"DNA seqFISH+ zones x chromosome (example nucleus {a.dna_nucleus})\n"
                 "each zone is enriched for distinct chromosomes (chromosome territories)")
    fig.tight_layout()
    out = os.path.join(a.dna_zones_dir, f"zone_chromosome_heatmap_{a.dna_nucleus}.png")
    fig.savefig(out, dpi=400, bbox_inches="tight"); plt.close(fig)
    print(f"[dnazone] -> {out}  ({Z.shape[0]} zones x {Z.shape[1]} chromosomes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True, choices=["if","dna","dnazone"])
    ap.add_argument("--if-dir", default="output_takei_if")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dna-radial", default="output_takei_feat_full1/certification_radial.csv")
    ap.add_argument("--dna-nucleus", default="0_1")
    ap.add_argument("--dna-zones-dir", default="output_takei_feat_full1")
    ap.add_argument("--dna-raw", default="dna_seqfish/DNAseqFISH+/DNAseqFISH+25kbloci-E14-replicate1.csv")
    a = ap.parse_args()
    {"if":panel_if,"dna":panel_dna,"dnazone":panel_dnazone}[a.panel](a)


if __name__ == "__main__":
    main()
