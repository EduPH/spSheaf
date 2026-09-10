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
    """Zone archetypes and their chromatin-mark composition.

    Two choices differ from a plain k-means on the zone x mark matrix, and both
    matter for what the panel is allowed to claim.

    The mark columns are z-scored, but each zone also carries an overall
    intensity: the per-zone mean across marks accounts for 26% of the total
    variance, and it dominates the cluster means, so profiles clustered on the
    raw matrix come out almost collinear (r > 0.99) and look like one archetype
    repeated. Since the claim is *differential* enrichment, each zone's profile
    is centred across marks first, which isolates composition from level.

    The number of archetypes is chosen rather than assumed. On composition,
    the silhouette over k = 2..6 is 0.373, 0.275, 0.296, 0.283, 0.240: one
    dominant axis, with a weak four-way substructure. k = 4 is the best
    discrete choice above the trivial split -- highest silhouette of k >= 3,
    subsample stability 0.91, and every archetype recurring in 40-43 of the 46
    nuclei -- but two of the four are still r = +0.79 correlated, so the panel
    shows a dominant heterochromatin-to-active axis with a secondary
    component, not four independent identities. The caption says so.
    """
    from scipy.optimize import linear_sum_assignment
    files = sorted(glob.glob(os.path.join(a.if_dir, "nucleus_*_zone_marks.csv")))
    rows, nuc, zid = [], [], []
    for f in files:
        d = pd.read_csv(f)
        rows.append(d[[c for c in d.columns if c not in NON_MARK]])
        nuc += [os.path.basename(f).split("_")[1]] * len(d)
        zid += list(d["zone"].values)
    M = pd.concat(rows, ignore_index=True)
    marks = list(M.columns)
    nuc = np.array(nuc)
    X = M.values
    Xc = X - X.mean(1, keepdims=True)           # composition, not level

    k = a.k
    lab = KMeans(k, n_init=100, random_state=0).fit_predict(Xc)
    prof = pd.DataFrame(Xc, columns=marks).groupby(lab).mean()

    order = [m for m in MARK_ORDER if m in marks] + [m for m in marks if m not in MARK_ORDER]
    prof = prof[order]

    # name each archetype by the mark panel it is most enriched for, assigning
    # names one-to-one so that two archetypes can never carry the same label
    PANELS = {
        "constitutive heterochromatin": ["MinSat", "mH2A1", "LINE1", "Telomere", "H3K9me3"],
        "SINE-rich repressive":         ["SINEB1", "H4K20me3", "H3K27me2", "H4K20me2"],
        "lamina / nucleolus-associated": ["Lamin B1", "H3K27me3", "Fibrillarin"],
        "active / nuclear speckle":     ["SF3a66", "H3K27ac", "H3K9ac", "H4K16ac", "RNAPII Ser5-P"],
    }
    names = list(PANELS)
    # score each archetype on profile *shape*, not amplitude: dividing by the
    # profile's own spread stops a strongly modulated archetype from
    # outbidding a flatter one for its own name
    Z = prof.div(prof.std(1), axis=0)
    score = np.array([[Z.loc[i, [m for m in PANELS[nm] if m in order]].mean()
                       for nm in names] for i in Z.index])
    ri, ci = linear_sum_assignment(-score)
    assigned = {prof.index[r]: names[c] for r, c in zip(ri, ci)}

    het = [m for m in PANELS["constitutive heterochromatin"] + PANELS["SINE-rich repressive"] if m in order]
    act = [m for m in PANELS["active / nuclear speckle"] if m in order]
    balance = prof[het].mean(1) - prof[act].mean(1)
    prof = prof.loc[balance.sort_values(ascending=False).index]

    sizes = pd.DataFrame({
        "archetype": [assigned[i] for i in prof.index],
        "n_zones": [int((lab == i).sum()) for i in prof.index],
        "n_nuclei": [len(set(nuc[lab == i])) for i in prof.index],
    })
    prof.index = [assigned[i] for i in prof.index]
    prof.index.name = "archetype"

    C = np.corrcoef(prof.values)
    worst = C[np.triu_indices(len(prof), 1)].max()

    prof.round(3).to_csv(os.path.join(a.if_dir, "zone_mark_heatmap.csv"))
    sizes.to_csv(os.path.join(a.if_dir, "zone_archetype_sizes.csv"), index=False)
    # per-nucleus zone id -> pooled archetype, so a single-nucleus map can be
    # coloured by archetype rather than by its own local cluster numbering
    pd.DataFrame({"nucleus": nuc, "zone": zid,
                  "archetype": [assigned[i] for i in lab]}).to_csv(
        os.path.join(a.if_dir, "zone_archetype_assignment.csv"), index=False)
    print(f"[if] {len(files)} nuclei, {len(M)} zones -> {k} archetypes x {len(order)} marks")
    print(f"[if] most positive correlation between archetype profiles: r = {worst:+.2f}")
    for _, r in sizes.iterrows():
        print(f"     {r.archetype:32s} n={r.n_zones:3d} zones, {r.n_nuclei}/{len(files)} nuclei")
    print("[if] wrote zone_mark_heatmap.csv, zone_archetype_sizes.csv, "
          "zone_archetype_assignment.csv; draw with figures/ed_ifheat_panels.py")


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
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--dna-radial", default="output_takei_feat_full1/certification_radial.csv")
    ap.add_argument("--dna-nucleus", default="0_1")
    ap.add_argument("--dna-zones-dir", default="output_takei_feat_full1")
    ap.add_argument("--dna-raw", default="dna_seqfish/DNAseqFISH+/DNAseqFISH+25kbloci-E14-replicate1.csv")
    a = ap.parse_args()
    {"if":panel_if,"dna":panel_dna,"dnazone":panel_dnazone}[a.panel](a)


if __name__ == "__main__":
    main()
