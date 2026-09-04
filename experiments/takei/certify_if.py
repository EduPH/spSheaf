#!/usr/bin/env python
"""
Certify the IF chromatin-mark nuclear zones across all nuclei.

The known ground-truth architecture of a nucleus is radial: constitutive
heterochromatin and the nuclear lamina (Lamin B1, MinSat, mH2A1, H3K9me2/3)
sit at the PERIPHERY, while active chromatin / nucleolus (H3K27ac, H3K9ac,
RNAPII, Fibrillarin) is INTERIOR. If the sheaf zones are real, then across
nuclei a zone's mean radial position should correlate positively with its
heterochromatin score and negatively with its active score, and the
peripheral-most zone should be a lamina/heterochromatin zone.

Reads the per-nucleus outputs of run_takei_if.py (``*_zones.csv`` for voxel
coords, ``*_zone_marks.csv`` for per-zone mark z-scores) and reports the
aggregate.
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

PERIPH = ["Lamin B1", "MinSat", "mH2A1", "H3K9me2", "H3K9me3", "H4K20me3"]
INTERIOR = ["H3K27ac", "H3K9ac", "RNAPII Ser5-P", "Fibrillarin", "SF3a66"]


def zone_radii(zones_csv):
    df = pd.read_csv(zones_csv)
    c = df[["x", "y", "z"]].to_numpy(float)
    center = c.mean(0)
    r = np.linalg.norm(c - center, axis=1)
    rn = r / np.percentile(r, 95)               # normalize by nuclear radius
    out = {}
    for z, g in pd.DataFrame({"zone": df.zone, "rn": rn}).groupby("zone"):
        out[int(z)] = float(g.rn.mean())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="output_takei_if")
    p.add_argument("--output-dir", default="output_takei_if")
    a = p.parse_args()

    rows = []
    for mk in sorted(glob.glob(os.path.join(a.dir, "nucleus_*_zone_marks.csv"))):
        cid = re.search(r"nucleus_(\d+)_", mk).group(1)
        zc = os.path.join(a.dir, f"nucleus_{cid}_zones.csv")
        if not os.path.exists(zc):
            continue
        radii = zone_radii(zc)
        sig = pd.read_csv(mk)
        present_p = [m for m in PERIPH if m in sig.columns]
        present_i = [m for m in INTERIOR if m in sig.columns]
        for _, r in sig.iterrows():
            z = int(r.zone)
            rows.append(dict(cell=int(cid), zone=z, radius=radii.get(z, np.nan),
                             periph_score=float(r[present_p].mean()),
                             interior_score=float(r[present_i].mean()),
                             hetero_z=float(r.get("hetero_z", np.nan)),
                             active_z=float(r.get("active_z", np.nan)),
                             laminB1=float(r.get("Lamin B1", np.nan))))
    Z = pd.DataFrame(rows)
    Z.to_csv(os.path.join(a.output_dir, "certification_zone_table.csv"), index=False)

    # --- per-nucleus correlation: radius vs peripheral/interior score
    def per_cell_corr(col):
        cs = []
        for _, g in Z.groupby("cell"):
            if g[col].std() > 0 and g.radius.std() > 0:
                cs.append(np.corrcoef(g.radius, g[col])[0, 1])
        return np.array(cs)

    cp = per_cell_corr("periph_score")
    ci = per_cell_corr("interior_score")
    n = Z.cell.nunique()

    # --- is the peripheral-most zone a lamina/heterochromatin zone?
    periph_zone_is_hetero = []
    for _, g in Z.groupby("cell"):
        outer = g.loc[g.radius.idxmax()]
        periph_zone_is_hetero.append(outer.periph_score > outer.interior_score)
    frac_periph_hetero = float(np.mean(periph_zone_is_hetero))

    print(f"Nuclei certified: {n}   (zones per nucleus: {len(Z)//n})")
    print("\n1) Radial architecture (per-nucleus Pearson r across its 5 zones):")
    print(f"   radius vs PERIPHERAL-mark score : mean r = {cp.mean():+.3f} "
          f"(sd {cp.std():.3f}); positive in {int((cp>0).sum())}/{len(cp)} nuclei")
    print(f"   radius vs INTERIOR-mark score   : mean r = {ci.mean():+.3f} "
          f"(sd {ci.std():.3f}); negative in {int((ci<0).sum())}/{len(ci)} nuclei")
    print(f"\n2) Peripheral-most zone is a lamina/heterochromatin zone in "
          f"{int(frac_periph_hetero*len(periph_zone_is_hetero))}/"
          f"{len(periph_zone_is_hetero)} nuclei ({frac_periph_hetero*100:.0f}%).")

    # pooled correlation (all zones, all nuclei)
    r_pool = np.corrcoef(Z.radius, Z.periph_score)[0, 1]
    print(f"\n3) Pooled (all {len(Z)} zones): radius vs peripheral score "
          f"r = {r_pool:+.3f}")

    pd.DataFrame([{"n_nuclei": n,
                   "mean_r_radius_periph": round(cp.mean(), 3),
                   "sd_r_radius_periph": round(cp.std(), 3),
                   "n_positive_periph": int((cp > 0).sum()),
                   "mean_r_radius_interior": round(ci.mean(), 3),
                   "n_negative_interior": int((ci < 0).sum()),
                   "frac_periphmost_is_hetero": round(frac_periph_hetero, 3),
                   "pooled_r_radius_periph": round(r_pool, 3)}]
                 ).to_csv(os.path.join(a.output_dir,
                          "certification_summary.csv"), index=False)

    _plot(Z, cp, ci, a.output_dir)
    print(f"\n-> {a.output_dir}/ (certification_summary.csv, "
          "certification_zone_table.csv, certification_if.png)")


def _plot(Z, cp, ci, out_dir):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(Z.radius, Z.periph_score, s=14, alpha=0.5, color="firebrick",
                  label="peripheral marks")
    ax[0].scatter(Z.radius, Z.interior_score, s=14, alpha=0.5, color="steelblue",
                  label="interior marks")
    ax[0].set_xlabel("zone mean normalized radius (0 = center, 1 = surface)")
    ax[0].set_ylabel("zone mean mark z-score")
    ax[0].set_title(f"All zones, all nuclei (n={Z.cell.nunique()})")
    ax[0].legend()
    ax[1].hist(cp, bins=np.linspace(-1, 1, 21), alpha=0.7, color="firebrick",
               label="radius vs peripheral")
    ax[1].hist(ci, bins=np.linspace(-1, 1, 21), alpha=0.7, color="steelblue",
               label="radius vs interior")
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("per-nucleus Pearson r"); ax[1].set_ylabel("nuclei")
    ax[1].set_title("Radial organization is reproducible")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "certification_if.png"), dpi=400)
    plt.close(fig)


if __name__ == "__main__":
    main()
