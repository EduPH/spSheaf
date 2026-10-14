"""Does the EMT score step or ramp across the mesenchymal compartment's edge?

The PECAb section is a continuum with one discrete compartment (the
mesenchymal / myofibroblast domain). This asks whether the EMT score changes
discontinuously at that compartment's inferred edge, as it would at an
interface between two cell populations, or continuously across it, as along
a spatial state transition, and whether the disagreement field is elevated
there. For every grid point the signed distance to the nearest grid point of
the other side of the edge (negative inside the compartment) is binned in
grid spacings; the median and quartiles of the EMT score and of the smoothed
field are recorded per bin, with a logistic fit of the EMT profile (width in
grid spacings) and the share of the total EMT change that falls within
+-1.5 spacings of the edge. Both replicates.

    python -m experiments.pecab.edge_profiles

Writes data/results/pecab/edge_profiles.csv and edge_summary.csv.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from spsheaf import paths

REPS = ("Rep1", "Rep2")
NBINS = 8                       # +-8 grid spacings


def profile(rep):
    base = paths.results("pecab") / f"output_pecab_signaling_{rep}" / "panels" / "pecab"
    d = np.load(base.with_suffix(".npz")); m = json.loads(base.with_suffix(".json").read_text())
    g, lab, B, E = d["grid"], d["labels"], d["boundary_smooth"], d["emt_score"]
    step = float(np.min(np.diff(np.unique(g[:, 0]))))
    inside = lab == int(m["mes_domain"])
    dist = np.where(inside, -cKDTree(g[~inside]).query(g)[0], cKDTree(g[inside]).query(g)[0])
    edges = np.arange(-NBINS, NBINS + 1) * step
    rows = []
    for i in range(len(edges) - 1):
        s = (dist >= edges[i]) & (dist < edges[i + 1])
        if s.sum() < 20:
            continue
        q = lambda v: np.percentile(v[s], [25, 50, 75])
        qe, qb = q(E), q(B)
        rows.append(dict(rep=rep, bin_lo=edges[i] / step, bin_hi=edges[i + 1] / step, n=int(s.sum()),
                         emt_q25=qe[0], emt_med=qe[1], emt_q75=qe[2],
                         field_q25=qb[0], field_med=qb[1], field_q75=qb[2]))
    p = pd.DataFrame(rows)
    sel = np.abs(dist) < NBINS * step
    f = lambda x, a, b, x0, w: a + b / (1 + np.exp(-(x - x0) / w))
    fit, _ = curve_fit(f, dist[sel] / step, E[sel], p0=[E[inside].mean(), E[~inside].mean() - E[inside].mean(), 0, 1],
                       maxfev=20000)
    far_in = p[p.bin_hi <= -4].emt_med.mean(); far_out = p[p.bin_lo >= 4].emt_med.mean()
    near_in = np.median(E[(dist < 0) & (dist >= -1.5 * step)]); near_out = np.median(E[(dist > 0) & (dist <= 1.5 * step)])
    interior = np.median(B[np.abs(dist) > 3 * step]); at_edge = np.median(B[np.abs(dist) <= 1.0 * step])
    summ = dict(rep=rep, step_px=step, bandwidth_px=m["bandwidth"], mes_domain=int(m["mes_domain"]),
                n_inside=int(inside.sum()), emt_inside=float(E[inside].mean()), emt_outside=float(E[~inside].mean()),
                emt_far_in=far_in, emt_far_out=far_out, emt_change_total=far_out - far_in,
                emt_change_within_1p5=near_out - near_in,
                share_within_1p5=(near_out - near_in) / (far_out - far_in),
                logistic_centre_steps=fit[2], logistic_width_steps=abs(fit[3]),
                field_edge_over_interior=at_edge / interior,
                emt_sd_inside=float(E[inside].std()), emt_sd_outside=float(E[~inside].std()))
    return p, summ


def main():
    out = paths.results("pecab")
    profs, summs = [], []
    for rep in REPS:
        p, s = profile(rep); profs.append(p); summs.append(s)
    pd.concat(profs).to_csv(out / "edge_profiles.csv", index=False)
    S = pd.DataFrame(summs); S.to_csv(out / "edge_summary.csv", index=False)
    print(S.T.to_string())


if __name__ == "__main__":
    main()
