"""Does the same annotated domain split, in the same way, on every slice?

Reads the ``k_split_<slice>.npz`` files written by ``k_split.py``. For each
slice and k every cluster is assigned to the annotated domain holding the
majority of its annotated grid points; an annotated domain with two or more
such clusters is *divided*. For every division the two largest children are
compared, restricted to the annotated domain: their share of it, their purity
(fraction of the cluster inside the domain), the offset between their
centroids, and the difference of every normalised fiber component between
them, with a per-gene Mann-Whitney test (Benjamini-Hochberg, q < 0.01). A
division whose halves lie more than ``HEMI`` apart in x on a 1.8 mm bilateral
section is a hemispheric division, flagged and set aside.

Consistency across sections is the correlation, over all genes, between the
difference vector of one slice and that of another for the same domain; it is
reported for every pair of slices on which the domain divides, and, for the
same slice, across k.

Writes ``k_split_events.csv`` (one row per division) and ``k_split_pairs.csv``
(one row per slice pair per domain per k) under the results root.

    python -m experiments.merfish.k_split_analysis
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from spsheaf import paths

SLICES = ("0.04", "0.09", "0.14", "0.19", "0.24")
KS = (8, 9, 10)
HEMI = 800.0          # um; the section is 1.8 mm wide
N_TOP = 6


def children(labels, gt, valid):
    """{annotated domain: [(cluster, n_in_domain, purity), ...]} by majority."""
    out = {}
    for c in np.unique(labels):
        m = (labels == c) & valid
        if m.sum() == 0:
            continue
        dom, cnt = np.unique(gt[m], return_counts=True)
        j = cnt.argmax()
        out.setdefault(int(dom[j]), []).append((int(c), int(cnt[j]), cnt[j] / m.sum()))
    return {d: sorted(v, key=lambda t: -t[1]) for d, v in out.items()}


def deg(F, a, b, genes, n_top=N_TOP):
    diff = F[a].mean(0) - F[b].mean(0)
    p = np.array([mannwhitneyu(F[a, g], F[b, g]).pvalue for g in range(F.shape[1])])
    order = np.argsort(p); q = np.empty_like(p)
    q[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    sig = q < 0.01
    up = [genes[g] for g in np.argsort(-diff) if sig[g]][:n_top]
    down = [genes[g] for g in np.argsort(diff) if sig[g]][:n_top]
    return diff, int(sig.sum()), up, down


def main():
    events, vec = [], {}
    for s in SLICES:
        p = paths.results("merfish") / f"k_split_MERFISH_{s}.npz"
        if not p.exists():
            print(f"  (missing {p.name})"); continue
        z = np.load(p, allow_pickle=False)
        xy, F, gt = z["grid_coords"], z["fibers_norm"], z["gt_int"]
        names, genes = list(z["gt_names"]), list(z["gene_names"])
        valid = gt >= 0
        for k in KS:
            lab = z[f"labels_k{k}"]
            ch = children(lab, gt, valid)
            for d, kids in ch.items():
                if len(kids) < 2:
                    continue
                n_dom = int(((gt == d) & valid).sum())
                (c1, n1, p1), (c2, n2, p2) = kids[:2]
                m1 = (lab == c1) & (gt == d) & valid
                m2 = (lab == c2) & (gt == d) & valid
                dx, dy = xy[m1].mean(0) - xy[m2].mean(0)
                diff, nsig, up, down = deg(F, m1, m2, genes)
                hemi = abs(dx) > HEMI
                vec[(s, k, names[d])] = (diff, hemi)
                events.append(dict(
                    slice=s, k=k, domain=names[d], n_domain=n_dom,
                    n_children=len(kids), share_1=n1 / n_dom, share_2=n2 / n_dom,
                    purity_1=p1, purity_2=p2, offset_x=dx, offset_y=dy,
                    hemispheric=hemi, n_deg=nsig,
                    up_in_larger=" ".join(up), up_in_smaller=" ".join(down)))
    ev = pd.DataFrame(events)

    pairs = []
    for k in KS:
        for dom in sorted(ev.domain.unique()):
            keys = [(s, kk, d) for (s, kk, d) in vec if d == dom and kk == k]
            for a, b in itertools.combinations(sorted(keys), 2):
                r = float(np.corrcoef(vec[a][0], vec[b][0])[0, 1])
                pairs.append(dict(k=k, domain=dom, slice_a=a[0], slice_b=b[0], r=r,
                                  hemispheric=bool(vec[a][1] or vec[b][1])))
    pr = pd.DataFrame(pairs)

    # best non-hemispheric partner for every event, for the table
    def best_r(row):
        q = pr[(pr.k == row.k) & (pr.domain == row.domain) & ~pr.hemispheric
               & ((pr.slice_a == row.slice) | (pr.slice_b == row.slice))]
        if row.hemispheric or q.empty:
            return np.nan, ""
        j = q.r.idxmax()
        other = q.loc[j, "slice_b"] if q.loc[j, "slice_a"] == row.slice else q.loc[j, "slice_a"]
        return float(q.loc[j, "r"]), other
    ev[["r_best", "r_best_with"]] = ev.apply(lambda r: pd.Series(best_r(r)), axis=1)

    across_k = []
    for (s, k, d), (v, h) in vec.items():
        for k2 in KS:
            if k2 > k and (s, k2, d) in vec:
                across_k.append(dict(slice=s, domain=d, k_a=k, k_b=k2,
                                     r=float(np.corrcoef(v, vec[(s, k2, d)][0])[0, 1])))
    ak = pd.DataFrame(across_k)

    root = paths.results("merfish")
    ev.to_csv(root / "k_split_events.csv", index=False)
    pr.to_csv(root / "k_split_pairs.csv", index=False)
    ak.to_csv(root / "k_split_across_k.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"{len(ev)} divisions over {len(SLICES)} slices x k = {KS}; "
          f"{int(ev.hemispheric.sum())} hemispheric")
    print(ev[["slice", "k", "domain", "share_1", "share_2", "purity_1", "purity_2",
              "offset_x", "offset_y", "hemispheric", "n_deg", "r_best", "r_best_with"]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nnon-hemispheric slice pairs, r of the difference vectors:")
    print(pr[~pr.hemispheric].to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
    print("\nsame slice, across k:")
    print(ak.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))


if __name__ == "__main__":
    main()
