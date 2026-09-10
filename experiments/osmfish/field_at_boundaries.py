"""Is the smoothed disagreement field elevated at the annotated boundaries?

Maps the (registered) cell annotation onto the grid by nearest cell, splits
the spatial-graph edges into within-domain and interface edges, and reports
the mean smoothed field on each, overall and per domain pair. On osmFISH the
interface mean is 0.72 x the within-domain mean (5 Sep 2026): the field is
highest inside the pia, ventricle and white matter, not at the layer
interfaces (manuscript Section 2.8).

    python -m experiments.osmfish.field_at_boundaries
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from experiments.osmfish import run_osmFISH as osm
from experiments.osmfish.shape_variants import state


def main():
    st = state()
    grid, e, b = st["grid"], st["edges"], st["bsm"]
    t = osm.registration_translation()
    if t is None:
        raise SystemExit("run python -m experiments.osmfish.register first")
    cells = st["cells"] + t
    gt = st["gt"]
    ok = np.asarray(pd.notna(gt))
    names = sorted(set(gt[ok]))
    gi = {n: i for i, n in enumerate(names)}
    gti = np.array([gi[x] if isinstance(x, str) else -1 for x in gt])
    d, ci = cKDTree(cells[ok]).query(grid, k=1)
    glab = gti[ok][ci]
    glab[d > np.percentile(d, 90)] = -1          # far from any cell: unlabeled
    lu, lv = glab[e[:, 0]], glab[e[:, 1]]
    eb = 0.5 * (b[e[:, 0]] + b[e[:, 1]])
    same = (lu == lv) & (lu >= 0)
    diff = (lu != lv) & (lu >= 0) & (lv >= 0)
    print(f"within-domain edges: mean field {eb[same].mean():.2f}   "
          f"interface edges: {eb[diff].mean():.2f}   ratio {eb[diff].mean()/eb[same].mean():.2f}")
    print("\nper-domain interior mean field:")
    for i, n in enumerate(names):
        print(f"  {n:32s} {b[glab == i].mean():.2f}")
    print("\ninterface mean / mean of the two interiors (pairs with >= 100 edges):")
    rows = []
    for i, a in enumerate(names):
        for j, c in enumerate(names):
            if j <= i:
                continue
            m = diff & (((lu == i) & (lv == j)) | ((lu == j) & (lv == i)))
            if m.sum() >= 100:
                inter = 0.5 * (b[glab == i].mean() + b[glab == j].mean())
                rows.append((eb[m].mean() / inter, a, c, int(m.sum())))
    for r, a, c, n in sorted(rows, reverse=True):
        print(f"  {a:26s} | {c:26s} {r:.2f}  ({n} edges)")


if __name__ == "__main__":
    main()
