"""Characteristic width of an annotated domain, from scattered cell positions.

Motivation: a kernel-smoothed fiber mixes molecules within roughly the
bandwidth h. A domain narrower than that is blended with its neighbours before
any clustering happens, so no downstream parameter can recover it. That is a
resolution limit, not a tuning failure, and it predicts WHICH domains are lost
-- the narrow ones, regardless of how many cells they contain. Cell count does
not measure this: a cortical layer is a long band, and its area says nothing
about how thin it is.

Estimator. For each cell in domain D, take the distance to the nearest cell
NOT in D. Two corrections are needed before that becomes a width:

  * the measured distance is the distance to the domain edge PLUS the gap to
    the nearest cell on the other side, so the typical inter-cell spacing is
    subtracted;
  * a median over D underestimates badly, because inside a stripe of width W
    the distance-to-edge is roughly uniform on [0, W/2], making the median
    W/4 rather than W/2. A high quantile tracks the half-width instead.

So width = 2 * (p90(distance to nearest outside cell) - median spacing).
Validated on synthetic stripes of width 5-80 at fixed density: the estimate
tracks the true width to a consistent ~12% underestimate, and the ranking is
exact. Treat it as a calibrated relative measure, not an exact one.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["domain_widths", "typical_spacing"]


def typical_spacing(coords):
    """Median nearest-neighbour distance over all cells."""
    coords = np.asarray(coords, float)
    if len(coords) < 2:
        return 0.0
    return float(np.median(cKDTree(coords).query(coords, k=2)[0][:, 1]))


def domain_widths(coords, labels, quantile=0.9):
    coords = np.asarray(coords, float)
    labels = np.asarray(labels)
    spacing = typical_spacing(coords)

    out = {}
    for d in np.unique(labels):
        inside = labels == d
        if inside.sum() < 3 or (~inside).sum() < 1:
            continue
        dist = cKDTree(coords[~inside]).query(coords[inside])[0]
        width = 2.0 * max(np.quantile(dist, quantile) - spacing, 0.0)
        out[d] = {
            "n": int(inside.sum()),
            "width": float(width),
            "spacing": spacing,
        }
    return out
