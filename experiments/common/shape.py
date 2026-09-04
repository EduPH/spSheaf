"""Shape descriptors for annotated domains: elongation, not just width.

An earlier analysis measured each domain's WIDTH and found it explained little.
That measured absolute thinness. It does not capture *elongation* -- a domain
can be 200 um wide and still be a long thin band, which is what a cortical
layer is.

Elongation matters for a specific, mechanical reason. The clustering step is
k-means on [weighted expression | alpha_s * xy]. k-means minimises within-
cluster variance, so it prefers compact, roughly spherical clusters. A long
thin band has large spatial variance however coherent its expression, so the
spatial term penalises precisely the shape a layered cortex is made of. The
prediction is that elongated domains are recovered worse, and that the penalty
weakens as alpha_s falls.

Two descriptors, because they fail differently:

    elongation  sqrt(lambda1 / lambda2) from PCA of the domain's cell
                coordinates. Scale-free. Wrong for curved or C-shaped domains,
                whose principal axes do not follow the shape.
    slenderness area / width^2, a length-to-width proxy that survives curvature
                because it never assumes a straight axis.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

__all__ = ["domain_shapes"]


def _elongation(xy):
    if len(xy) < 3:
        return np.nan
    c = xy - xy.mean(0)
    ev = np.linalg.eigvalsh(np.cov(c.T))
    ev = np.clip(ev, 1e-12, None)
    return float(np.sqrt(ev[-1] / ev[0]))


def _area(xy):
    """Convex-hull area. Overestimates for curved or disconnected domains, so
    slenderness derived from it is a lower bound on how slender they are."""
    if len(xy) < 3:
        return np.nan
    try:
        return float(ConvexHull(xy).volume)   # 'volume' is area in 2D
    except Exception:
        return np.nan


def domain_shapes(coords, labels, widths=None):
    coords = np.asarray(coords, float)
    labels = np.asarray(labels)
    out = {}
    for d in np.unique(labels):
        xy = coords[labels == d]
        rec = {"n": int(len(xy)), "elongation": _elongation(xy),
               "area": _area(xy)}
        if widths is not None and d in widths:
            w = widths[d]["width"] if isinstance(widths[d], dict) else widths[d]
            rec["width"] = float(w)
            rec["slenderness"] = (float(rec["area"] / w ** 2)
                                  if w > 0 and np.isfinite(rec["area"])
                                  else np.nan)
        out[d] = rec
    return out
