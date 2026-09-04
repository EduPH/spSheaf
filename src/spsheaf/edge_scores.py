"""Alternative edge scores, for isolating what the sheaf construction adds.

Answers reviewer memo point 2: the existing ablation shows the *pipeline*
helps, not that the *sheaf* does. A skeptical reader can describe the method
as "local PCA plus SVD alignment plus boundary-weighted k-means", so the
honest test is to swap only the edge score and hold everything else fixed.

Every function here takes the same arguments and returns one non-negative
value per edge, so they are drop-in substitutes for one another:

    sheaf        the method: || F_u xi_u - F_v xi_v ||^2, the residual
                 disagreement of local coordinates after optimal alignment
    grassmann    principal angles only: sqrt(sum theta_i^2). Depends on the
                 subspaces alone and ignores where the data sit inside them
    projection   || U_u U_u^T - U_v U_v^T ||_F, the subspace/covariance
                 distance -- a non-sheaf local-geometry baseline
    procrustes   min_Q || U_u - U_v Q ||_F^2, the alignment residual with no
                 sheaf interpretation
    identity     || xi_u - xi_v ||^2 with identity restriction maps: the
                 control that shows whether the alignment step does anything
    neighbour    || f_u - f_v ||^2, the plain kernel baseline
    sheaf_local  the method, but with xi referenced to the LOCAL neighbourhood
                 mean instead of the global mean (the centering ablation)

If `sheaf` does not separate from `grassmann`, `projection` and `procrustes`,
the sheaf framing is decoration. That is the question this module exists to
answer, and it is worth knowing the answer before a referee asks.
"""
from __future__ import annotations

import numpy as np

__all__ = ["EDGE_SCORES", "compute_edge_score"]


def _svd_overlap(U_u, U_v):
    M = U_u.T @ U_v
    try:
        L, S, Rt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:                       # pragma: no cover
        d = M.shape[0]
        L, S, Rt = np.eye(d), np.ones(d), np.eye(d)
    return L, np.clip(S, -1.0, 1.0), Rt


def _xi(U, f):
    return U.T @ f


def compute_edge_score(kind, stalks, edges, fib_centered,
                       fib_local_centered=None):
    """One score per edge under the named construction."""
    if kind not in EDGE_SCORES:
        raise KeyError(f"unknown edge score {kind!r}; have {sorted(EDGE_SCORES)}")
    fn = EDGE_SCORES[kind]
    out = np.zeros(len(edges))
    for ei, (u, v) in enumerate(edges):
        out[ei] = fn(stalks[u], stalks[v],
                     fib_centered[u], fib_centered[v],
                     None if fib_local_centered is None else fib_local_centered[u],
                     None if fib_local_centered is None else fib_local_centered[v])
    return out


def _sheaf(U_u, U_v, f_u, f_v, fl_u, fl_v):
    L, S, Rt = _svd_overlap(U_u, U_v)
    d = L.T @ _xi(U_u, f_u) - Rt @ _xi(U_v, f_v)
    return float(d @ d)


def _sheaf_local(U_u, U_v, f_u, f_v, fl_u, fl_v):
    if fl_u is None:
        raise ValueError("sheaf_local needs locally centered fibers")
    L, S, Rt = _svd_overlap(U_u, U_v)
    d = L.T @ _xi(U_u, fl_u) - Rt @ _xi(U_v, fl_v)
    return float(d @ d)


def _grassmann(U_u, U_v, *_):
    _, S, _ = _svd_overlap(U_u, U_v)
    theta = np.arccos(np.clip(S, 0.0, 1.0))
    return float(np.sqrt(np.sum(theta ** 2)))


def _projection(U_u, U_v, *_):
    _, S, _ = _svd_overlap(U_u, U_v)
    # || U_u U_u^T - U_v U_v^T ||_F^2 = 2 * sum sin^2(theta_i)
    return float(np.sqrt(max(2.0 * np.sum(1.0 - S ** 2), 0.0)))


def _procrustes(U_u, U_v, *_):
    d = U_u.shape[1]
    _, S, _ = _svd_overlap(U_u, U_v)
    return float(max(2.0 * d - 2.0 * np.sum(S), 0.0))


def _identity(U_u, U_v, f_u, f_v, *_):
    d = _xi(U_u, f_u) - _xi(U_v, f_v)
    return float(d @ d)


def _neighbour(U_u, U_v, f_u, f_v, *_):
    d = f_u - f_v
    return float(d @ d)


EDGE_SCORES = {
    "sheaf": _sheaf,
    "sheaf_local": _sheaf_local,
    "grassmann": _grassmann,
    "projection": _projection,
    "procrustes": _procrustes,
    "identity": _identity,
    "neighbour": _neighbour,
}
