"""Behaviour of the alternative edge scores on controlled constructions."""
import numpy as np, pytest
from spsheaf.edge_scores import EDGE_SCORES, compute_edge_score


def build(G=40, d=4, n=60, smooth=True, seed=0):
    """Two regions with ORTHOGONAL subspaces.

    With smooth=True the coefficients vary slowly along the chain within each
    region, which is what kernel-smoothed fibers actually look like: adjacent
    grid points aggregate overlapping molecule sets. With smooth=False the
    coefficients are i.i.d., which no real fiber field is.
    """
    rng = np.random.default_rng(seed)
    A = np.linalg.qr(rng.normal(size=(G, 2 * d)))[0]
    Ua, Ub = A[:, :d], A[:, d:2 * d]

    def coeffs():
        c = rng.normal(size=(n, d))
        if smooth:                       # cumulative walk -> locally similar
            c = np.cumsum(c, axis=0) / np.sqrt(np.arange(1, n + 1))[:, None]
        return c

    fib = np.vstack([coeffs() @ Ua.T, coeffs() @ Ub.T])
    fib = fib - fib.mean(0)
    stalks = [Ua] * n + [Ub] * n
    within = [(i, i + 1) for i in range(0, 20)] + [(n + i, n + i + 1) for i in range(0, 20)]
    across = [(n - 1 - i, n + i) for i in range(20)]
    return stalks, fib, within, across


@pytest.mark.parametrize("kind", ["grassmann", "projection", "procrustes"])
def test_subspace_only_scores_fire_across_boundary(kind):
    """These depend on the stalks alone, so they fire regardless of smoothness."""
    stalks, fib, within, across = build()
    w = compute_edge_score(kind, stalks, within, fib).mean()
    a = compute_edge_score(kind, stalks, across, fib).mean()
    assert a > 3 * w, f"{kind}: within={w:.4g} across={a:.4g}"


def test_sheaf_fires_when_fibers_are_locally_smooth():
    stalks, fib, within, across = build(smooth=True)
    w = compute_edge_score("sheaf", stalks, within, fib).mean()
    a = compute_edge_score("sheaf", stalks, across, fib).mean()
    assert a > 3 * w, f"within={w:.4g} across={a:.4g}"


def test_sheaf_does_not_separate_without_local_smoothness():
    """Documents a real property: the sheaf score measures residual coordinate
    disagreement, so it needs within-domain fibers to actually agree. With
    i.i.d. coefficients it carries no boundary signal, while the purely
    subspace-based scores still do. Any ablation must not conflate the two."""
    stalks, fib, within, across = build(smooth=False)
    w = compute_edge_score("sheaf", stalks, within, fib).mean()
    a = compute_edge_score("sheaf", stalks, across, fib).mean()
    assert 0.4 < a / w < 2.5, f"within={w:.4g} across={a:.4g}"


def test_neighbour_distance_is_blind_to_subspace_rotation():
    stalks, fib, within, across = build(smooth=False)
    w = compute_edge_score("neighbour", stalks, within, fib).mean()
    a = compute_edge_score("neighbour", stalks, across, fib).mean()
    assert 0.4 < a / w < 2.5, f"within={w:.4g} across={a:.4g}"


def test_all_scores_finite_and_nonnegative():
    stalks, fib, within, across = build()
    for kind in EDGE_SCORES:
        fl = fib if kind == "sheaf_local" else None
        s = compute_edge_score(kind, stalks, within + across, fib, fl)
        assert np.all(np.isfinite(s)) and np.all(s >= -1e-12), kind


def test_identical_stalks_give_near_zero_angle_scores():
    """Tolerance is 1e-6, not 1e-12: arccos is ill-conditioned near sigma = 1,
    so the Grassmann distance loses about half its digits for aligned stalks."""
    rng = np.random.default_rng(1)
    U = np.linalg.qr(rng.normal(size=(20, 3)))[0][:, :3]
    fib = rng.normal(size=(2, 20))
    for kind in ["grassmann", "projection", "procrustes"]:
        s = compute_edge_score(kind, [U, U], [(0, 1)], fib)
        assert s[0] < 1e-6, f"{kind} = {s[0]}"
