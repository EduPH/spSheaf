import numpy as np, pytest
from experiments.common.width import domain_widths, typical_spacing

def _stripe(W, L=400.0, density=0.125, seed=0):
    """Non-overlapping construction: every cell carries exactly one label,
    as in a real annotation. An earlier version of this test drew the domain
    ON TOP of the background, so cells overlapped and every domain measured
    the same width -- the inter-cell spacing."""
    rng = np.random.default_rng(seed)
    # n scales with AREA so that changing L changes the domain's length at
    # constant density. Fixing n instead changes density, which was a bug in
    # an earlier version of this test and made the length check meaningless.
    n = int(density * 400.0 * L)
    pts = rng.uniform([-200, 0], [200, L], size=(n, 2))
    lab = np.where(np.abs(pts[:, 0]) < W / 2, "d", "bg")
    return pts, lab

@pytest.mark.parametrize("W", [5.0, 10.0, 20.0, 40.0, 80.0])
def test_tracks_true_width(W):
    pts, lab = _stripe(W)
    got = domain_widths(pts, lab)["d"]["width"]
    assert 0.75 * W < got < 1.25 * W, f"W={W} -> {got:.1f}"

def test_ranking_is_exact():
    widths = [domain_widths(*_stripe(W))["d"]["width"] for W in (5, 10, 20, 40, 80)]
    assert widths == sorted(widths), widths

def test_width_is_independent_of_domain_length():
    """The whole point: two domains with very different cell counts can have
    the same width, and area must not stand in for width."""
    a = domain_widths(*_stripe(10.0, L=200.0))["d"]
    b = domain_widths(*_stripe(10.0, L=800.0))["d"]
    # same width, same density, 4x the length -> ~4x the cells
    assert b["n"] > 1.8 * a["n"], (a["n"], b["n"])          # much larger domain
    assert abs(a["width"] - b["width"]) < 0.35 * a["width"]  # same width

def test_disconnected_domain():
    rng = np.random.default_rng(2)
    pts = rng.uniform([-100, 0], [200, 200], size=(20000, 2))
    inside = (np.abs(pts[:, 0] - 0) < 5) | (np.abs(pts[:, 0] - 100) < 5)
    lab = np.where(inside, "d", "bg")
    w = domain_widths(pts, lab)["d"]["width"]
    assert 6 < w < 14, w
