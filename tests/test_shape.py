import numpy as np, pytest
from experiments.common.shape import domain_shapes

def _band(L, W, n=4000, seed=0, angle=0.0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform([-L/2, -W/2], [L/2, W/2], size=(n, 2))
    c, s = np.cos(angle), np.sin(angle)
    return xy @ np.array([[c, -s], [s, c]])

@pytest.mark.parametrize("L,W,expect", [(100,100,1.0), (400,100,4.0), (1000,50,20.0)])
def test_elongation_matches_aspect_ratio(L, W, expect):
    xy = _band(L, W)
    got = domain_shapes(xy, np.zeros(len(xy)))[0.0]["elongation"]
    assert 0.75*expect < got < 1.35*expect, f"{L}x{W} -> {got:.2f}"

def test_elongation_is_rotation_invariant():
    a = domain_shapes(*(lambda x: (x, np.zeros(len(x))))(_band(600,60,angle=0.0)))[0.0]
    b = domain_shapes(*(lambda x: (x, np.zeros(len(x))))(_band(600,60,angle=0.7)))[0.0]
    assert abs(a["elongation"] - b["elongation"]) < 0.15 * a["elongation"]

def test_slenderness_tracks_length_over_width():
    xy = _band(800, 40)
    w = {0.0: {"width": 40.0}}
    s = domain_shapes(xy, np.zeros(len(xy)), widths=w)[0.0]["slenderness"]
    assert 15 < s < 25, s          # area/W^2 = (800*40)/1600 = 20

def test_elongation_fails_on_a_curved_domain_but_slenderness_does_not():
    """Documents the known limitation: a C-shape has principal axes that do not
    follow its arc, so elongation reads near 1 while it is plainly slender."""
    t = np.linspace(0, np.pi, 3000)
    rng = np.random.default_rng(0)
    r = 200 + rng.uniform(-15, 15, len(t))
    xy = np.c_[r*np.cos(t), r*np.sin(t)]
    sh = domain_shapes(xy, np.zeros(len(xy)), widths={0.0: {"width": 30.0}})[0.0]
    assert sh["elongation"] < 3.0, sh["elongation"]
    assert sh["slenderness"] > 5.0, sh["slenderness"]
