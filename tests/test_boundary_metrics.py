"""Sanity: a field that IS the boundary must beat noise, and metrics must be sane."""
import numpy as np, pytest
from spsheaf.boundary_metrics import (ground_truth_boundary, evaluate_field,
                              pr_curve, best_f1, average_precision)

def setup():
    rng = np.random.default_rng(0)
    g = 40
    xs, ys = np.meshgrid(np.arange(g)*15.0, np.arange(g)*15.0)
    coords = np.c_[xs.ravel(), ys.ravel()]
    gt = (coords[:,0] // 300).astype(int)              # 2 vertical domains
    # 4-neighbour graph
    idx = np.arange(g*g).reshape(g,g)
    edges = [(int(idx[i,j]), int(idx[i,j+1])) for i in range(g) for j in range(g-1)]
    edges += [(int(idx[i,j]), int(idx[i+1,j])) for i in range(g-1) for j in range(g)]
    return coords, gt, edges, rng

def test_perfect_field_beats_noise():
    coords, gt, edges, rng = setup()
    b = ground_truth_boundary(gt, edges)
    assert 0 < b.mean() < 0.2
    perfect = b.astype(float) + rng.normal(0, .01, len(b))
    noise   = rng.normal(0, 1, len(b))
    rp = evaluate_field(perfect, coords, b, tols=(0.0,))
    rn = evaluate_field(noise,   coords, b, tols=(0.0,))
    assert rp["F1@0"] > 0.9, rp
    assert rp["F1@0"] > rn["F1@0"] + 0.5
    assert rp["mean_pred_to_gt"] < rn["mean_pred_to_gt"]

def test_tolerance_is_monotone():
    coords, gt, edges, rng = setup()
    b = ground_truth_boundary(gt, edges)
    # field offset by one grid step: should be poor at tol=0, good at tol=15
    shifted = np.roll(b.reshape(40,40), 1, axis=1).ravel().astype(float)
    r = evaluate_field(shifted, coords, b, tols=(0.0, 15.0))
    assert r["F1@15"] > r["F1@0"], r
    assert r["F1@15"] > 0.9

def test_graded_field_ranks_between():
    coords, gt, edges, rng = setup()
    b = ground_truth_boundary(gt, edges)
    d = np.abs(coords[:,0] - 300.0)
    graded = np.exp(-d/60.0)                # smooth bump on the true boundary
    noise  = rng.normal(0,1,len(b))
    rg = evaluate_field(graded, coords, b, tols=(15.0,))
    rn = evaluate_field(noise,  coords, b, tols=(15.0,))
    assert rg["AP@15"] > rn["AP@15"]
    assert 0.0 <= rg["AP@15"] <= 1.0 and 0.0 <= rg["F1@15"] <= 1.0
