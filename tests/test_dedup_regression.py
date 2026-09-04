"""all_metrics must keep working after the metric deduplication."""
import numpy as np

from benchmarks.segfree import common as bench


def test_all_metrics_shape_and_values():
    rng = np.random.default_rng(1)
    coords = np.vstack([rng.normal(c, .6, (50, 2)) for c in [(0, 0), (8, 0), (0, 8)]])
    gt = np.repeat([0, 1, 2], 50)
    pred = gt.copy()
    pred[:5] = 2                                     # a few deliberate errors
    m = bench.all_metrics(gt, pred, coords)
    assert set(m) == {"ARI", "NMI", "HOM", "COM",
                      "CHAOS", "PAS", "ASW", "n_pred_domains"}
    assert 0.8 < m["ARI"] <= 1.0
    assert m["n_pred_domains"] == 3
    assert np.isfinite([m["CHAOS"], m["PAS"], m["ASW"]]).all()
