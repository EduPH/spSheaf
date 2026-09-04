"""Parity between the two SDMBench metric implementations.

The repository currently carries two: ``spsheaf.evaluation`` (used for the
manuscript's own numbers) and ``benchmarks.segfree.common`` (used to score the
comparators, inherited from the collaborator). Both claim to implement the
SDMBench definitions of Yuan et al. 2024.

Until these agree, no cross-method comparison in the manuscript is safe: a
difference here would mean spSheaf and its competitors were scored on
different rulers. This module is the gate that must pass before either
implementation is deleted.
"""

import numpy as np
import pytest

from spsheaf import evaluation as pkg
from benchmarks.segfree import common as bench


def _cases():
    rng = np.random.default_rng(0)
    out = {}

    # well-separated blobs
    n = 60
    coords, labels = [], []
    for k, c in enumerate([(0, 0), (10, 0), (0, 10), (10, 10)]):
        coords.append(rng.normal(c, 0.6, size=(n, 2)))
        labels.append(np.full(n, k))
    out["separated_blobs"] = (np.vstack(coords), np.concatenate(labels))

    # fully interleaved — labels carry no spatial structure
    c = rng.uniform(0, 10, size=(240, 2))
    out["interleaved"] = (c, rng.integers(0, 4, size=240))

    # elongated stripes
    c = rng.uniform(0, 10, size=(240, 2))
    out["stripes"] = (c, (c[:, 0] // 2.5).astype(int))

    # a singleton cluster alongside two real ones
    c = np.vstack([rng.normal((0, 0), .5, (80, 2)),
                   rng.normal((6, 6), .5, (80, 2)),
                   np.array([[20.0, 20.0]])])
    l = np.concatenate([np.zeros(80), np.ones(80), [2]]).astype(int)
    out["singleton_cluster"] = (c, l)

    # a two-point cluster: exercises the small-cluster cutoff
    c = np.vstack([rng.normal((0, 0), .5, (80, 2)),
                   rng.normal((6, 6), .5, (80, 2)),
                   np.array([[20.0, 20.0], [20.1, 20.1]])])
    l = np.concatenate([np.zeros(80), np.ones(80), [2, 2]]).astype(int)
    out["two_point_cluster"] = (c, l)

    return out


CASES = _cases()


@pytest.mark.parametrize("name", list(CASES))
def test_chaos_parity(name):
    coords, labels = CASES[name]
    assert pkg.compute_chaos(labels, coords) == pytest.approx(
        bench.compute_CHAOS(labels, coords), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("k", [6, 10, 20])
def test_pas_parity(name, k):
    coords, labels = CASES[name]
    assert pkg.compute_pas(labels, coords, k=k) == pytest.approx(
        bench.compute_PAS(labels, coords, k=k), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("name", list(CASES))
def test_asw_parity(name):
    coords, labels = CASES[name]
    a = pkg.compute_asw(labels, coords)
    b = bench.compute_ASW(labels, coords)
    if np.isnan(a) and np.isnan(b):
        return
    assert a == pytest.approx(b, rel=1e-9, abs=1e-12)
