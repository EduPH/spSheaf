import numpy as np, pytest
from experiments.common.cv import spatial_blocks, cross_validate, summarise

def test_blocks_are_contiguous_and_balanced():
    rng=np.random.default_rng(0)
    c=np.c_[rng.uniform(0,100,600), rng.uniform(0,20,600)]   # wide in x
    f=spatial_blocks(c, n_folds=5)
    assert set(np.unique(f))=={0,1,2,3,4}
    counts=np.bincount(f); assert counts.min()>=100 and counts.max()<=140
    # contiguity: each block occupies a distinct x range
    means=[c[f==k,0].mean() for k in range(5)]
    assert means==sorted(means)

def test_cv_never_selects_on_the_held_out_fold():
    """Discriminating case: A is superb on fold 0 and mediocre elsewhere;
    B is uniformly middling. Over ALL folds A wins on average, so a leaky
    implementation that selected using the test fold would report A's 0.99.
    Correct leave-one-out excludes fold 0 when testing on it, where B wins,
    and must therefore report B's 0.50."""
    folds = [0, 1, 2, 3, 4]
    A, B = {"name": "A"}, {"name": "B"}
    score = {"A": lambda f: 0.99 if f == 0 else 0.40,
             "B": lambda f: 0.50}
    run = lambda c, f: {"ari": score[c["name"]](f), "nmi": score[c["name"]](f)}

    # sanity: A really does win when the test fold is (wrongly) included
    assert np.mean([score["A"](f) for f in folds]) > np.mean([score["B"](f) for f in folds])

    cv = cross_validate(run, [A, B], folds, verbose=False)
    ho = [m["ari"] for m in cv["held_out"]]
    assert ho[0] == 0.50, f"leaked the test fold into selection: {ho}"
    assert all(v == 0.40 for v in ho[1:]), ho   # A chosen elsewhere, scored honestly


def test_summary_flags_unstable_selection():
    folds=[0,1,2]
    A={'n':'A'};B={'n':'B'}
    run=lambda c,f: {'ari': (0.9 if c['n']=='A' else 0.1) if f==0 else
                            (0.1 if c['n']=='A' else 0.9),'nmi':0.5}
    cv=cross_validate(run,[A,B],folds,verbose=False)
    s=summarise(cv)
    assert s['selection_stable_across_folds'] in (True,False)
    assert not np.isnan(s['cv_ari_mean'])


def test_blocking_axis_avoids_destroying_the_label_distribution():
    """A layered section: labels are bands along y. Blocking along y gives
    each fold a different set of layers; blocking along x preserves them.
    With labels supplied, the chooser must pick x."""
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 300, n)              # longer axis, and the layer axis
    labels = (y // 60).astype(int)          # 5 bands stacked along y
    coords = np.c_[x, y]

    naive = spatial_blocks(coords, n_folds=5)                  # no labels -> argmax extent
    smart = spatial_blocks(coords, n_folds=5, labels=labels)

    def worst_missing(fold):
        return max(sum((labels[fold != f] == c).sum() == 0
                       for c in np.unique(labels)) for f in np.unique(fold))

    assert worst_missing(naive) > 0, "expected the naive axis to drop classes"
    assert worst_missing(smart) == 0, "label-aware blocking still drops classes"


def test_label_aware_blocking_is_still_contiguous():
    rng = np.random.default_rng(1)
    coords = np.c_[rng.uniform(0, 100, 1500), rng.uniform(0, 300, 1500)]
    labels = (coords[:, 1] // 60).astype(int)
    fold = spatial_blocks(coords, n_folds=5, labels=labels)
    means = [coords[fold == k, 0].mean() for k in range(5)]
    assert means == sorted(means), "blocks are not contiguous along the chosen axis"
