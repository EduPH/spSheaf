"""Cross-validated hyperparameter selection, matched across datasets.

The manuscript states that (sigma_s, alpha_b, alpha_s) were chosen by
leave-one-out cross-validation across the five MERFISH slices. No code
implementing that existed in this repository, so the claim was not
reproducible. This module implements it, and extends it to a single-section
dataset.

Why folds differ by dataset, and why that is not a cheat:

    MERFISH   5 slices  -> leave-one-slice-out. Exactly the published protocol.
    osmFISH   1 section -> leave-one-spatial-block-out. Leave-one-out across
                           slices is impossible with one section; contiguous
                           spatial blocks are the closest matched analogue,
                           and they preserve the property that matters: the
                           configuration is never selected on the data it is
                           scored on.

Both report a held-out score. Neither reports an argmax over the annotation,
which is the trap the existing osmFISH `sweep` mode falls into and which the
review memo (point 5) calls out.

Every configuration's annotation-free criteria are recorded alongside its
labelled scores, so a fully label-free selection rule can be applied post hoc
-- to this method and to the competitors -- without rerunning anything.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

__all__ = ["spatial_blocks", "cross_validate", "summarise"]


def spatial_blocks(coords, n_folds=5, labels=None, axis=None, verbose=False):
    """Contiguous spatial blocks that preserve the label distribution.

    Contiguous rather than random, because random folds leak: a held-out cell
    has training cells microns away, and a classifier on coordinates then does
    nearest-neighbour lookup rather than learning anything. On osmFISH that
    inflated coordinates-only ARI to 0.93.

    But contiguity alone is not enough. Blocking a layered cortex along the
    axis the layers run *across* gives each fold a different set of layers --
    several classes vanish from training entirely, recall goes to 0.00 for
    them, and the CV silently measures extrapolation to unseen classes rather
    than generalisation. So when `labels` are supplied the blocking axis is
    chosen to keep every fold's label distribution close to the global one,
    i.e. cut *along* the layers rather than across them.

    Passing `labels` is not leakage: it shapes how folds are drawn, exactly as
    stratified k-fold does, and model selection still happens only on held-out
    scores.
    """
    coords = np.asarray(coords, float)
    n = len(coords)

    def _assign(ax):
        v = coords[:, ax]
        edges = np.quantile(v, np.linspace(0, 1, n_folds + 1))
        edges[0] -= 1e-9
        return np.digitize(v, edges[1:-1]).astype(int)

    axes = [axis] if axis is not None else list(range(coords.shape[1]))

    if labels is None or axis is not None:
        chosen = axes[0] if axis is not None else int(
            np.argmax(coords.max(0) - coords.min(0)))
        return _assign(chosen)

    labels = np.asarray(labels)
    classes = np.unique(labels)
    glob = np.array([(labels == c).mean() for c in classes])

    best, best_cost, best_fold = None, np.inf, None
    for ax in axes:
        fold = _assign(ax)
        # worst total-variation distance between a fold and the global mix
        cost = 0.0
        for f in np.unique(fold):
            m = fold == f
            p = np.array([(labels[m] == c).mean() for c in classes])
            cost = max(cost, 0.5 * np.abs(p - glob).sum())
        if cost < best_cost:
            best, best_cost, best_fold = ax, cost, fold

    missing = [(f, [str(c) for c in classes
                    if (labels[best_fold != f] == c).sum() == 0])
               for f in np.unique(best_fold)]
    missing = [(f, cs) for f, cs in missing if cs]
    if verbose or missing:
        print(f"    blocking axis {best} (label-distribution cost {best_cost:.3f})")
    if missing:
        for f, cs in missing:
            print(f"    WARNING fold {f}: classes absent from training: {cs}")
        print("    Folds do not share a label set; the CV is measuring "
              "extrapolation to unseen classes, not generalisation.")
    return best_fold


def cross_validate(run_fn, configs, fold_of_unit, score_fn=None, verbose=True):
    """Leave-one-fold-out selection.

    Parameters
    ----------
    run_fn : callable(config, fold) -> dict
        Must return at least {'ari', 'nmi'} for that config restricted to that
        fold, plus any annotation-free criteria to record.
    configs : list of dict
    fold_of_unit : sequence of fold ids
        The distinct folds to iterate over (slice ids, or block ids).
    score_fn : callable(metrics) -> float
        Selection objective on the TRAINING folds. Defaults to ARI.

    Returns
    -------
    dict with per-fold held-out scores, the config chosen on each training
    split, and the config chosen using every fold (for reporting a single
    final setting).
    """
    if score_fn is None:
        score_fn = lambda m: m.get("ari", float("nan"))

    folds = list(dict.fromkeys(fold_of_unit))
    # metrics[config_index][fold] -> dict
    table = [[run_fn(c, f) for f in folds] for c in configs]

    held_out, chosen = [], []
    for i, test in enumerate(folds):
        train = [j for j, _ in enumerate(folds) if j != i]
        means = [np.nanmean([score_fn(table[c][j]) for j in train])
                 for c in range(len(configs))]
        best = int(np.nanargmax(means))
        m = table[best][i]
        held_out.append(m)
        chosen.append(configs[best])
        if verbose:
            print(f"  fold {test}: selected on the other {len(train)} -> "
                  f"held-out ARI={m.get('ari', float('nan')):.4f} "
                  f"NMI={m.get('nmi', float('nan')):.4f}")

    all_means = [np.nanmean([score_fn(table[c][j]) for j in range(len(folds))])
                 for c in range(len(configs))]
    final = configs[int(np.nanargmax(all_means))]

    return {
        "held_out": held_out,
        "chosen_per_fold": chosen,
        "final_config": final,
        "table": table,
        "folds": folds,
        "configs": configs,
    }


def summarise(cv):
    ari = [m.get("ari", np.nan) for m in cv["held_out"]]
    nmi = [m.get("nmi", np.nan) for m in cv["held_out"]]
    stable = all(c == cv["chosen_per_fold"][0] for c in cv["chosen_per_fold"])
    return {
        "cv_ari_mean": float(np.nanmean(ari)),
        "cv_ari_sd": float(np.nanstd(ari)),
        "cv_nmi_mean": float(np.nanmean(nmi)),
        "cv_nmi_sd": float(np.nanstd(nmi)),
        "selection_stable_across_folds": bool(stable),
        "final_config": cv["final_config"],
    }


def score_labels(gt, pred, valid=None):
    gt, pred = np.asarray(gt), np.asarray(pred)
    if valid is not None:
        gt, pred = gt[valid], pred[valid]
    return {"ari": float(adjusted_rand_score(gt, pred)),
            "nmi": float(normalized_mutual_info_score(gt, pred))}
