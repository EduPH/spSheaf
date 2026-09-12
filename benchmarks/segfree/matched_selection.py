"""Matched model selection across all four methods (memo point 5; Y.I. 2.6).

The published comparison selects each competitor's hyperparameters by taking,
for every slice independently, the configuration that maximises ARI against
that slice's ground-truth annotation. spSheaf's parameters instead come from
leave-one-slice-out cross-validation, so they were never fitted on the slice
they are scored on. The comparison is therefore biased in the competitors'
favour, and the reported margin is a lower bound.

Since 12 Sep 2026 spSheaf is put under the SAME three rules as the competitors,
from its own 27-configuration grid (``experiments/merfish/loo_cv.py``,
``reported/merfish/loo_cv_table.csv``): its per-slice argmax, its one
configuration by mean (which is the development configuration, 0.571) and its
leave-one-slice-out value (0.552), with the number of distinct fold winners.
Nothing about spSheaf is hard-coded any more.

This script removes the asymmetry without re-running anything. The competitor
sweeps in ``results/metrics/`` record every configuration's score on every
slice, which is exactly what leave-one-slice-out needs. By default the
``*_merfish_clean__*`` tables are read (competitors run on the physical section
selected by bregma, 8 Sep 2026); ``--legacy`` reads the original
``*_merfish__*`` tables, which were computed on two superimposed sections (see
``common.load_merfish_slice``) and are kept only as a record.

    for each held-out slice s:

        c* = argmax over configurations of mean ARI on the other four slices
        report the score of c* on s

Three selection rules are reported so the effect is visible:

``per-slice argmax``
    A different configuration for each slice, chosen on that slice's own
    annotation. This is what the manuscript currently reports.
``one config by mean``
    A single configuration for all five slices, still chosen in-sample.
``leave-one-slice-out``
    The matched rule. Directly comparable to the spSheaf number.

Only configurations scored on all five slices are eligible, so every method is
selected from a complete grid.

    python -m benchmarks.segfree.matched_selection
    python -m benchmarks.segfree.matched_selection --metric NMI --latex

osmFISH is deliberately excluded: it is a single section, so there is nothing
to hold out. The matched protocol there would need leave-one-spatial-block-out
(``experiments/common/cv.py``) with each competitor re-scored per block, and
the stored metrics are whole-section only. The osmFISH comparison therefore
keeps the competitor-favouring in-sample selection, which must be stated.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SLICES = ["0.04", "0.09", "0.14", "0.19", "0.24"]

#: spSheaf's own grid: every (sigma_s, alpha_b, alpha_s) x slice of
#: experiments/merfish/loo_cv.py, so that spSheaf is selected under exactly the
#: rules the competitors are. The tracked copy under reported/ is read first;
#: the live results tree second.
def _spsheaf_table() -> Path:
    here = Path(__file__).resolve().parents[2]
    cands = [here / "reported" / "merfish" / "loo_cv_table.csv"]
    try:
        from spsheaf import paths
        cands.append(paths.results("merfish") / "loo_cv_table.csv")
    except Exception:
        pass
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError("spSheaf grid: run `make merfish-loo-cv` (loo_cv_table.csv)")


def load_spsheaf(metric: str) -> dict:
    """spSheaf configurations -> {slice: score}, from loo_cv_table.csv."""
    out: dict = {}
    with _spsheaf_table().open() as fh:
        for r in csv.DictReader(fh):
            s = r["slice"].replace("MERFISH_", "").replace(".h5ad", "")
            key = (float(r["smooth_radius"]), float(r["boundary_alpha"]),
                   float(r["spatial_weight"]))
            out.setdefault(key, {})[s] = float(r[metric.lower()])
    return out


LEGACY = False          # set by --legacy: the contaminated bbox inputs
SUFFIX = lambda: "" if LEGACY else "_clean"


# --- loaders: config identity -> {slice: score} -----------------------------

def load_points2regions(metric: str) -> dict:
    out: dict = {}
    for s in SLICES:
        path = RESULTS / "metrics" / f"points2regions_merfish{SUFFIX()}__metrics_{s}.json"
        for e in json.loads(path.read_text())["grid"]:
            key = (e["pixel_width"], e["pixel_smoothing"])
            out.setdefault(key, {})[s] = e[metric]
    return out


#: SSAM cell-type clustering variants that are part of its configuration grid.
#: The empty tag is the published recipe of the SSAM paper as SSAM 1.1.3
#: implements it (built-in normalisation, medoid-correlation outlier filter on,
#: SNN Louvain resolution 0.15), which in 1.1.3 leaves 5-7 cell types; the
#: ``_sct_r*`` tags use sctransform normalisation with the outlier filter off
#: and the Louvain resolution raised to 4 and 8, which recovers 39-44 and 61-63
#: cell types -- the order of the 50-68 types the SSAM paper reports.
SSAM_TAGS = ["", "_sct_r4", "_sct_r8"]


def load_ssam(metric: str, guided: bool = False) -> dict:
    stem = "ssam_guided_merfish" if guided else f"ssam_merfish{SUFFIX()}"
    tags = [""] if guided else SSAM_TAGS
    out: dict = {}
    for tag in tags:
        for s in SLICES:
            path = RESULTS / "metrics" / f"{stem}__metrics_{s}{tag}.json"
            if not path.exists():
                if tag == "":
                    return {}
                break
            for e in json.loads(path.read_text())["sweep"]:
                if e[metric] != e[metric]:      # NaN: nothing above norm_thres
                    continue
                params = dict(e.get("params", {}))
                params["clustering"] = tag or "published"
                key = json.dumps(params, sort_keys=True)
                out.setdefault(key, {})[s] = e[metric]
    return out


#: FICTURE's default seed. Its `spatial` and `kmeans` aggregations were run at
#: five seeds each while `argmax` is deterministic; the seeded runs are reduced
#: to this one seed so that a FICTURE configuration is a single run, which is
#: what a Points2Regions, SSAM or spSheaf configuration is (see
#: ``load_points2regions`` and ``load_ssam`` above, and spSheaf's own grids at
#: k-means seed 42). Averaging FICTURE's seeds instead -- which this function
#: used to do -- scores it at a configuration's expected value while every other
#: method is scored at one draw, and the two are not comparable: on MERFISH the
#: difference is 0.016 held-out ARI, against FICTURE.
FICTURE_SEED = "0.0"


def _ficture_default_seed(rows: list[tuple[str, float]]) -> float:
    """The default-seed run of one configuration, or its only run if unseeded."""
    by_seed = dict(rows)
    for k in (FICTURE_SEED, ""):
        if k in by_seed:
            return by_seed[k]
    return next(iter(by_seed.values()))


def load_ficture(metric: str) -> dict:
    if not LEGACY:
        # clean rerun: one table per slice from eval_ficture.py; a configuration
        # is (strategy, model), taken at FICTURE_SEED so that it is one run
        out: dict = {}
        for s in SLICES:
            path = RESULTS / "metrics" / f"ficture_merfish_clean__metrics_{s}.csv"
            acc: dict = {}
            with path.open() as fh:
                for r in csv.DictReader(fh):
                    acc.setdefault((r["strategy"], r["model"]), []).append(
                        (r.get("seed", ""), float(r[metric])))
            for key, rows in acc.items():
                out.setdefault(key, {})[s] = _ficture_default_seed(rows)
        return out
    out: dict = {}
    with (RESULTS / "ficture_all_settings.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] != "MERFISH":
                continue
            s = r["unit"].replace("merfish_", "")
            key = (r["strategy"], r["model"], r["n_factors"])
            try:
                out.setdefault(key, {})[s] = float(r[metric])
            except (ValueError, KeyError):
                continue
    return out


# --- selection rules --------------------------------------------------------

def _complete(cfgs: dict) -> dict:
    """Configurations scored on every slice."""
    return {k: v for k, v in cfgs.items() if all(s in v for s in SLICES)}


def per_slice_argmax(cfgs: dict) -> float:
    return st.mean(max(v[s] for v in cfgs.values()) for s in SLICES)


def one_config_by_mean(cfgs: dict) -> float:
    best = max(cfgs.values(), key=lambda v: st.mean(v[s] for s in SLICES))
    return st.mean(best[s] for s in SLICES)


def leave_one_slice_out(cfgs: dict) -> tuple[float, list, int]:
    """Return (mean held-out score, per-slice scores, number of distinct winners)."""
    scores, winners = [], []
    for s in SLICES:
        rest = [r for r in SLICES if r != s]
        key, vals = max(
            cfgs.items(), key=lambda kv: st.mean(kv[1][r] for r in rest)
        )
        scores.append(vals[s])
        winners.append(key)
    return st.mean(scores), scores, len(set(map(str, winners)))


def evaluate_grid(name: str, metric: str) -> dict:
    return {"Points2Regions": load_points2regions, "FICTURE": load_ficture,
            "SSAM (de novo)": load_ssam, "spSheaf": load_spsheaf,
            "SSAM (guided)": lambda m: load_ssam(m, guided=True)}[name](metric)


def evaluate(metric: str = "ARI") -> list[dict]:
    methods = [
        ("Points2Regions", load_points2regions(metric)),
        ("FICTURE", load_ficture(metric)),
        ("SSAM (de novo)", load_ssam(metric)),
        ("spSheaf", load_spsheaf(metric)),
    ]
    if LEGACY:
        methods.append(("SSAM (guided)", load_ssam(metric, guided=True)))
    rows = []
    for name, cfgs in methods:
        full = _complete(cfgs)
        if not full:
            rows.append({"method": name, "n_configs": 0})
            continue
        loo, per, n_win = leave_one_slice_out(full)
        rows.append(
            {
                "method": name,
                "n_configs": len(full),
                "per_slice_argmax": per_slice_argmax(full),
                "one_config": one_config_by_mean(full),
                "loo": loo,
                "loo_per_slice": per,
                "distinct_winners": n_win,
            }
        )
    return rows


# --- reporting --------------------------------------------------------------

def report(rows: list[dict], metric: str) -> str:
    out = [
        f"MERFISH, mean {metric} over 5 slices, by model-selection rule",
        "",
        f"{'method':17s}{'configs':>9s}{'per-slice':>11s}{'one config':>12s}"
        f"{'LOO-CV':>9s}{'winners':>9s}",
        f"{'':17s}{'':>9s}{'argmax':>11s}{'by mean':>12s}{'(matched)':>9s}{'':>9s}",
        "-" * 67,
    ]
    for r in rows:
        if not r["n_configs"]:
            out.append(f"{r['method']:17s}{'no complete grid':>45s}")
            continue
        out.append(
            f"{r['method']:17s}{r['n_configs']:>9d}"
            f"{r['per_slice_argmax']:>11.3f}{r['one_config']:>12.3f}"
            f"{r['loo']:>9.3f}{r['distinct_winners']:>7d}/5"
        )
    sp = next((r for r in rows if r["method"] == "spSheaf" and r["n_configs"]), None)
    comp = [r for r in rows if r["method"] != "spSheaf" and r["n_configs"]]
    if sp and comp:
        out += ["-" * 67, "spSheaf margin over the best competitor, same rule on both sides"]
        for rule, label in (("per_slice_argmax", "per-slice argmax"),
                            ("one_config", "one configuration"),
                            ("loo", "leave-one-slice-out")):
            out.append(f"  {label:22s}: {sp[rule] - max(r[rule] for r in comp):+.3f}")
    return "\n".join(out)


def latex(rows: list[dict], metric: str) -> str:
    lines = [
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{selection rule} & \\",
        r"\cmidrule(lr){2-4}",
        r"Method & per-slice & single & LOO-CV & folds \\",
        r"& argmax & config & (matched) & won \\",
        r"\midrule",
    ]
    for r in rows:
        if not r["n_configs"]:
            continue
        lines.append(
            f"{r['method']} & {r['per_slice_argmax']:.3f} & {r['one_config']:.3f} "
            f"& {r['loo']:.3f} & {r['distinct_winners']}/5 \\\\"
        )
    lines += [
        r"\botrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metric", default="ARI", choices=["ARI", "NMI"])
    ap.add_argument("--latex", action="store_true", help="emit a LaTeX tabular")
    ap.add_argument("--csv", type=Path, help="also write the rows to a CSV")
    ap.add_argument("--per-slice-csv", type=Path,
                    help="also write per-slice held-out and argmax scores (Fig. 2b)")
    ap.add_argument("--legacy", action="store_true",
                    help="use the original bbox-selected (contaminated) competitor tables")
    args = ap.parse_args(argv)
    global LEGACY
    LEGACY = args.legacy

    rows = evaluate(args.metric)
    print(report(rows, args.metric))
    if args.latex:
        print("\n" + latex(rows, args.metric))
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "n_configs", "per_slice_argmax",
                        "one_config", "loo_cv", "distinct_winners"])
            for r in rows:
                if not r["n_configs"]:
                    continue
                w.writerow([r["method"], r["n_configs"],
                            f"{r['per_slice_argmax']:.4f}", f"{r['one_config']:.4f}",
                            f"{r['loo']:.4f}", r["distinct_winners"]])
        print(f"\nwrote {args.csv}")
    if args.per_slice_csv:
        with args.per_slice_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "slice", "held_out", "per_slice_argmax"])
            for r in rows:
                if not r["n_configs"]:
                    continue
                full = _complete(dict(evaluate_grid(r["method"], args.metric)))
                for s, v in zip(SLICES, r["loo_per_slice"]):
                    w.writerow([r["method"], s, f"{v:.4f}",
                                f"{max(c[s] for c in full.values()):.4f}"])
        print(f"wrote {args.per_slice_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
