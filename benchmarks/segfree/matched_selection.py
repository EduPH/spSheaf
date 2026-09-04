"""Matched model selection across all four methods (memo point 5; Y.I. 2.6).

The published comparison selects each competitor's hyperparameters by taking,
for every slice independently, the configuration that maximises ARI against
that slice's ground-truth annotation. spSheaf's parameters instead come from
leave-one-slice-out cross-validation, so they were never fitted on the slice
they are scored on. The comparison is therefore biased in the competitors'
favour, and the reported margin is a lower bound.

This script removes the asymmetry without re-running anything. The competitor
sweeps in ``results/metrics/`` record every configuration's score on every
slice, which is exactly what leave-one-slice-out needs:

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

#: spSheaf's own numbers, already selected by leave-one-slice-out.
SPSHEAF = {"ARI": 0.557, "NMI": 0.631}


# --- loaders: config identity -> {slice: score} -----------------------------

def load_points2regions(metric: str) -> dict:
    out: dict = {}
    for s in SLICES:
        path = RESULTS / "metrics" / f"points2regions_merfish__metrics_{s}.json"
        for e in json.loads(path.read_text())["grid"]:
            key = (e["pixel_width"], e["pixel_smoothing"])
            out.setdefault(key, {})[s] = e[metric]
    return out


def load_ssam(metric: str, guided: bool = False) -> dict:
    stem = "ssam_guided_merfish" if guided else "ssam_merfish"
    out: dict = {}
    for s in SLICES:
        path = RESULTS / "metrics" / f"{stem}__metrics_{s}.json"
        for e in json.loads(path.read_text())["sweep"]:
            key = json.dumps(e.get("params", {}), sort_keys=True)
            out.setdefault(key, {})[s] = e[metric]
    return out


def load_ficture(metric: str) -> dict:
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


def evaluate(metric: str = "ARI") -> list[dict]:
    methods = [
        ("Points2Regions", load_points2regions(metric)),
        ("FICTURE", load_ficture(metric)),
        ("SSAM (de novo)", load_ssam(metric)),
        ("SSAM (guided)", load_ssam(metric, guided=True)),
    ]
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
    sp = SPSHEAF[metric]
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
    out += [
        "-" * 67,
        f"{'spSheaf':17s}{'--':>9s}{'--':>11s}{'--':>12s}{sp:>9.3f}{'1':>7s}/5",
        "",
    ]
    best_pub = max(r["per_slice_argmax"] for r in rows if r["n_configs"])
    best_loo = max(r["loo"] for r in rows if r["n_configs"])
    out += [
        f"spSheaf margin over the best competitor",
        f"  as published (per-slice argmax) : +{sp - best_pub:.3f}",
        f"  under matched LOO-CV            : +{sp - best_loo:.3f}",
        "",
        "The published rule lets a competitor use a different configuration on",
        "every slice, chosen on that slice's own annotation. Removing that",
        f"advantage widens the margin by {(sp - best_loo) - (sp - best_pub):+.3f} {metric}.",
    ]
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
        r"\midrule",
        f"\\textbf{{spSheaf}} & --- & --- & \\textbf{{{SPSHEAF[metric]:.3f}}} & 1/5 \\\\",
        r"\botrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metric", default="ARI", choices=["ARI", "NMI"])
    ap.add_argument("--latex", action="store_true", help="emit a LaTeX tabular")
    ap.add_argument("--csv", type=Path, help="also write the rows to a CSV")
    args = ap.parse_args(argv)

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
            w.writerow(["spSheaf", "", "", "", f"{SPSHEAF[args.metric]:.4f}", ""])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
