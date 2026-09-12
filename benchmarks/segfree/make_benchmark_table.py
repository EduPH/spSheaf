"""Build ``results/benchmark_table.csv`` from the tracked per-run metric files.

This is the whole-section comparison table: one row per (dataset, method,
setting), for the three single sections that have one annotated section each.
It used to be maintained by hand, which is how it came to disagree with the
manuscript -- it still carried SSAM's osmFISH score from before resolution was
treated as the hyperparameter it is (0.566 against the 0.729 it actually
reaches). Deriving it from the files instead means it cannot drift again.

What it reads, all under ``benchmarks/segfree/results/metrics/``:
    <ds>__metrics_ssam[_r<res>].json     benchmarks.segfree.ssam_sweep_offline
    <ds>__metrics_p2r.json               benchmarks.segfree.run_{osmfish,spacetx}
    ficture__metrics_<ds>.csv            benchmarks.segfree.eval_ficture

One rule for every method: **one run per configuration, at a fixed seed, then
the argmax over configurations.** Points2Regions' grid is one run per setting,
SSAM's find_domains sweep likewise, spSheaf's own grids are at k-means seed 42,
and FICTURE -- the only method with seed replicates -- is reduced to its default
seed rather than averaged or maximised over them. ``search_n`` records how many
configurations each argmax was taken over, which is what makes the comparison
auditable.

SSAM is reported per resolution AND as the argmax over all of them, because
resolution is a hyperparameter of SSAM's clustering, not a fixed property of the
method: the "best of all settings" row is what the manuscript compares against,
and it is the fair comparison precisely because it gives SSAM its whole grid.

spSheaf's own rows are NOT added here; they live in ``reported/``, so that this
file stays what it says it is -- the competitors, scored on the same tables.

    python -m benchmarks.segfree.make_benchmark_table
    python -m benchmarks.segfree.make_benchmark_table --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
METRICS = HERE / "results" / "metrics"
OUT = HERE / "results" / "benchmark_table.csv"

FIELDS = ["ARI", "NMI", "HOM", "COM", "CHAOS", "PAS", "ASW", "n_pred_domains"]

#: file prefix -> the dataset name to print
DATASETS = {
    "osmfish": "osmFISH",
    "spacetx_MERFISH_Allen_Prototype": "Allen prototype MERFISH",
    "spacetx_BARISTASeq": "BaristaSeq",
}


def row(dataset, method, setting, m, *, search_n=None, params=None, note=""):
    r = {"dataset": dataset, "unit": "whole section", "method": method,
         "setting": setting, "search_n": search_n}
    r.update({k: m.get(k) for k in FIELDS})
    r["params"] = json.dumps(params, default=str) if params is not None else None
    r["note"] = note
    return r


def ssam_rows(prefix, dataset):
    """One row per resolution, plus the argmax over all of them."""
    out, best, best_key = [], None, None
    for p in sorted(METRICS.glob(f"{prefix}__metrics_ssam*.json")):
        tag = p.stem.split("__metrics_ssam")[1].lstrip("_") or "published_settings"
        d = json.loads(p.read_text())
        res = d.get("resolution") or d.get("ssam_settings", {}).get("resolution")
        norm = d.get("normalization") or d.get("ssam_settings", {}).get("normalization")
        label = f"resolution {res:g}" if res else "default resolution"
        if norm:
            label += f", {norm}"
        sweep = d.get("sweep") or []
        if d.get("published"):
            out.append(row(dataset, "SSAM (de novo)", f"published settings, {label}",
                           d["published"], search_n=1,
                           note=f"{d.get('n_celltypes', '?')} cell types, "
                                f"{d.get('n_localmax', '?')} local maxima"))
        if sweep:
            b = max(sweep, key=lambda r: r["ARI"])
            out.append(row(dataset, "SSAM (de novo)", f"best find_domains sweep, {label}",
                           b, search_n=len(sweep), params=b.get("params")))
            if best is None or b["ARI"] > best["ARI"]:
                best, best_key = b, (label, len(sweep))
    # every setting SSAM was given, across resolutions: the comparison the paper makes
    total = sum(r["search_n"] or 0 for r in out if "best find_domains" in r["setting"])
    if best is not None:
        out.append(row(dataset, "SSAM (de novo)", "best of all settings", best,
                       search_n=total, params=best.get("params"),
                       note=f"argmax at {best_key[0]}; this is the Table 2 value"))
    return out


def p2r_rows(prefix, dataset):
    p = METRICS / f"{prefix}__metrics_p2r.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    grid = d.get("grid") or []
    out = []
    if d.get("best"):
        b = max(grid, key=lambda r: r["ARI"]) if grid else {}
        out.append(row(dataset, "Points2Regions", "best of grid", d["best"],
                       search_n=len(grid) or None, params=b.get("params"),
                       note="this is the Table 2 value"))
    seeds = d.get("seeds") or []
    if seeds:
        s = pd.DataFrame(seeds)
        if "ARI" in s:
            out.append(row(dataset, "Points2Regions",
                           f"best setting, mean of {len(s)} seeds",
                           s[[c for c in FIELDS if c in s]].mean().to_dict(),
                           search_n=len(grid) or None,
                           note=f"ARI sd {s.ARI.std():.3f}"))
    return out


#: what identifies one FICTURE configuration, as opposed to one run of it
FIC_CONFIG = ["strategy", "model", "n_factors"]
#: FICTURE's default seed -- see ficture_rows
FIC_SEED = 0.0


def ficture_one_run(d):
    """The rows of a FICTURE metric table that count as one run per
    configuration: the default seed, or the only run of a deterministic
    strategy. The figures read the same rule (ficture_labels.py, ed_spacetx.py)."""
    return d[(d.seed == FIC_SEED) | d.seed.isna()] if "seed" in d.columns else d


def ficture_rows(prefix, dataset):
    """FICTURE under the same rule as every other method here.

    That rule is: one run per configuration, at a fixed seed, then the argmax
    over configurations. Points2Regions' grid is one run per (pixel width,
    smoothing); SSAM's find_domains sweep is one run per setting; spSheaf's own
    grids are one run per configuration at k-means seed 42. FICTURE is the only
    method with seed replicates -- its `spatial` and `kmeans` aggregations were
    run at five seeds each, while `argmax` is deterministic -- so its seeded
    configurations are taken at FIC_SEED and the deterministic ones as they are.

    Neither alternative is comparable to the others. Averaging the seeds scores
    FICTURE at a configuration's expected value while every other method sits at
    one draw; taking the best of all 66 runs hands it a luckier draw than a
    method with one run per setting. Both are emitted below for reference, with
    the rule named in the row, but the matched one is what the manuscript uses.
    """
    p = METRICS / f"ficture__metrics_{prefix}.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p).dropna(subset=["ARI"])
    if d.empty:
        return []
    keys = [k for k in FIC_CONFIG if k in d.columns]
    num = [c for c in FIELDS if c in d.columns]

    def params_of(name):
        return dict(zip(keys, name if isinstance(name, tuple) else (name,)))

    # (a) the matched rule: one run per configuration, at the default seed
    one = ficture_one_run(d)
    a = one.loc[one.ARI.idxmax()]
    out = [row(dataset, "FICTURE", "best configuration", a.to_dict(),
               search_n=len(one), params=params_of(tuple(a[k] for k in keys)),
               note=f"one run per configuration at seed {FIC_SEED:g} "
                    f"(deterministic strategies as run), argmax over "
                    f"{len(one)} -- the rule every method here is under; "
                    f"this is the Table 2 value")]

    # (b), (c) the two alternatives, for reference only
    if "seed" in d.columns and len(one) < len(d):
        g = d.groupby(keys, dropna=False)
        agg = g[num].mean()
        agg["n_runs"] = g.size()
        agg["ari_sd"] = g.ARI.std()
        b = agg.loc[agg.ARI.idxmax()]
        out.append(row(dataset, "FICTURE", "best configuration, seed mean",
                       b.to_dict(), search_n=len(agg), params=params_of(b.name),
                       note=f"mean of up to {int(agg.n_runs.max())} runs; an "
                            f"expected value, so NOT comparable to the "
                            f"single-run figures above. Reference only."))
        c = d.loc[d.ARI.idxmax()]
        out.append(row(dataset, "FICTURE", "best single run", c.to_dict(),
                       search_n=len(d), params=params_of(tuple(c[k] for k in keys)),
                       note="maximum over every run including seed replicates, "
                            "which no other method here is given. Reference only."))
    out.append(row(dataset, "FICTURE", "all runs", {}, search_n=len(d),
                   note=f"{len(one)} configurations, {len(d)} runs -- the search "
                        f"budget quoted in Methods"))
    return out


def build():
    rows = []
    for prefix, dataset in DATASETS.items():
        rows += ssam_rows(prefix, dataset)
        rows += p2r_rows(prefix, dataset)
        rows += ficture_rows(prefix, dataset)
    if not rows:
        raise SystemExit(f"no metric files under {METRICS}")
    d = pd.DataFrame(rows)
    order = {n: i for i, n in enumerate(DATASETS.values())}
    d = d.sort_values(["dataset", "method", "setting"],
                      key=lambda c: c.map(order) if c.name == "dataset" else c)
    cols = ["dataset", "unit", "method", "setting", "search_n", *FIELDS, "params", "note"]
    return d[[c for c in cols if c in d.columns]].reset_index(drop=True)


def main():
    a = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--check", action="store_true",
                   help="print what would change and exit non-zero; write nothing")
    a = a.parse_args()

    d = build()
    headline = d[d.note.fillna("").str.contains("Table 2")]
    print(headline[["dataset", "method", "search_n", "ARI", "NMI"]].to_string(index=False))

    new = d.to_csv(index=False)
    if OUT.exists() and OUT.read_text() == new:
        print(f"\n{OUT.name} unchanged ({len(d)} rows)")
        return
    if a.check:
        print(f"\n{OUT.name} would change; run without --check to rewrite")
        raise SystemExit(1)
    OUT.write_text(new)
    print(f"\nwrote {OUT.relative_to(HERE.parent.parent)} ({len(d)} rows)")


if __name__ == "__main__":
    main()
