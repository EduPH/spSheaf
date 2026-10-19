"""Extended Data — SDMBench ranking and seed sensitivity.

Rebuilds, from the stored ranking tables, the two legacy figures
``benchmark_nmi.png`` (210 dpi at submission width) and the ranking panels of
``Merfish.png``.

Sources, all under ``data/results/benchmark_sdmbench``:
  output_benchmark/benchmark_MERFISH_<metric>_ranking.csv
  output_seed_sensitivity/seed_sensitivity_per_slice.csv

    python ed_benchmark.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import sheafstyle as ss
from layout import Layout, Panel

HERE = Path(__file__).parent
REPO = HERE.parent
PANELS = HERE / "panels"
PANELS.mkdir(exist_ok=True)
def _results(*parts):
    """Resolve a results path through ``spsheaf.paths``.

    Building it from the repository root instead ignores SPSHEAF_DATA and
    paths.toml, so a checkout whose data lives elsewhere -- which is the normal
    case, the tree is tens of gigabytes -- cannot find its own inputs.
    """
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from spsheaf import paths
    return paths.results().joinpath(*parts)


BENCH = _results("benchmark_sdmbench")
OUT = HERE / "out" / "ed_benchmark.pdf"

OURS = "Sheaf (ours)"
LABEL = "spSheaf"

#: Metrics ranked in the comparison, and whether a higher value ranks better.
#: Moran's I and Geary's C are excluded: recomputed from molecule-level data the
#: ground-truth partition itself scores below every baseline on them.
METRICS = [("NMI", True), ("HOM", True), ("COM", True),
           ("CHAOS", False), ("PAS", False), ("ASW", True)]

TICK = ss.TYPE_PT["normal"]["tick"]
GREY = "#B6C0CA"


def ranking(metric: str) -> list[tuple[str, float, int]]:
    path = BENCH / "output_benchmark" / f"benchmark_MERFISH_{metric}_ranking.csv"
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return [(r["Method"], float(r[metric]), int(r["rank"])) for r in rows]


def seed_table() -> list[dict]:
    path = BENCH / "output_seed_sensitivity" / "seed_sensitivity_per_slice.csv"
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["slice"] = r["slice"].replace("MERFISH_", "").replace(".h5ad", "")
    return rows


def panel_a(path):
    """NMI ranking across the twelve methods."""
    rows = ranking("NMI")
    fig, ax = ss.panel(84, 58)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.97, bottom=0.13)
    rows = rows[::-1]
    for i, (name, val, _) in enumerate(rows):
        ax.barh(i, val, height=0.74,
                color=ss.METHOD_COLORS["spSheaf"] if name == OURS else GREY)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL if n == OURS else n for n, _, _ in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0, max(v for _, v, _ in rows) * 1.08)
    ax.set_xlabel("NMI (pooled median over slices and seeds)")
    ss.save_panel(fig, path)


def panel_b(path):
    """Rank of every method on each metric; spSheaf traced across."""
    fig, ax = ss.panel(84, 58)
    fig.subplots_adjust(left=0.12, right=0.86, top=0.97, bottom=0.13)
    ours = []
    for xi, (metric, _) in enumerate(METRICS):
        rows = ranking(metric)
        for name, _, rank in rows:
            if name == OURS:
                ours.append((xi, rank))
            else:
                ax.plot(xi, rank, marker="_", ms=7, mew=0.9, color=GREY, zorder=1)
    ax.plot([x for x, _ in ours], [r for _, r in ours], marker="o", ms=3.4,
            lw=1.2, color=ss.METHOD_COLORS["spSheaf"], zorder=3)
    ax.annotate(LABEL, ours[-1], xytext=(7, 0), textcoords="offset points",
                color=ss.METHOD_COLORS["spSheaf"], fontsize=TICK, va="center")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([m for m, _ in METRICS])
    ax.set_xlim(-0.4, len(METRICS) - 0.4)
    ax.invert_yaxis()
    ax.set_ylim(12.6, 0.4)
    ax.set_yticks([1, 3, 6, 9, 12])
    ax.set_ylabel("rank of twelve")
    ss.save_panel(fig, path)


def panel_c(path):
    """Seed sensitivity, per slice."""
    rows = seed_table()
    fig, ax = ss.panel(84, 52)
    # left margin equal to panel a's, so the two y-axes sit on one vertical
    fig.subplots_adjust(left=0.16, right=0.86, top=0.96, bottom=0.16)
    x = range(len(rows))
    for key, colour, name in (("nmi", "#5AA9D6", "NMI"),
                              ("ari", ss.METHOD_COLORS["spSheaf"], "ARI")):
        mean = [float(r[f"{key}_mean"]) for r in rows]
        sd = [float(r[f"{key}_std"]) for r in rows]
        ax.errorbar(list(x), mean, yerr=sd, marker="o", ms=3.0, lw=1.2,
                    capsize=1.6, elinewidth=0.7, color=colour)
        ax.annotate(name, (len(rows) - 1, mean[-1]), xytext=(7, 0),
                    textcoords="offset points", color=colour, fontsize=TICK,
                    va="center")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["slice"] for r in rows])
    ax.set_xlim(-0.3, len(rows) - 0.7)
    ax.set_ylim(0.40, 0.76)
    ax.set_xlabel("MERFISH slice (Bregma)")
    ax.set_ylabel("mean $\\pm$ s.d. over 20 seeds")
    ss.save_panel(fig, path)


def main():
    ss.apply()
    panel_a(PANELS / "edb_a.pdf")
    panel_b(PANELS / "edb_b.pdf")
    panel_c(PANELS / "edb_c.pdf")
    lay = Layout(
        name="ed_benchmark", width_mm=180.0, height_mm=130.0,
        panels=[
            Panel("a", PANELS / "edb_a.pdf", x=6,  y=6,  label_dx=-4, label_dy=-5),
            Panel("b", PANELS / "edb_b.pdf", x=94, y=6,  label_dx=-4, label_dy=-5),
            Panel("c", PANELS / "edb_c.pdf", x=6,  y=72, label_dx=-4, label_dy=-5),
        ],
    )
    print(lay.summary())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("\nwrote", lay.compose(OUT))


if __name__ == "__main__":
    main()
