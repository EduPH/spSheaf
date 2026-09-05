"""Extended Data — the three distinct causes of the two missed osmFISH domains.

Reads ``data/results/osmfish/elongation.csv`` (``make osmfish-elongation``).
Every value plotted is measured; nothing here is illustrative.

    python ed_osmfish_modes.py
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


DATA = _results("osmfish", "elongation.csv")
OUT = HERE / "out" / "ed_osmfish_modes.pdf"

H_UM = 32.5           # kernel bandwidth on osmFISH
ALPHAS = [0.0, 0.25, 0.5, 0.75]

#: The three domains that carry the three mechanisms, and their colours.
FOCUS = {
    "Layer 3-4":         ("#C1541A", "resolution limit"),
    "Layer 2-3 lateral": ("#0072B2", "expression-degenerate"),
    "Pia Layer 1":       ("#00795C", "compactness bias"),
}
OTHER = "#B6C0CA"
TICK = ss.TYPE_PT["normal"]["tick"]


def load():
    with DATA.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["n"] = int(r["n"])
        r["width_um"] = float(r["width_um"])
        r["elongation"] = float(r["elongation"])
        r["recall"] = [float(r[f"recall_a{a:g}"]) for a in ALPHAS]
    return rows


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos + 1.0
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def panel_a(rows, path):
    """Per-domain recall against the spatial weight."""
    fig, ax = ss.panel(84, 52)
    fig.subplots_adjust(left=0.11, right=0.62, top=0.96, bottom=0.15)
    for r in rows:
        if r["domain"] in FOCUS:
            continue
        ax.plot(ALPHAS, r["recall"], color=OTHER, lw=0.7, zorder=1)
    dy = {"Layer 3-4": -4.0, "Layer 2-3 lateral": 0.0, "Pia Layer 1": 4.0}
    for r in rows:
        d = r["domain"]
        if d not in FOCUS:
            continue
        c, _ = FOCUS[d]
        ax.plot(ALPHAS, r["recall"], color=c, lw=1.4, marker="o", ms=3.0, zorder=3)
        ax.annotate(d, (ALPHAS[-1], r["recall"][-1]), xytext=(6, dy[d]),
                    textcoords="offset points", color=c, fontsize=TICK, va="center")
    ax.axvline(0.5, color="#555555", lw=0.5, ls=(0, (2, 2)))
    ax.text(0.5, 1.03, "published", fontsize=4, color="#555555", ha="center")
    ax.set_xticks(ALPHAS)
    ax.set_xlim(-0.04, 0.79)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel(r"spatial weight $\alpha_s$")
    ax.set_ylabel("per-domain recall")
    ss.save_panel(fig, path)


def panel_b(rows, path):
    """Uncontaminated interior width."""
    fig, ax = ss.panel(84, 52)
    fig.subplots_adjust(left=0.42, right=0.97, top=0.96, bottom=0.15)
    rs = sorted(rows, key=lambda r: r["width_um"])
    for i, r in enumerate(rs):
        c = FOCUS.get(r["domain"], (OTHER, None))[0]
        ax.barh(i, r["width_um"] - 2 * H_UM, height=0.72, color=c)
    ax.set_yticks(range(len(rs)))
    ax.set_yticklabels([r["domain"] for r in rs])
    ax.set_ylim(-0.7, len(rs) - 0.3)
    ax.set_xlabel(r"uncontaminated interior $W-2h$ ($\mu$m)")
    ax.axvline(0, color="#333333", lw=0.5)
    ss.save_panel(fig, path)


def panel_c(rows, path):
    """Elongation against recall, at the published weight and above it."""
    fig, ax = ss.panel(84, 52)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.15)
    el = [r["elongation"] for r in rows]
    for idx, (alpha, marker, shade) in enumerate(
            [(0.5, "o", "#0072B2"), (0.75, "s", "#C1541A")]):
        k = ALPHAS.index(alpha)
        rec = [r["recall"][k] for r in rows]
        ax.scatter(el, rec, s=9, marker=marker, facecolor="none",
                   edgecolor=shade, linewidths=0.7,
                   label=rf"$\alpha_s={alpha:g}$   $\rho={spearman(el, rec):+.2f}$")
    ax.set_xlabel("domain elongation (PCA aspect ratio)")
    ax.set_ylabel("recall")
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2,
              handletextpad=0.4, columnspacing=1.4, fontsize=TICK)
    ss.save_panel(fig, path)


def main():
    ss.apply()
    rows = load()
    panel_a(rows, PANELS / "edosm_a.pdf")
    panel_b(rows, PANELS / "edosm_b.pdf")
    panel_c(rows, PANELS / "edosm_c.pdf")
    lay = Layout(
        name="ed_osmfish_modes", width_mm=180.0, height_mm=124.0,
        panels=[
            Panel("a", PANELS / "edosm_a.pdf", x=8,  y=6,  label_dx=-4, label_dy=-5),
            Panel("b", PANELS / "edosm_b.pdf", x=96, y=6,  label_dx=-4, label_dy=-5),
            Panel("c", PANELS / "edosm_c.pdf", x=8,  y=66, label_dx=-4, label_dy=-5),
        ],
    )
    print(lay.summary())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("\nwrote", lay.compose(OUT))


if __name__ == "__main__":
    main()
