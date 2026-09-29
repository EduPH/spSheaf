"""The division table of the manuscript: every division of an annotated domain at
k = 8, 9, 10 on the five MERFISH sections, from ``k_split_events.csv``.

    python -m experiments.merfish.k_split_table > k_split_table.tex
"""
from __future__ import annotations

import pandas as pd

from spsheaf import paths


def main():
    ev = pd.read_csv(paths.results("merfish") / "k_split_events.csv")
    ev = ev.sort_values(["slice", "k", "domain"])
    rows = []
    for r in ev.itertuples():
        if r.hemispheric:
            rr = "hemispheric"
            genes = "---"
        elif pd.isna(r.r_best):
            rr = "---"
            genes = " ".join(str(r.up_in_smaller).split()[:4])
        else:
            rr = f"${r.r_best:+.2f}$ ({r.r_best_with})"
            genes = " ".join(str(r.up_in_smaller).split()[:4])
        rows.append(f"{r.slice} & {r.k} & {r.domain} & "
                    f"{r.share_1:.2f} / {r.share_2:.2f} & "
                    f"{r.purity_1:.2f} / {r.purity_2:.2f} & "
                    f"${r.offset_x:+.0f}$, ${r.offset_y:+.0f}$ & {rr} & {genes} \\\\")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
