#!/usr/bin/env python
"""
Resumable, memory-safe streaming cache for the marker-aware Atera re-analysis.

Processes the parquet ONE ROW GROUP at a time (bounded memory), keeping ALL
molecules of the curated marker genes (sparse, biologically decisive) plus a
random `--subsample` fraction of every other gene. Resumable: a progress file
records the next row-group index, so the script can be called repeatedly until
`done=True` (each call stops after ~`--budget` seconds to respect the shell cap).
Per-gene molecule counts (from the kept sample) accumulate in gene_counts.csv.
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import pyarrow.parquet as pq

MARKERS = [
    "ESR1","PGR","ERBB2","GATA3","FOXA1","SCGB2A2","KRT5","CDK1","PLIN1","COL1A2",
    "CDKN2A","KRT14","KRT17","TP63","CD68","XBP1","PTPRC","DCN","GZMB",
    "EPCAM","KRT8","KRT18","KRT19","MKI67","TOP2A","VIM","PECAM1","VWF",
    "COL1A1","COL3A1","ACTA2","POSTN","MS4A1","CD3D","CD3E","CD8A","LYZ","MUC1","TFF1","TFF3",
    # cervical squamous program
    "SOX2","KRT6A","SERPINB3","SERPINB4","KRT15","DSP","IVL",
]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--subsample", type=float, default=0.01)
    p.add_argument("--min-qv", type=float, default=20.0)
    p.add_argument("--budget", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    a = build_parser().parse_args()
    os.makedirs(a.cache_dir, exist_ok=True)
    prog_path = os.path.join(a.cache_dir, "progress.json")
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {"next_rg": 0, "kept": 0}
    pf = pq.ParquetFile(a.input)
    n_rg = pf.num_row_groups
    mset = set(MARKERS)
    cols = ["feature_name","x_location","y_location","qv","is_gene"]

    t0 = time.time(); parts = []; gcounts = {}
    rg = prog["next_rg"]
    while rg < n_rg and time.time() - t0 < a.budget:
        d = pf.read_row_group(rg, columns=cols).to_pandas()
        d = d[(d.qv >= a.min_qv) & (d.is_gene)]
        g = d.feature_name.astype(str)
        rng = np.random.default_rng(a.seed * 100000 + rg)
        keep = g.isin(mset).to_numpy() | (rng.random(len(d)) < a.subsample)
        dk = d.loc[keep, ["feature_name","x_location","y_location"]].copy()
        dk["feature_name"] = dk.feature_name.astype(str)
        parts.append(dk)
        for k, v in dk.feature_name.value_counts().items():
            gcounts[k] = gcounts.get(k, 0) + int(v)
        rg += 1

    if parts:
        part = pd.concat(parts, ignore_index=True)
        part.to_parquet(os.path.join(a.cache_dir, f"part_{prog['next_rg']:04d}.parquet"), index=False)
        prog["kept"] += len(part)
        gcp = os.path.join(a.cache_dir, "gene_counts.csv")
        old = pd.read_csv(gcp, index_col=0).iloc[:, 0].to_dict() if os.path.exists(gcp) else {}
        for k, v in gcounts.items(): old[k] = old.get(k, 0) + v
        pd.Series(old, name="count").sort_values(ascending=False).to_csv(gcp)

    prog["next_rg"] = rg
    prog["done"] = rg >= n_rg
    json.dump(prog, open(prog_path, "w"))
    print(f"row groups {prog['next_rg']}/{n_rg} ({100*rg/n_rg:.1f}%); "
          f"kept total {prog['kept']:,}; done={prog['done']}")


if __name__ == "__main__":
    main()
