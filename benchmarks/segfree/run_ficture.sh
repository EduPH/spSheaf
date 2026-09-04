#!/usr/bin/env bash
# Run FICTURE over a dataset with several factor counts and training widths.
#
# Usage: bash run_ficture.sh <input_dir> <out_prefix> [n_factor_csv] [train_width_csv]
#
# FICTURE's factor count is a free parameter with two sensible regimes:
#   n_factor == number of annotated domains  -> factors act as domains directly
#   n_factor >  number of annotated domains  -> factors act as cell types, and
#       domains are recovered from their local composition (how FICTURE is used
#       in its own paper). eval_ficture.py handles both.
# Training width 18 um is FICTURE's own default for MERSCOPE-scale data; 12 um is
# the package default and is included as a second setting.
set -euo pipefail

IN_DIR=$1
OUT_PREFIX=$2
NFACTOR=${3:-8,12,20}
WIDTH=${4:-12,18}
THREADS=${THREADS:-8}
JOBS=${JOBS:-4}

ficture run_together \
    --in-tsv "${IN_DIR}/transcripts.tsv.gz" \
    --in-minmax "${IN_DIR}/coordinate_minmax.tsv" \
    --in-feature "${IN_DIR}/feature.clean.tsv.gz" \
    --major-axis Y \
    --out-dir "${OUT_PREFIX}" \
    --n-factor "${NFACTOR}" \
    --train-width "${WIDTH}" \
    --seed 1 --threads "${THREADS}" --n-jobs "${JOBS}" --all --dry-run

# -k so that a failure in a downstream cosmetic step (differential expression,
# figure rendering) does not prevent the other models' pixel output from being
# produced; the evaluation only needs *.pixel.sorted.tsv.gz.
make -f "${OUT_PREFIX}/Makefile" -j "${JOBS}" -k || true

echo "pixel outputs:"
find "${OUT_PREFIX}/analysis" -name "*.pixel.sorted.tsv.gz" -printf "  %p\n"
