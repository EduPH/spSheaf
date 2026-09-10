#!/usr/bin/env bash
# Download the SpaceTx mouse-VISp imaging datasets (spot tables + cell tables).
#
# Source: https://spacetx.github.io/data.html  (S3 bucket starfish.data.spacetx)
# Each dataset ships:
#   s3_spot_table.csv         per-molecule spots (the input spSheaf needs)
#   s3_cell_by_gene.csv       segmented cell x gene counts
#   s3_mapped_cell_table.csv  per-cell metadata / mapping
#   <name>_Readme.txt         column definitions -- READ IT before writing a loader
#
# Usage:  bash experiments/spacetx/download.sh [all|BARISTASeq|MERFISH_Allen_Prototype|ExSeq|ISS|smFISH_Allen]
# Files land under $SPSHEAF_DATA/raw/spacetx/<dataset>/ (default: ./data/raw/spacetx/).
# Plain bash 3 (macOS default): no associative arrays.
set -eu
BASE="https://s3.amazonaws.com/starfish.data.spacetx/spacetx-website/data"
ROOT="${SPSHEAF_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/data}/raw/spacetx"
sel="${1:-all}"

readme_of() {
  case "$1" in
    BARISTASeq) echo BARISTASeq_Readme.txt ;;
    MERFISH_Allen_Prototype) echo MERFISH_Allen_Prototype_Readme.txt ;;
    ExSeq) echo ExSeq_Readme.txt ;;
    ISS) echo ISS_Readme.txt ;;
    smFISH_Allen) echo smFISH_Readme.txt ;;
    *) echo "unknown dataset: $1" >&2; exit 1 ;;
  esac
}

if [ "$sel" = "all" ]; then
  sets="BARISTASeq MERFISH_Allen_Prototype ExSeq ISS smFISH_Allen"
else
  sets="$sel"
fi

for ds in $sets; do
  mkdir -p "$ROOT/$ds"
  for f in "$(readme_of "$ds")" s3_spot_table.csv s3_mapped_cell_table.csv s3_cell_by_gene.csv; do
    url="$BASE/$ds/$f"; out="$ROOT/$ds/$f"
    if [ -s "$out" ]; then echo "have   $ds/$f"; continue; fi
    echo "fetch  $ds/$f"
    curl -fL --retry 3 -o "$out" "$url" || { echo "  FAILED: $url" >&2; rm -f "$out"; }
  done
done

echo
echo "== what arrived =="
for ds in $sets; do
  for f in "$ROOT/$ds"/*; do
    [ -f "$f" ] && printf "%-70s %10s bytes\n" "${f#$ROOT/}" "$(wc -c < "$f" | tr -d ' ')"
  done
done
echo
echo "== first lines of each spot table (check for x, y, gene columns and units) =="
for ds in $sets; do
  f="$ROOT/$ds/s3_spot_table.csv"; [ -s "$f" ] || continue
  echo "-- $ds"; head -3 "$f"
done
echo
echo "== first lines of each mapped cell table (look for a layer / region column) =="
for ds in $sets; do
  f="$ROOT/$ds/s3_mapped_cell_table.csv"; [ -s "$f" ] || continue
  echo "-- $ds"; head -3 "$f"
done
