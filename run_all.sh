#!/bin/sh
# Reproduce every number, table and figure of the paper from the public NASA archive.
# Usage (from this folder):  sh run_all.sh
# Duration: about 30 minutes on a 10-core laptop (step 2 dominates).
set -e
cd "$(dirname "$0")"
python3 data/download_data.py              # NASA archive -> data/raw/mill.mat (checksum verified)
python3 src/01_prepare_data.py             # mill.mat -> data/processed/features.csv
python3 src/02_run_experiments.py          # nested cross-validation -> results/raw/*.csv
python3 src/03_analyse.py                  # -> results/tables/*.csv
python3 src/04_make_figures.py             # -> results/figures/*.pdf, *.png
