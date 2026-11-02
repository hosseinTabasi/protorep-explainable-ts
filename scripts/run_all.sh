#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
python scripts/run_experiments.py e1 e2 e3 e4b e5
python scripts/make_report.py
