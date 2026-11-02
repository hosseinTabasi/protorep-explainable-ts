#!/usr/bin/env bash
# Fetch the public datasets into $INDUSTRIAL_TS_DATA (default: ./data/raw).
set -euo pipefail
DEST="${INDUSTRIAL_TS_DATA:-$(dirname "$0")/../data/raw}"
mkdir -p "$DEST"; cd "$DEST"
[ -d SKAB ] || git clone --depth 1 https://github.com/waico/SKAB.git
[ -d TEP_Braatz ] || git clone --depth 1 https://github.com/camaramm/tennessee-eastman-profBraatz.git TEP_Braatz
echo "Done. export INDUSTRIAL_TS_DATA=$(pwd)"
