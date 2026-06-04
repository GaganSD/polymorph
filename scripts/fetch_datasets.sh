#!/usr/bin/env bash
# Fetch the Polymorph training/benchmark corpora into ./data/raw/.
# Idempotent: skips anything already present. Safe to re-run.
#
#   bash scripts/fetch_datasets.sh
#
# Datasets:
#   1. TrainTicket microservice logs+traces  (Zenodo 6979726, CC-BY-4.0)  -- no auth
#   2. server-logs (Apache access logs)       (Kaggle vishnu0399, CC0)     -- public
#   3. logs-dataset kernel                     (Kaggle adepvenugopal)       -- NEEDS AUTH
#
# Kaggle auth (only needed for #3, and sometimes #2): set up ONE of:
#   - export KAGGLE_API_TOKEN=...                (from kaggle.com/settings/api)
#   - ~/.kaggle/kaggle.json                      (downloaded token file)
#   - kaggle auth login                          (interactive OAuth)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW/trainticket" "$RAW/server_logs" "$RAW/kaggle_logs" "$ROOT/data/bench" "$ROOT/data/distilled"

# Ensure kaggle CLI is available (installed via uv tool).
if ! command -v kaggle >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v kaggle >/dev/null 2>&1; then
  echo "[setup] installing kaggle CLI via uv tool ..."
  uv tool install kaggle >/dev/null 2>&1 || pip install --user kaggle
  export PATH="$HOME/.local/bin:$PATH"
fi

# 1. TrainTicket (Zenodo, CC-BY-4.0) -----------------------------------------
TT_ZIP="$RAW/trainticket/trainticket.zip"
TT_DIR="$RAW/trainticket/anomalies_microservice_trainticket_version_configurations"
if [ -d "$TT_DIR" ]; then
  echo "[1/3] TrainTicket already unpacked -> skip"
else
  echo "[1/3] downloading TrainTicket (153 MB, CC-BY-4.0) from Zenodo ..."
  curl -fSL -o "$TT_ZIP" \
    "https://zenodo.org/api/records/6979726/files/anomalies_microservice_trainticket_version_configurations.zip/content"
  ( cd "$RAW/trainticket" && unzip -o -q trainticket.zip )
  echo "      unpacked to $TT_DIR"
fi

# 2. server-logs (Kaggle dataset, CC0) ---------------------------------------
if [ -f "$RAW/server_logs/logfiles.log" ]; then
  echo "[2/3] server-logs already present -> skip"
else
  echo "[2/3] downloading kaggle dataset vishnu0399/server-logs (CC0) ..."
  if kaggle datasets download vishnu0399/server-logs -p "$RAW/server_logs" 2>/dev/null; then
    ( cd "$RAW/server_logs" && unzip -o -q server-logs.zip )
    echo "      unpacked logfiles.log"
  else
    echo "      [warn] kaggle download failed (auth?). See header for auth setup."
  fi
fi

# 3. logs-dataset kernel (Kaggle, NEEDS AUTH) --------------------------------
if [ -n "$(ls -A "$RAW/kaggle_logs" 2>/dev/null)" ]; then
  echo "[3/3] kaggle logs-dataset already pulled -> skip"
else
  echo "[3/3] pulling kaggle kernel adepvenugopal/logs-dataset (needs auth) ..."
  if kaggle kernels pull adepvenugopal/logs-dataset -p "$RAW/kaggle_logs" 2>/dev/null; then
    echo "      pulled kernel into $RAW/kaggle_logs"
  else
    echo "      [SKIP] Kaggle kernels require authentication. To fetch it:"
    echo "        export KAGGLE_API_TOKEN=...   # from https://www.kaggle.com/settings/api"
    echo "        kaggle kernels pull adepvenugopal/logs-dataset -p $RAW/kaggle_logs"
  fi
fi

echo ""
echo "done. Corpora under data/raw/. See data/DATA_CARD.md for licenses + attribution."
