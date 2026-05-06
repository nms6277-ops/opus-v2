#!/usr/bin/env bash
# Local install / setup script for opus.
#
# Run this once after cloning:
#   bash scripts/install_local.sh
#
# It creates a venv in ./.venv, installs deps, copies .env.example -> .env
# if missing, and prints next-step commands.

set -euo pipefail

cd "$(dirname "$0")/.."

PY=${PY:-python3.11}
if ! command -v "$PY" >/dev/null 2>&1; then
    PY=python3
fi

echo "[opus] using $($PY --version) at $(command -v "$PY")"

if [[ ! -d .venv ]]; then
    echo "[opus] creating venv in .venv"
    "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -e '.[dev,ml]'

if [[ ! -f .env ]]; then
    echo "[opus] creating .env from .env.example"
    cp .env.example .env
    echo "[opus] EDIT .env to set guards before running LIVE mode"
fi

mkdir -p data logs models

echo
echo "[opus] install done."
echo
echo "Next steps:"
echo "  1. Activate venv:   source .venv/bin/activate"
echo "  2. Edit .env (at least OPUS_DAILY_LOSS_LIMIT_USD, OPUS_MAX_POSITION_USD)"
echo "  3. Start backend:   python -m backend.main"
echo "  4. Open UI via SSH tunnel from your laptop:"
echo "       ssh -L 8080:127.0.0.1:8080 your-user@your-vps"
echo "       then browse http://127.0.0.1:8080"
