#!/usr/bin/env bash
# VoxCustomer — environment bootstrap
# Creates a local Python venv and installs all dependencies.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PY_BIN="${PYTHON:-python3}"
VENV_DIR=".venv"

echo "▸ Creating virtual environment at $VENV_DIR"
"$PY_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "▸ Upgrading pip"
python -m pip install --upgrade pip wheel setuptools

echo "▸ Installing project dependencies"
pip install -r requirements.txt

echo ""
echo "✓ Environment ready."
echo ""
echo "Activate it with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then launch the app with:"
echo "  streamlit run app.py"
