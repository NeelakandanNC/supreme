#!/usr/bin/env bash
# Create the project virtualenv.
#
# Pinned to Python 3.12: as of writing, PyTorch does not publish wheels for
# 3.14, which is the default `python3` on this machine.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python3.12}
VENV=${VENV:-.venv}

if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PY" "$VENV"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  uv pip install -r requirements.txt
else
  "$PY" -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
fi

echo
echo "Done. Activate with:  source $VENV/bin/activate"
echo "Then smoke-test with: python wm.py study -c configs/tmaze_fast.yaml --backbones lstm,gru --no-control"
