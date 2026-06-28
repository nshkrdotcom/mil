#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_BACKEND="${TORCH_BACKEND:-cu128}"

uv venv --python "${PYTHON_VERSION}" --allow-existing .venv

uv pip install \
  --python .venv/bin/python \
  --torch-backend "${TORCH_BACKEND}" \
  --exact \
  -e ".[models,app,viz,sae,evidence,dev]"

.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/gpu_check.py
