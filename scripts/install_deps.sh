#!/usr/bin/env bash
# Install PyTorch (CPU wheels) then project dependencies.
# For GPU, install torch from https://pytorch.org/get-started/locally/ instead of the first line.

set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
