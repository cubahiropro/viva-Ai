#!/usr/bin/env bash
# One-time Python env for viva-Ai training / conversion.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Ready. Activate with: source $ROOT/.venv/bin/activate"
echo "Convert model:       python scripts/convert_tflite.py --mode int8"
echo "Sync to Flutter app: bash scripts/sync_to_flutter_app.sh"
