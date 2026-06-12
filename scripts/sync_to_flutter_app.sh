#!/usr/bin/env bash
# Copy shipped ML artifacts into the Flutter app bundle.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${VIVA_FLUTTER_APP:-$ROOT/../../Flutter/viva-app}"
DEST="$APP/assets/ai"

if [[ ! -d "$APP" ]]; then
  echo "Flutter app not found at: $APP" >&2
  echo "Set VIVA_FLUTTER_APP to your viva-app path." >&2
  exit 1
fi

mkdir -p "$DEST"

for f in viva_ai.tflite feature_metadata.json; do
  src="$ROOT/models/final/$f"
  if [[ ! -f "$src" ]]; then
    echo "Missing $src — run training + convert_tflite.py first." >&2
    exit 1
  fi
  cp "$src" "$DEST/$f"
  echo "Copied $f → $DEST/"
done

# Refresh Dart policy from evaluation metrics (Phase 1).
python3 "$ROOT/scripts/export_ml_policy.py"

echo "Done. Rebuild the Flutter app (hot restart is not enough for new assets)."
