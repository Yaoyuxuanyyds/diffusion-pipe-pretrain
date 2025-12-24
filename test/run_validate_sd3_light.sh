#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_dir> [ema_shadow_path]"
  echo "Example: $0 /path/to/sd3_light/step100 /path/to/ema_shadow.pt"
  exit 1
fi

MODEL_DIR="$1"
EMA_SHADOW="${2:-}"

ARGS=(
  --model_dir "${MODEL_DIR}"
  --prompt "a photo of a cute cat"
  --height 512
  --width 512
  --steps 12
  --seed 1234
  --dtype float16
  --output_dir sd3_light_validation_outputs
)

if [[ -n "${EMA_SHADOW}" ]]; then
  ARGS+=(--ema_shadow "${EMA_SHADOW}")
fi

python test/validate_sd3_light.py "${ARGS[@]}"
