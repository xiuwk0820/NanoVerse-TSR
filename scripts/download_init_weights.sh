#!/usr/bin/env bash
# Download third-party init weights (not paper checkpoints).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DATA="${ROOT}/classifier/model_data"
mkdir -p "${MODEL_DATA}" "${ROOT}/data/weights/detector" "${ROOT}/data/weights/classifier"

echo "==> CLIP ViT-B/16 (OpenAI)"
curl -L --fail -o "${MODEL_DATA}/ViT-B-16-OpenAI.pth" \
  "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"

echo "==> YOLO11s (Ultralytics)"
curl -L --fail -o "${ROOT}/yolo11s.pt" \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"

echo "Done. Paper checkpoints still go in data/weights/ — see docs/WEIGHTS.md"
