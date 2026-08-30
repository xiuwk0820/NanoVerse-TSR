#!/usr/bin/env bash
# NanoVerse-TSR — one-line environment setup
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> NanoVerse-TSR setup"
echo "    Root: $ROOT"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

echo "==> Running sanity checks..."
python scripts/verify_install.py --skip-weights --skip-data

echo ""
echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Download datasets  → docs/DATASETS.md"
echo "  2. Download weights   → docs/WEIGHTS.md"
echo "  3. Run inference      → python pipeline/infer.py <image>"
