#!/usr/bin/env python3
"""Sanity checks for NanoVerse-TSR installation."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(name: str) -> None:
    importlib.import_module(name)
    print(f"[OK] import {name}")


def check_path(label: str, path: Path, required: bool = True) -> None:
    if path.exists():
        print(f"[OK] {label}: {path}")
    elif required:
        print(f"[MISSING] {label}: {path}")
    else:
        print(f"[SKIP] {label}: {path} (optional)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify NanoVerse-TSR installation.")
    parser.add_argument(
        "--skip-weights",
        action="store_true",
        help="Skip checkpoint file checks (useful for CI without downloaded weights).",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip dataset path checks.",
    )
    args = parser.parse_args()

    print(f"NanoVerse-TSR root: {ROOT}\n")

    for mod in ["yaml", "numpy", "PIL", "torch", "tqdm", "ultralytics"]:
        check(mod)

    from nanoverse.paths import DATASETS_DIR, WEIGHTS_DIR

    if not args.skip_data:
        check_path("TT100K detection dataset", DATASETS_DIR / "TT100K_detection_221cls")
        check_path("TSTIAD images", DATASETS_DIR / "TSTIAD_images")
        check_path("CCTSDB dataset", DATASETS_DIR / "CCTSDB")

    if not args.skip_weights:
        check_path("Classifier weight", WEIGHTS_DIR / "classifier" / "enhance_token_91.6_best.pth")
        check_path("Detector weight (world8)", WEIGHTS_DIR / "detector" / "cctsdb_yolo11s_world8_best.pt")

    sys.path.insert(0, str(ROOT / "classifier"))
    check("clip")
    check("model")

    print("\nAll basic checks passed.")


if __name__ == "__main__":
    main()
