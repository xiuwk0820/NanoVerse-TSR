#!/usr/bin/env python3
"""Evaluate Top-1 classification accuracy on TSTIAD test split."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

CLASSIFIER_DIR = Path(__file__).resolve().parent
ROOT = CLASSIFIER_DIR.parent
if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import CLIP_TSR  # noqa: E402


def main() -> None:
    from nanoverse.paths import DATASETS_DIR

    test_json = DATASETS_DIR / "CLIP_new_test_data.json"
    project_json = CLASSIFIER_DIR / "project" / "TSR_new.json"

    model = CLIP_TSR(project_path=str(project_json))

    with test_json.open(encoding="utf-8") as fp:
        json_data = json.load(fp)

    def resolve_image(relative: str) -> Path:
        path = DATASETS_DIR / relative
        if path.exists():
            return path
        # Legacy JSON entries use TT100K_images/; images live under TSTIAD_images/
        alt = relative.replace("TT100K_images/", "TSTIAD_images/", 1)
        return DATASETS_DIR / alt

    correct = 0
    total = 0
    for label in tqdm(json_data, desc="Evaluating"):
        image_path = resolve_image(label["image"])
        image = Image.open(image_path)
        _, _, result = model.classify(image)
        if result == label["caption"][0]:
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    print(f"Top-1 Accuracy: {acc * 100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
