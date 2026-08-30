#!/usr/bin/env python3
"""End-to-end NanoVerse-TSR inference: detect traffic signs then classify with CLIP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_DIR = ROOT / "classifier"
if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

from model import CLIP_TSR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="NanoVerse-TSR inference")
    parser.add_argument("image", type=Path, help="Input image or directory")
    parser.add_argument("--yolo-weight", type=Path, default=None)
    parser.add_argument("--clip-weight", type=Path, default=None)
    parser.add_argument("--project-json", type=Path, default=CLASSIFIER_DIR / "project" / "TSR_new.json")
    args = parser.parse_args()

    kwargs = {"project_path": str(args.project_json)}
    if args.yolo_weight:
        kwargs["yolo_path"] = str(args.yolo_weight)
    model = CLIP_TSR(**kwargs)

    image_path = args.image
    if image_path.is_dir():
        images = sorted(
            p for p in image_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    else:
        images = [image_path]

    for img in images:
        _, results = model.scan(str(img))
        print(f"\n{img.name}: {len(results)} detections")
        for det in results:
            print(f"  {det[5]} @ [{det[1]}, {det[2]}, {det[3]}, {det[4]}]")


if __name__ == "__main__":
    main()
