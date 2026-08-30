"""Central path helpers for NanoVerse-TSR."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("NANOVERSE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


ROOT = project_root()
DETECTOR_DIR = ROOT / "detector"
CLASSIFIER_DIR = ROOT / "classifier"
DATA_DIR = ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
WEIGHTS_DIR = DATA_DIR / "weights"
EXPERIMENTS_DIR = DATA_DIR / "experiments"

# Legacy data locations (kept as symlinks during migration)
LEGACY_YOLO_TS = ROOT.parent / "YOLO-TS"

DEFAULT_DETECTOR_WEIGHT = WEIGHTS_DIR / "detector" / "cctsdb_yolo11s_world8_best.pt"
DEFAULT_CLASSIFIER_WEIGHT = WEIGHTS_DIR / "classifier" / "enhance_token_91.6_best.pth"
DEFAULT_YOLO_DETECTOR_WEIGHT = WEIGHTS_DIR / "detector" / "yolo_detector_best.pt"
