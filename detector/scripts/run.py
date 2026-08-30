#!/usr/bin/env python3
"""Train / validate / weather-eval entry point for NanoVerse-TSR detector."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanoverse.paths import DATASETS_DIR, DETECTOR_DIR, WEIGHTS_DIR  # noqa: E402

DETECTOR_SCRIPTS = DETECTOR_DIR / "scripts"
if str(DETECTOR_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DETECTOR_SCRIPTS))


def ensure_world_text_dependencies() -> None:
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        import packaging

        shim = types.ModuleType("pkg_resources")
        shim.packaging = packaging
        sys.modules["pkg_resources"] = shim

    try:
        import ftfy  # noqa: F401
    except ModuleNotFoundError:
        shim = types.ModuleType("ftfy")
        shim.fix_text = lambda text: text
        sys.modules["ftfy"] = shim


ensure_world_text_dependencies()

from ultralytics import YOLO, YOLOWorld  # noqa: E402


DEFAULT_DATA = DETECTOR_DIR / "configs" / "cctsdb.yaml"
DEFAULT_MODEL_CFG = DETECTOR_DIR / "models" / "yolo11s-p2.yaml"
DEFAULT_RUN_NAME = "cctsdb_yolo11s_p2"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "detector" / "cctsdb_yolo11s_world8_best.pt"


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def materialize_data_yaml(data_yaml: Path) -> Path:
    """Write a temp dataset yaml with absolute paths (Ultralytics resolves relative paths against its global datasets_dir)."""
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    raw_path = Path(cfg["path"])
    if raw_path.is_absolute():
        dataset_root = raw_path
    else:
        name = raw_path.name
        if name in {"CCTSDB", "TT100K_detection_221cls", "TT100K_3cls"}:
            dataset_root = (DATASETS_DIR / name).resolve()
        else:
            dataset_root = (data_yaml.parent / raw_path).resolve()

    cfg["path"] = str(dataset_root)
    out = DETECTOR_DIR / "tmp_data" / f"{data_yaml.stem}_resolved.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return out


def load_names(data_yaml: Path) -> list[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    names = data.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return names
    raise ValueError(f"Could not parse class names from {data_yaml}")


def is_world_yaml(model_path: Path) -> bool:
    if model_path.suffix not in {".yaml", ".yml"} or not model_path.exists():
        return False
    with model_path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    head = cfg.get("head", [])
    return any(isinstance(layer, list) and len(layer) >= 3 and layer[2] == "WorldDetect" for layer in head)


def build_model(model_path: Path, data_yaml: Path) -> YOLO | YOLOWorld:
    model = YOLOWorld(str(model_path)) if is_world_yaml(model_path) else YOLO(str(model_path))
    if isinstance(model, YOLOWorld):
        model.set_classes(load_names(data_yaml))
    return model


def cmd_train(args: argparse.Namespace) -> None:
    model = build_model(args.model, args.data)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        device=args.device,
    )


def cmd_val(args: argparse.Namespace) -> None:
    model = build_model(args.weights, args.data)
    model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        split=args.split,
        device=args.device,
    )


def cmd_weather(args: argparse.Namespace) -> None:
    import eval_weather

    argv = [
        "eval_weather.py",
        "--weights",
        str(args.weights),
        "--data",
        str(args.data),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--output",
        str(args.output),
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        eval_weather.main()
    finally:
        sys.argv = old_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NanoVerse-TSR detector train/val/weather eval.")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Train detector.")
    p_train.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p_train.add_argument("--model", type=Path, default=DEFAULT_MODEL_CFG)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--batch", type=int, default=64)
    p_train.add_argument("--workers", type=int, default=8)
    p_train.add_argument("--project", type=Path, default=DETECTOR_DIR / "runs")
    p_train.add_argument("--name", type=str, default=DEFAULT_RUN_NAME)
    p_train.add_argument("--device", type=str, default="0")
    p_train.set_defaults(func=cmd_train)

    p_val = sub.add_parser("val", help="Validate detector.")
    p_val.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p_val.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p_val.add_argument("--imgsz", type=int, default=640)
    p_val.add_argument("--batch", type=int, default=16)
    p_val.add_argument("--workers", type=int, default=4)
    p_val.add_argument("--split", type=str, default="val")
    p_val.add_argument("--device", type=str, default="0")
    p_val.set_defaults(func=cmd_val)

    p_weather = sub.add_parser("weather", help="Evaluate on CCTSDB weather subsets.")
    p_weather.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p_weather.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p_weather.add_argument("--imgsz", type=int, default=640)
    p_weather.add_argument("--batch", type=int, default=16)
    p_weather.add_argument("--workers", type=int, default=4)
    p_weather.add_argument("--device", type=str, default="0")
    p_weather.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "experiments" / "NanoVerse-TSR" / "weather_metrics.txt",
    )
    p_weather.set_defaults(func=cmd_weather)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.data = resolve_path(args.data)
    if args.mode in {"train", "val"}:
        args.data = materialize_data_yaml(args.data)
    args.model = resolve_path(getattr(args, "model", DEFAULT_MODEL_CFG))
    if hasattr(args, "weights"):
        args.weights = resolve_path(args.weights)
    if hasattr(args, "project"):
        args.project = resolve_path(args.project)
    if hasattr(args, "output"):
        args.output = resolve_path(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
