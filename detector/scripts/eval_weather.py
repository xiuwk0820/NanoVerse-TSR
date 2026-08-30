#!/usr/bin/env python3
"""Evaluate detector on CCTSDB weather subsets (Table VI)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DETECTOR_DIR = ROOT / "detector"
DATA_DIR = ROOT / "data"
CCTSDB = DATA_DIR / "datasets" / "CCTSDB"
BASE_YAML = DETECTOR_DIR / "configs" / "cctsdb.yaml"
TMP_DIR = DETECTOR_DIR / "tmp_weather_eval"
DEFAULT_WEIGHTS = DATA_DIR / "weights" / "detector" / "cctsdb_yolo11s_world8_best.pt"


def weather_subset_dir(cctsdb_root: Path) -> Path:
    for name in ("weather_subsets", "测试集天气光照情况分类"):
        candidate = cctsdb_root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Weather subset folder not found under {cctsdb_root}. "
        "Expected `weather_subsets/` (see docs/DATASETS.md)."
    )


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def load_base_cfg(base_yaml: Path) -> dict:
    with base_yaml.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_subset(cond: str, weather_dir: Path) -> tuple[Path, Path]:
    subset_src = weather_dir / cond
    img_dst = TMP_DIR / cond / "images"
    lb_dst = TMP_DIR / cond / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lb_dst.mkdir(parents=True, exist_ok=True)

    for p in subset_src.iterdir():
        if not p.is_file():
            continue

        stem = p.stem
        img_path = CCTSDB / "test_img" / f"{stem}.jpg"
        if not img_path.exists():
            img_path = CCTSDB / "test_img" / f"{stem}.png"
        if not img_path.exists():
            continue

        target = img_dst / img_path.name
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(img_path.resolve())

        label_src = CCTSDB / "test_labels" / f"{stem}.txt"
        label_dst = lb_dst / f"{stem}.txt"
        if label_dst.exists() or label_dst.is_symlink():
            label_dst.unlink()
        if label_src.exists():
            label_dst.symlink_to(label_src.resolve())
        else:
            label_dst.write_text("", encoding="utf-8")
    return img_dst, lb_dst


def run_one(
    model: YOLO,
    cond: str,
    weather_dir: Path,
    base_cfg: dict,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> dict:
    img_dst, _ = build_subset(cond, weather_dir)
    cfg = dict(base_cfg)
    cfg["path"] = str((TMP_DIR / cond).resolve())
    cfg["train"] = str(img_dst.relative_to(TMP_DIR / cond))
    cfg["val"] = str(img_dst.relative_to(TMP_DIR / cond))
    cond_yaml = TMP_DIR / f"{cond}.yaml"
    with cond_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    metrics = model.val(
        data=str(cond_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        verbose=False,
    )
    return {
        "P": float(metrics.box.mp),
        "R": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCTSDB weather subset evaluation.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", type=Path, default=BASE_YAML)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "experiments" / "NanoVerse-TSR" / "weather_metrics.txt",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.weights = resolve_path(args.weights)
    args.data = resolve_path(args.data)
    args.output = resolve_path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    weather_dir = weather_subset_dir(CCTSDB)
    base_cfg = load_base_cfg(args.data)
    model = YOLO(str(args.weights))
    weather_order = ["sunny", "cloud", "night", "rain", "foggy", "snow"]

    lines = []
    for cond in weather_order:
        m = run_one(model, cond, weather_dir, base_cfg, args.imgsz, args.batch, args.workers, args.device)
        line = (
            f"{cond}: "
            f"P={m['P']*100:.2f}, "
            f"R={m['R']*100:.2f}, "
            f"mAP50={m['mAP50']*100:.2f}, "
            f"mAP50-95={m['mAP50_95']*100:.2f}"
        )
        print(line)
        lines.append(line)

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
