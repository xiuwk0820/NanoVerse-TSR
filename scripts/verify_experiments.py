#!/usr/bin/env python3
"""Run smoke tests for every NanoVerse-TSR experiment entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd or ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, out.strip()
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        return False, f"TIMEOUT after {timeout}s\n{out}"


def check_dataset_links() -> tuple[bool, str]:
    from nanoverse.paths import DATASETS_DIR

    issues: list[str] = []
    cctsdb = DATASETS_DIR / "CCTSDB"
    for sub in ("images/val", "images/train", "labels/val", "labels/train"):
        d = cctsdb / sub
        if not d.exists():
            issues.append(f"missing {d}")
            continue
        files = list(d.glob("*"))
        broken = sum(1 for f in files if not f.exists())
        if broken:
            issues.append(f"{sub}: {broken}/{len(files)} broken symlinks")

    tstiad = DATASETS_DIR / "TT100K_images"
    if not tstiad.exists():
        issues.append("missing TT100K_images -> TSTIAD_images symlink")

    test_json = DATASETS_DIR / "CLIP_new_test_data.json"
    if test_json.exists():
        data = json.loads(test_json.read_text(encoding="utf-8"))
        sample = DATASETS_DIR / data[0]["image"]
        if not sample.exists():
            alt = DATASETS_DIR / data[0]["image"].replace("TT100K_images/", "TSTIAD_images/", 1)
            if not alt.exists():
                issues.append(f"TSTIAD image missing: {data[0]['image']}")

    if issues:
        return False, "; ".join(issues)
    return True, "dataset links OK"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify NanoVerse-TSR experiments.")
    parser.add_argument("--quick", action="store_true", help="Skip long eval runs (detector val / classify).")
    args = parser.parse_args()

    results: list[tuple[str, bool, str]] = []

    ok, out = run([PYTHON, str(ROOT / "scripts" / "verify_install.py")])
    results.append(("install", ok, "passed" if ok else out[-500:]))

    sys.path.insert(0, str(ROOT))
    ok, msg = check_dataset_links()
    results.append(("dataset_links", ok, msg))

    ok, out = run(
        [
            PYTHON,
            str(ROOT / "pipeline" / "infer.py"),
            "data/datasets/TT100K_detection_221cls/images/test/10056.jpg",
        ],
        timeout=120,
    )
    results.append(("pipeline_infer", ok, "3 detections" if "detections" in out else out[-300:]))

    if not args.quick:
        ok, out = run(
            [
                PYTHON,
                str(ROOT / "detector" / "scripts" / "run.py"),
                "val",
                "--weights",
                "data/weights/detector/cctsdb_yolo11s_world8_best.pt",
                "--data",
                "detector/configs/cctsdb.yaml",
                "--batch",
                "8",
            ],
            timeout=300,
        )
        summary = "mAP50" if "mAP50" in out else out[-300:]
        results.append(("detector_cctsdb_val", ok, summary.splitlines()[-3:] if ok else out[-400:]))

        ok, out = run(
            [
                PYTHON,
                str(ROOT / "detector" / "scripts" / "run.py"),
                "weather",
                "--batch",
                "8",
            ],
            timeout=600,
        )
        results.append(("detector_weather", ok, "6 subsets" if "sunny:" in out else out[-400:]))

    ok, out = run(
        [
            PYTHON,
            "-c",
            "import json; from pathlib import Path; "
            "import sys; sys.path.insert(0,'classifier'); sys.path.insert(0,'.'); "
            "from utils.dataloader import ClipDataset; from utils.utils import get_configs; "
            "root=Path('data/datasets'); cfg=get_configs('openai/VIT-B-16'); "
            "val=json.load(open(root/'CLIP_new_test_data.json')); "
            "ds=ClipDataset([cfg['input_resolution']]*2, val[:8], str(root), False, False); "
            "img,cap=ds[0]; print('OK', img.shape, len(val))",
        ],
        timeout=60,
    )
    results.append(("classifier_dataloader", ok, out.splitlines()[-1] if ok else out[-300:]))

    print(f"NanoVerse-TSR experiment verification ({ROOT})\n")
    failed = 0
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            failed += 1
        print(f"[{status}] {name}: {detail}")

    if failed:
        print(f"\n{failed} check(s) failed.")
        sys.exit(1)
    print("\nAll experiment checks passed.")


if __name__ == "__main__":
    main()
