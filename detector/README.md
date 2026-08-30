# Stage 1 — Semantic-Visual Fusion Detector

RepVL-PAN + P2 small-object head built on YOLO11 / YOLO-World.

## Models

| File | Description |
|------|-------------|
| `models/yolo11s-world.yaml` | RepVL-PAN (`WorldDetect`) + P2 head — **default** |
| `models/yolo11s-p2.yaml` | P2 head only (visual baseline) |

## Commands

```bash
# Train
python detector/scripts/run.py train \
  --model detector/models/yolo11s-world.yaml \
  --data detector/configs/cctsdb.yaml

# Validate
python detector/scripts/run.py val \
  --weights data/weights/detector/cctsdb_yolo11s_world8_best.pt

# Weather subsets (Table VI)
python detector/scripts/run.py weather
```

Configs: `configs/cctsdb.yaml`, `configs/tt100k.yaml`
