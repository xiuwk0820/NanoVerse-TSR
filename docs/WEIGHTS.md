# Weights

Paper checkpoints are **coming soon**. This repo does not ship `.pt` / `.pth` files.

When they are hosted, put them under `data/weights/` and run `python scripts/verify_install.py`.

---

## 1. Paper checkpoints — coming soon

| Checkpoint | Save as | Paper | Status |
|------------|---------|-------|--------|
| RepVL-PAN + P2 (CCTSDB) | `data/weights/detector/cctsdb_yolo11s_world8_best.pt` | Table III | Coming soon |
| P2 baseline (CCTSDB) | `data/weights/detector/cctsdb_yolo11s_p2_best.pt` | Ablation | Coming soon |
| YOLO detector (end-to-end) | `data/weights/detector/yolo_detector_best.pt` | `pipeline/infer.py` | Coming soon |
| Rule-BERT + ViT (91.6% Top-1) | `data/weights/classifier/enhance_token_91.6_best.pth` | Table IV | Coming soon |

Until the Release is up, train from scratch (Section 3) or use your local copies.

---

## 2. Public init weights (available now)

Third-party files. Do not commit them.

| File | Used by | Official download |
|------|---------|-------------------|
| `classifier/model_data/ViT-B-16-OpenAI.pth` | `classifier/train.py` (`phi = openai/VIT-B-16`) | [OpenAI CLIP ViT-B/16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) — convert / rename to `.pth` if needed |
| `classifier/model_data/ViT-B-32-OpenAI.pth` | optional `openai/VIT-B-32` | [OpenAI CLIP ViT-B/32](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) |
| `yolo11s.pt` | detector train | [ultralytics/assets YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) — Ultralytics also auto-downloads |
| BERT text encoder | Rule-BERT init | Hugging Face [`bert-base-chinese`](https://huggingface.co/bert-base-chinese) (auto-download) |

```bash
bash scripts/download_init_weights.sh
```

---

## 3. Train instead of waiting for paper weights

```bash
python detector/scripts/run.py train \
  --model detector/models/yolo11s-world.yaml \
  --data detector/configs/cctsdb.yaml

cd classifier && python train.py
```

---

## Verify

```bash
python scripts/verify_install.py --skip-weights   # until the Release is up
python scripts/verify_install.py                  # after weights are in place
```
