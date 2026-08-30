# Stage 2 — Cross-Modal Contrastive Classifier

ViT image encoder + Rule-BERT text encoder, trained on TSTIAD.

## Key Files

| File | Role |
|------|------|
| `train.py` | Train on TSTIAD JSON under `data/datasets/` |
| `eval_classify.py` | Top-1 accuracy on test split |
| `model.py` | `CLIP_TSR` — detect then classify pipeline |
| `clip.py` | CLIP inference wrapper |
| `project/TSR_new.json` | 220+ sign text descriptions |

## Commands

```bash
# Train classifier
cd classifier && python train.py

# Evaluate Top-1 (expect ~91.6% on TT100K crops)
cd classifier && python eval_classify.py
```

Default checkpoint: `../data/weights/classifier/enhance_token_91.6_best.pth`
