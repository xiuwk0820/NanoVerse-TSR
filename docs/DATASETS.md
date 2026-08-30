# Datasets

Images are **not** in this repo. JSON captions and category text **are** included.

Our prepared dataset packs (TSTIAD crops, ready-to-use TT100K / CCTSDB layouts) are **coming soon**. You can already download the official sources and arrange them as below.

Place raw data under `data/datasets/` to match the configs.

---

## TT100K (detection, 221 classes)

**Use:** Stage-1 training and Table II.

**Official page:** [Tsinghua-Tencent 100K](http://cg.cs.tsinghua.edu.cn/traffic-sign/)

**Expected layout:**

```
data/datasets/TT100K_detection_221cls/
├── images/
│   ├── train/    # 6,104 images
│   └── test/     # 3,071 images
└── labels/
    ├── train/
    └── test/
```

**Config:** `detector/configs/tt100k.yaml`

---

## TSTIAD (classifier, image–text pairs)

**Use:** Stage-2 contrastive training (Section III-C).

**Included here (JSON only):**

| File | Description |
|------|-------------|
| `data/datasets/CLIP_new_train_data.json` | Training pairs |
| `data/datasets/CLIP_new_test_data.json` | Test pairs |
| `data/datasets/CLIP_train_data_en.json` | English captions (train) |
| `data/datasets/CLIP_test_data_en.json` | English captions (test) |
| `classifier/project/TSR_new.json` | 220+ category text descriptions |

**Images** (cropped patches referenced as `TT100K_images/<id>.jpg`):

```
data/datasets/TSTIAD_images/
```

Build TSTIAD by cropping TT100K boxes and pairing them with GB5768-2022 text (paper Fig. 3). A packaged crop set is **coming soon**.

---

## CCTSDB2021 (detection + weather)

**Use:** Table III generalization and Table VI weather splits.

**Official repo:** [csust7zhangyu/CCTSDB2021](https://github.com/csust7zhangyu/CCTSDB2021)

**Expected layout:**

```
data/datasets/CCTSDB/
├── images/train/
├── images/val/
├── test_img/
├── test_labels/
└── weather_subsets/          # or the original Chinese folder names
    ├── sunny/
    ├── cloud/
    ├── night/
    ├── rain/
    ├── foggy/
    └── snow/
```

**Config:** `detector/configs/cctsdb.yaml`

---

## GTSRB (optional classifier transfer)

**Official page:** [GTSRB @ INI](https://benchmark.ini.rub.de/gtsrb_news.html)

Used for the 97.3% Top-1 number in Table IV. Not required for the default pipeline.

---

## Sanity check

```bash
python scripts/verify_install.py
```

Dataset folders should print `[OK]` once images are in place.
