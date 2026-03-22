# Conditional PixelCNN++ — Image Generation & Classification

A class-conditional extension of PixelCNN++ that does two things simultaneously: **generates category-specific images** and **classifies images** — trained on a custom 4-class dataset as part of UBC CPEN 455 (Deep Learning).

**Results:** ~79% classification accuracy · FID ~24 · 300 epochs · 70+ experimental runs

---

## What this is

PixelCNN++ is an autoregressive generative model that learns to model pixel distributions. This project extends it with:

- **Class conditioning** via one-hot input concatenation — the model sees the class label from the first layer
- **Classifier head** — global average pooling → fully-connected layer → softmax, trained jointly with generation
- **Joint loss** combining negative log-likelihood (generation) and cross-entropy (classification)

The result is a single model that can both *generate* new images for a given class and *classify* unseen images.

---

## Architecture overview

```
Input image (3×32×32) + one-hot class label
        ↓
  [Concatenated input: 7×32×32]
        ↓
  PixelCNN++ backbone
  (masked convolutions, residual blocks, U-Net-style up/downsampling)
        ↓
    ┌───────────────────┐
    │                   │
 Logistic mixture    Classifier head
 (generation output) (global avg pool → FC → softmax)
```

**Key design choices:**
- Input concatenation for class conditioning (vs. learned bias per layer — tested both, concatenation won marginally)
- Discretized mixture of logistics output (not 256-way softmax per pixel)
- `nr_filters=40`, `nr_resnet=1`, `nr_logistic_mix=15` — found via ablation to be the sweet spot

---

## Results

| Metric | Value |
|---|---|
| Classification accuracy (validation) | 79.38% |
| Classification accuracy (test / leaderboard) | 79.15% |
| F1 score (test) | 79.02% |
| FID score | 23.69 |
| Training epochs | 300 |

The grading rubric required >75% accuracy for bonus marks and FID <30 for full generation score — both achieved.

---

## Training

```bash
python pcnn_train.py \
  --batch_size 48 \
  --nr_resnet 1 \
  --nr_filters 40 \
  --nr_logistic_mix 15 \
  --lr 0.001 \
  --lr_decay 0.5 \
  --max_epochs 300 \
  --dataset cpen455 \
  --en_wandb True
```

Trained on Google Colab (T4/L4 GPUs). MPS backend (Apple M1) tested locally but showed ~35% throughput of T4.

**Dataset:** 4,160 labeled training images across 4 classes, 520 validation, 520 test (32×32 RGB)

---

## Ablation study highlights

Over 70 W&B runs varying:

- **Filters (nr_filters):** Reducing from 80→40 improved both accuracy and FID — fewer filters generalized better on this dataset size
- **Mixture components:** M=15 beat M=5 significantly on FID; accuracy less sensitive
- **Conditioning method:** Input concatenation vs. learned per-layer bias — concatenation was simpler and performed slightly better
- **Epochs:** Accuracy plateaued ~250 epochs; generation (BPD) continued improving. Overtraining past ~300 hurt FID.

---

## Evaluation

**Classification:**
```bash
python classification_evaluation.py --mode validation
```

**Generation + FID:**
```bash
python generation_evaluation.py --ref_data_dir data/test
```

Model checkpoint expected at `models/conditional_pixelcnn.pth`.

---

## Setup

```bash
conda create -n pcnn python=3.10
conda activate pcnn
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

---

## Files

| File | Description |
|---|---|
| `model.py` | PixelCNN++ backbone + classifier head |
| `layers.py` | Masked conv layers, gated ResNet blocks |
| `pcnn_train.py` | Training loop with joint loss |
| `dataset.py` | Custom dataset loader (CSV-based) |
| `utils.py` | Loss functions, sampling, image saving |
| `classification_evaluation.py` | Accuracy evaluation script |
| `generation_evaluation.py` | FID evaluation + image generation |

---

## Tech stack

`PyTorch` · `W&B` · `pytorch-fid` · `Google Colab (T4/L4)` · `Python 3.10`

---

*UBC CPEN 455 — Deep Learning · Winter 2024*
