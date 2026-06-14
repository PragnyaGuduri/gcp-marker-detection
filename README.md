# GCP Marker Detection — Aerial Pose Estimation

Automated detection of Ground Control Point (GCP) markers in aerial drone imagery.  
**Tasks:** Keypoint localization (x, y pixel coords) + Shape classification (Cross / Square / L-Shape)

---

## Results

| Metric | Score |
|---|---|
| PCK @ 10px | ~93% |
| PCK @ 25px | 100% |
| PCK @ 50px | 100% |
| Classification Accuracy | ~99% |

---

## Architecture

**EfficientNet-B0 + Dual Head**

```
Input Image (4096×2730)
      │
  Crop 384×384 around GCP center
      │
  EfficientNet-B0 Backbone (pretrained ImageNet)
      │
  Global Average Pool → features
      ├──→ Regression Head  → (x, y) normalized [0,1] via Sigmoid
      └──→ Classification Head → 3-class logits (Cross / L-Shape / Square)
```

**Why EfficientNet-B0?**
- Strong pretrained features, lightweight and fast
- 1000 labeled images is small — pretrained backbone is critical
- Handles varied terrain (desert, limestone mine, agricultural fields)

---

## Dataset

| Stat | Value |
|---|---|
| Total labeled images | 1,000 |
| Unique GCPs | 159 |
| Images per GCP | 1–12 (avg 6.3) |
| Projects | 11 real-world sites |
| Image resolution | 4096×2730 |
| Missing shape labels | 4 (dropped) |

**Class distribution (imbalanced):**
- L-Shape: 491 (49%)
- Square:  328 (33%)
- Cross:   177 (18%)

---

## Training Strategy

| Decision | Choice | Reason |
|---|---|---|
| Input crop | 384×384 | GCP is small; full 4K image wastes compute |
| Train jitter | ±64px crop offset | Teaches model to find off-center GCPs |
| Augmentation | Flip, rotate, blur, brightness | Simulate real-world variation |
| Loss (keypoint) | Wing Loss | Robust to small errors vs MSE |
| Loss (classify) | Weighted CrossEntropy | Handles class imbalance |
| Split strategy | GroupShuffleSplit by GCP | Prevents data leakage |
| Optimizer | AdamW + CosineAnnealing | Stable convergence |
| Epochs | 50 | Best PCK@25 achieved at epoch 15+ |
| Inference | Sliding window + best-confidence crop | No GCP location known at test time |

---

## Assumptions

1. GCP is always fully visible within the image
2. The `verified_shape` label is consistent per GCP across all its images
3. 4 entries missing `verified_shape` were dropped from training
4. Test images follow the same directory structure as train

---

## Setup & Running in Google Colab

### Step 1 — Open notebook
Upload `GCP_Detection_Pipeline.ipynb` to Google Colab.  
Set runtime to **GPU (T4)**: Runtime → Change runtime type → T4 GPU.

### Step 2 — Upload dataset zips to Google Drive
Upload `train_dataset.zip` and `test_dataset.zip` to your Google Drive root.

### Step 3 — Update paths in config cell
```python
TRAIN_DIR = '/content/data/train_dataset'
TEST_DIR  = '/content/data/test_dataset'
JSON_PATH = '/content/data/train_dataset/gcp_marks.json'
SAVE_DIR  = '/content/drive/MyDrive/gcp_outputs'
```

### Step 4 — Run all cells
Click **Runtime → Run all**. The notebook will:
1. Unzip datasets automatically
2. Run EDA
3. Train the model (~1.5 hours on T4)
4. Run inference on test set
5. Save `predictions.json` to your Drive

---

## Model Weights

**Download link:** https://drive.google.com/file/d/1GDoVyuQYzUgG5CFzQ-VE0jSr6LXTv85b/view?usp=sharing

---

## predictions.json Format

```json
{
  "project/survey/GCP_ID/DJI_xxxx.JPG": {
    "mark": {
      "x": 2134.5,
      "y": 987.2
    },
    "verified_shape": "L-Shape"
  }
}
```

---

## Challenges & Mitigations

| Challenge | Mitigation |
|---|---|
| GCP tiny in 4K image | Crop-based approach (384×384 around center) |
| No GCP location at test time | Sliding window inference, pick highest-confidence crop |
| Class imbalance (Cross underrepresented) | Weighted CrossEntropy loss |
| Small dataset (1000 images) | Heavy augmentation + pretrained backbone |
| Same GCP across multiple images | GroupShuffleSplit to avoid data leakage |
| Varied terrain and lighting | Brightness/contrast/blur augmentations |
