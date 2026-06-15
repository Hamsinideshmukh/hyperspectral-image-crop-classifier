# Hyperspectral Crop Classification 

Pixel-wise land-cover classification on a 270-band hyperspectral cube using PCA + 1D-CNN. Achieves **99.25% OA** on 9 crop classes.

---

## Dataset

**WHU-Hi-LongKou** — 550×400×270 HSI scene (400–1000 nm), 204,542 labelled pixels across 9 classes: Corn, Cotton, Sesame, Broad-Leaf Soybean, Narrow-Leaf Soybean, Rice, Water Body, Mixed Weed, Roads & Houses. Severe class imbalance (Rice dominant, Sesame minority).

---

## Pipeline

**EDA** — Band variance profiling, per-class mean spectral signatures, NDVI map (NIR≈band 55, Red≈band 30), false-colour RGB composite (bands 50/30/10).

**Preprocessing** — Min-max normalization → PCA (270→30 components, ≥95% variance retained). t-SNE on PCA features confirms class separability before training.

**Class Balancing** — Minority classes oversampled by repetition (`reps = max_count / class_count`). Augmentation (vertical/horizontal flip, ±180° rotation) applied to 50% of training patches.

**Model** — 1D-CNN on the 30-component spectrum per pixel: `Conv1D(20, kernel=3, ReLU) → MaxPool1D → Flatten → Dense(128, ReLU) → Dense(9, Softmax)`. No spatial context — the spectral fingerprint alone is sufficient.

**Training** — Adam (lr=1e-3), categorical cross-entropy, batch size 64, early stopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=3). 75/25 stratified split.

---

## Results

| Metric | Score |
|---|---|
| Overall Accuracy (OA) | 99.25% |
| Average Accuracy (AA) | ≥99% |
| Cohen's Kappa (κ) | ≈0.99 |

Per-class precision/recall/F1 computed for all 9 classes including Sesame — confirming oversampling prevented minority class collapse. Full 550×400 spatial prediction map generated at inference.

---

## Key Decisions

- **1×1 window (pure spectral)** — spatial context adds compute without gain at this resolution; the 270-band signature is discriminative enough.
- **PCA is non-negotiable** — removes inter-band collinearity and sensor noise; scree plot confirms 95%+ variance in 30 components.
- **Oversampling before splitting** — without it, Sesame collapses into Rice; high OA masks catastrophic per-class failure.

---


**Data:** `WHU_Hi_LongKou.mat`, `WHU_Hi_LongKou_gt.mat` — [WHU Hyperspectral Dataset on Kaggle](https://www.kaggle.com)

---

Course Project — Multimedia Processing and Analysis (EC5110), IIITDM Kancheepuram. Team: ME23B2016 Hamsini Deshmukh , ME23B2010 Dhanya Venkatesh
