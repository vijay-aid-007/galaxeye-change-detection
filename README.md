# GalaxEye — Binary Change Detection on EO-SAR Image Pairs

**Position:** Satellite AI Research Intern  
**Task:** Binary pixel-level change detection from paired Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery  
**Dataset:** [doron333/change-detection-dataset](https://huggingface.co/datasets/doron333/change-detection-dataset)

---

## Project Description

Given a co-registered pre-event EO image and post-event SAR image of the same location, this model predicts a binary pixel-level change mask indicating which pixels have changed (damaged/destroyed buildings) versus remained unchanged.

**Approach:** UNet with pretrained ResNet34 encoder, early fusion of 4-channel input (3 EO + 1 SAR), trained with Focal+Dice loss to handle severe class imbalance (0.61% change pixels).

---

## Results

| Split      | IoU    | F1     | Precision | Recall | Threshold |
|------------|--------|--------|-----------|--------|-----------|
| Validation | 0.3182 | 0.5271 | 0.5352    | 0.5193 | 0.65      |
| Test       | 0.0176 | 0.0347 | 0.0188    | 0.2255 | 0.35      |

**Note on Val/Test gap:** The test set contains geographically diverse scenes 
from different regions (North American suburban areas) not represented in 
training data, causing "significant domain shift". The model achieves F1 up 
to 0.955 on individual validation samples with actual damage. Full analysis 
in the technical report.

---
Model weights link —  **[ https://drive.google.com/file/d/1oTnWJ8InrZBs2arlWKmrS2eK_EnWuJoi/view?usp=sharing ](#)** 
---
Model Check points — **[ https://drive.google.com/drive/folders/1LNl-GkWTJLfga-gPoabBJQkDMkSxY6P7?usp=sharing ](#)** 
---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: 8GB+ VRAM)

---

## Environment Setup

```bash
# Clone repository
git clone https://github.com/vijay-aid-007/galaxeye-change-detection.git
cd galaxeye-change-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate             # Linux/Mac
source venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Structure

The dataset is loaded automatically from HuggingFace — no manual download needed.

Internally, the dataset has 3 blocks per split:
```
Block 1: indices 0     to N-1    → Post-event SAR images  (grayscale, 1024×1024)
Block 2: indices N     to 2N-1   → Pre-event  EO  images  (RGB,       1024×1024×3)
Block 3: indices 2N    to 3N-1   → Target masks           (binary,    1024×1024)

Triplet i = (SAR[i], EO[i+N], Mask[i+2N])
```

Label remapping applied before all training and evaluation:
```
0 (Background) → 0 (No-Change)
1 (Intact)     → 0 (No-Change)
2 (Damaged)    → 1 (Change)
3 (Destroyed)  → 1 (Change)
```

---

## Training

```bash
python train.py --config config.yaml
```

All hyperparameters are in `config.yaml`. Modify there — do not hardcode in scripts.

---

## Evaluation

```bash
# Evaluate on test split
python eval.py --config config.yaml \
               --weights checkpoints/best_model.pth \
               --split test \
               --threshold 0.5

# Evaluate with visualizations
python eval.py --config config.yaml \
               --weights checkpoints/best_model.pth \
               --split test \
               --visualize
```

---

## Model Weights

Download trained model checkpoint:  
**[best_model.pth — https://drive.google.com/file/d/1oTnWJ8InrZBs2arlWKmrS2eK_EnWuJoi/view?usp=sharing ](#)**  

## Reproducibility Note

The model weights were trained on Google Colab (Tesla T4 GPU, 15.6GB VRAM)
using the dataset loaded from Google Drive cache via `load_from_disk`.

To reproduce training from scratch, the dataset must first be downloaded
and saved locally:

```python
from datasets import load_dataset
ds = load_dataset("doron333/change-detection-dataset")
ds.save_to_disk("data/dataset_dict")
```

Then update `dataset.py` line 113 to load from disk:
```python
ds = load_from_disk("data/dataset_dict")
```

Alternatively, the pretrained checkpoint can be downloaded directly
from the link above and used with `eval.py` without retraining.
---

## Repository Structure

```
galaxeye-change-detection/
├── config.yaml          ← all hyperparameters
├── train.py             ← training entry point
├── eval.py              ← evaluation entry point
├── requirements.txt
├── README.md
└── notebooks/
    ├── eda.ipynb        ← exploratory data analysis (EDA)
└── src/
    ├── __init__.py
    ├── dataset.py       ← dataloader + label remapping
    ├── model.py         ← UNet + ResNet34 architecture
    ├── losses.py        ← Focal loss + Dice loss
    ├── metrics.py       ← IoU, F1, Precision, Recall
    ├── utils.py         ← visualization + checkpoint utilities
    └── transforms.py    ← augmentation pipelines
```

---

## Citation / References / Optional 

- Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation
- Lin et al. (2017) — Focal Loss for Dense Object Detection  
- Chen et al. (2021) — Remote Sensing Change Detection with Transformers (BIT)
- Codegoni et al. (2022) — TINYCD: A (Not So) Tiny Model For Change Detection
- xBD Dataset — Gupta et al. (2019), Building Damage Assessment
- segmentation-models-pytorch: https://github.com/qubvel/segmentation_models.pytorch
- doron333/change-detection-dataset: https://huggingface.co/datasets/doron333/change-detection-dataset