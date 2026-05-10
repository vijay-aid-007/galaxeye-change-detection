"""
transforms.py — GalaxEye Change Detection

All augmentation pipelines for training and validation.

Why augmentation matters for this dataset:
  - Only 2781 training triplets — relatively small for deep learning
  - Heavy augmentation artificially increases effective dataset size
  - Spatial augmentations (flip, rotate) teach the model that change
    detection is rotation-invariant — a destroyed building looks the
    same regardless of orientation
  - Radiometric augmentations (brightness, contrast) simulate different
    lighting and atmospheric conditions in EO imagery
  - Gaussian noise augmentation simulates SAR speckle variation across
    different acquisition conditions

IMPORTANT: Albumentations applies IDENTICAL spatial transforms to both
the image AND the mask simultaneously. This is critical — if you flip
the image, the mask must be flipped the same way. Albumentations
handles this automatically when you pass mask= to the transform call.

Input image format  : numpy array (H, W, 4) — float32 — [EO_R, EO_G, EO_B, SAR]
Input mask format   : numpy array (H, W)    — int64   — binary {0, 1}
Output image format : torch.FloatTensor (4, H, W) — normalized
Output mask format  : torch.LongTensor  (H, W)    — binary {0, 1}
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ------------------------------------------------------------------
# Normalization constants
# ------------------------------------------------------------------
# EO channels (RGB): ImageNet mean and std — valid because our ResNet34
# encoder was pretrained on ImageNet. Using these exact values ensures
# the pretrained features activate correctly.
#
# SAR channel: mean=0.5, std=0.25 — SAR values are uint8 (0-255),
# normalized to [0,1] by Normalize, then centered around 0.5.
# No established SAR-specific normalization standard exists for this
# dataset, so we use a neutral prior. A dataset-specific mean/std
# could be computed from the training set for marginal improvement.

EO_MEAN  = [0.485, 0.456, 0.406]   # ImageNet RGB mean
EO_STD   = [0.229, 0.224, 0.225]   # ImageNet RGB std
SAR_MEAN = [0.5]                    # SAR channel mean
SAR_STD  = [0.25]                   # SAR channel std

NORM_MEAN = EO_MEAN + SAR_MEAN      # [0.485, 0.456, 0.406, 0.5]
NORM_STD  = EO_STD  + SAR_STD       # [0.229, 0.224, 0.225, 0.25]


# ------------------------------------------------------------------
# Training transforms
# ------------------------------------------------------------------
def get_train_transforms(image_size: int = 512) -> A.Compose:
    """
    Augmentation pipeline for training split.

    Each augmentation and its justification:

    RandomCrop(512, 512):
        Original images are 1024x1024. Random cropping introduces
        positional variation — the model sees different parts of each
        scene at each epoch, effectively multiplying dataset size.
        512x512 is large enough to capture spatial context around
        change regions while fitting in GPU memory at batch_size=8.

    HorizontalFlip(p=0.5):
        Satellite imagery has no preferred horizontal orientation.
        Flipping doubles effective dataset size with no information loss.

    VerticalFlip(p=0.5):
        Same reasoning. Combined with HorizontalFlip gives 4x coverage
        of each spatial orientation.

    RandomRotate90(p=0.5):
        90-degree rotations maintain pixel alignment between image and
        mask. Combined with flips gives 8 possible orientations per image.
        Note: we use 90-degree only (not arbitrary angle) to avoid
        introducing interpolation artifacts in the binary mask.

    ColorJitter(brightness=0.2, contrast=0.2, p=0.3):
        Simulates variation in solar illumination and atmospheric haze
        in EO imagery. Applied only to image pixels, not mask.
        Parameters kept conservative (0.2) to avoid unrealistic imagery.

    GaussNoise(p=0.2):
        Adds random Gaussian noise. For the SAR channel this simulates
        speckle variation across different acquisition dates and
        incidence angles. For EO channels it simulates sensor noise.

    Normalize + ToTensorV2:
        Mandatory final step. Converts HWC numpy array to CHW tensor
        and normalizes to ImageNet statistics for EO, neutral for SAR.
    """
    return A.Compose([
        # Spatial augmentations
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Radiometric augmentations (image only, mask unaffected)
        A.ColorJitter(
            brightness = 0.2,
            contrast   = 0.2,
            saturation = 0.1,
            hue        = 0.0,   # no hue shift — satellite imagery is calibrated
            p          = 0.3,
        ),
        A.GaussNoise(
            var_limit = (5.0, 20.0),
            p         = 0.2,
        ),

        # Normalize and convert to tensor
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


# ------------------------------------------------------------------
# Validation / Test transforms
# ------------------------------------------------------------------
def get_val_transforms(image_size: int = 512) -> A.Compose:
    """
    Minimal pipeline for validation and test splits.

    No spatial augmentation — we want deterministic, reproducible
    evaluation. CenterCrop takes the center region of the 1024x1024
    image, which tends to have the most valid (non-black) pixels.

    CenterCrop(512, 512):
        Consistent crop position across all evaluation runs.
        Ensures metrics are comparable across experiments.

    Normalize + ToTensorV2:
        Same normalization as training — mandatory for correct
        activation of pretrained features.
    """
    return A.Compose([
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


# ------------------------------------------------------------------
# Test-time augmentation (TTA) — optional, use at inference only
# ------------------------------------------------------------------
def get_tta_transforms(image_size: int = 512) -> list:
    """
    Test-Time Augmentation: run inference on multiple versions of
    each test image and average the probability maps.

    Why TTA helps:
        A model trained with random flips learns flip-invariant features
        but still predicts slightly differently on a flipped vs unflipped
        image due to training stochasticity. Averaging predictions across
        all 4 orientations reduces this variance and typically improves
        F1 by 1-3 points without any retraining.

    Returns list of 4 transforms — one per orientation.
    Use in eval.py like:
        probs = []
        for tta_tf in get_tta_transforms():
            aug = tta_tf(image=image, mask=mask)
            with torch.no_grad():
                pred = model(aug['image'].unsqueeze(0))
            probs.append(torch.sigmoid(pred))
        final_prob = torch.stack(probs).mean(0)
    """
    base = [
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ]

    return [
        A.Compose(base),                                        # original
        A.Compose([A.HorizontalFlip(p=1.0)] + base),           # H-flip
        A.Compose([A.VerticalFlip(p=1.0)]   + base),           # V-flip
        A.Compose([A.HorizontalFlip(p=1.0),
                   A.VerticalFlip(p=1.0)]   + base),           # both flips
    ]


# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np
    import torch

    print("=== transforms.py sanity check ===")

    # Simulate a 4-channel image and binary mask
    H, W = 1024, 1024
    fake_image = np.random.randint(0, 255, (H, W, 4)).astype(np.float32)
    fake_mask  = np.zeros((H, W), dtype=np.int64)
    fake_mask[200:300, 200:300] = 1  # small change region

    # Training transform
    train_tf = get_train_transforms(512)
    out = train_tf(image=fake_image, mask=fake_mask)
    img_t  = out['image']
    mask_t = out['mask']

    print(f"  Train image output shape : {img_t.shape}   "
          f"(expected [4, 512, 512])")
    print(f"  Train mask  output shape : {mask_t.shape}  "
          f"(expected [512, 512])")
    print(f"  Train image dtype        : {img_t.dtype}   "
          f"(expected torch.float32)")
    print(f"  Train mask  dtype        : {mask_t.dtype}  "
          f"(expected torch.int64)")
    print(f"  Train mask  unique       : {mask_t.unique().tolist()}")

    assert img_t.shape  == (4, 512, 512), "Image shape mismatch"
    assert mask_t.shape == (512, 512),    "Mask shape mismatch"
    assert img_t.dtype  == torch.float32, "Image dtype mismatch"

    # Validation transform
    val_tf = get_val_transforms(512)
    out_v  = val_tf(image=fake_image, mask=fake_mask)
    assert out_v['image'].shape == (4, 512, 512), "Val image shape mismatch"
    print(f"  Val   image output shape : {out_v['image'].shape}  PASSED")

    # TTA transforms
    tta_tfs = get_tta_transforms(512)
    assert len(tta_tfs) == 4, "Should have 4 TTA transforms"
    for i, tta in enumerate(tta_tfs):
        out_t = tta(image=fake_image, mask=fake_mask)
        assert out_t['image'].shape == (4, 512, 512)
    print(f"  TTA  ({len(tta_tfs)} variants) : PASSED")

    print("\ntransforms.py: ALL PASSED")