# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# NORM_MEAN = [0.485, 0.456, 0.406, 0.5]
# NORM_STD  = [0.229, 0.224, 0.225, 0.25]

# def get_train_transforms(image_size=512):
#     return A.Compose([
#         A.RandomCrop(height=image_size, width=image_size),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.Normalize(mean=NORM_MEAN, std=NORM_STD),
#         ToTensorV2(),
#     ])

# def get_val_transforms(image_size=256):
#     return A.Compose([
#         A.CenterCrop(height=image_size, width=image_size),
#         A.Normalize(mean=NORM_MEAN, std=NORM_STD),
#         ToTensorV2(),
#     ])

# def get_tta_transforms(image_size=256):
#     base = [
#         A.CenterCrop(height=image_size, width=image_size),
#         A.Normalize(mean=NORM_MEAN, std=NORM_STD),
#         ToTensorV2(),
#     ]
#     return [
#         A.Compose(base),
#         A.Compose([A.HorizontalFlip(p=1.0)] + base),
#         A.Compose([A.VerticalFlip(p=1.0)] + base),
#         A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base),
#     ]






"""
transforms.py — GalaxEye Change Detection

FIX 1: TTA transform ordering corrected.
        Flips now come AFTER CenterCrop, not before.
        Previously: [HorizontalFlip] + [CenterCrop, Normalize, ToTensor]
        → flip happened before crop, breaking spatial alignment across TTA variants.
        Now:        [CenterCrop, HorizontalFlip, Normalize, ToTensor]
        → all variants operate on the same crop window.

FIX 2: Pylance type errors resolved.
        A.Compose() requires Sequence[BasicTransform], not list[SpecificFlip].
        The TTA list now passes transforms directly into A.Compose([...]) in
        the correct order — no list concatenation with incompatible types.

FIX 3: Consistent default image_size=256 across all three functions.
        Previously get_train_transforms defaulted to 512 while val/tta defaulted
        to 256. All defaults are now 256; callers that want 512 pass it explicitly.
"""

from typing import List
import albumentations as A
from albumentations.pytorch import ToTensorV2

# SAR channel mean/std: 0.5 / 0.25 are reasonable priors for a grayscale
# channel normalised from uint8 [0,255]. Ideally replace with per-channel
# statistics measured from the actual dataset after EDA.
NORM_MEAN = [0.485, 0.456, 0.406, 0.5]
NORM_STD  = [0.229, 0.224, 0.225, 0.25]


def get_train_transforms(image_size: int = 256) -> A.Compose:
    """
    Augmentation pipeline for training.
    RandomCrop + spatial augmentations + normalize + to tensor.
    """
    return A.Compose([
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 256) -> A.Compose:
    """
    Deterministic pipeline for validation / test / inference.
    CenterCrop + normalize + to tensor. No augmentation.
    """
    return A.Compose([
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 256) -> List[A.Compose]:
    """
    Test-Time Augmentation (TTA) pipelines.

    Returns 4 deterministic variants:
      0 — no flip        (identity)
      1 — horizontal flip
      2 — vertical flip
      3 — horizontal + vertical flip

    FIX: Flips are placed AFTER CenterCrop and BEFORE Normalize.
    Previously flips were prepended before the crop, so each TTA variant
    cropped a differently-oriented image — breaking spatial consistency.
    Now all variants crop the same window first, then flip, then normalize.

    Caller is responsible for averaging logits across all 4 pipelines
    before thresholding. Inverse flips must be applied to predicted masks
    before averaging (flip-back step).
    """
    return [
        # Variant 0: no flip
        A.Compose([
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ]),
        # Variant 1: horizontal flip
        A.Compose([
            A.CenterCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ]),
        # Variant 2: vertical flip
        A.Compose([
            A.CenterCrop(height=image_size, width=image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ]),
        # Variant 3: horizontal + vertical flip
        A.Compose([
            A.CenterCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD),
            ToTensorV2(),
        ]),
    ]


# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np

    print("=== transforms.py sanity check ===")

    dummy_image = np.random.randint(0, 255, (1024, 1024, 4), dtype=np.uint8).astype(np.float32)
    dummy_mask  = np.random.randint(0, 2, (1024, 1024), dtype=np.int64)

    train_t = get_train_transforms(256)
    val_t   = get_val_transforms(256)
    tta_ts  = get_tta_transforms(256)

    out = train_t(image=dummy_image, mask=dummy_mask)
    assert out['image'].shape == (4, 256, 256), f"Train image shape wrong: {out['image'].shape}"
    print(f"  Train transform : image={out['image'].shape}  mask={out['mask'].shape}  PASSED")

    out = val_t(image=dummy_image, mask=dummy_mask)
    assert out['image'].shape == (4, 256, 256), f"Val image shape wrong: {out['image'].shape}"
    print(f"  Val transform   : image={out['image'].shape}  mask={out['mask'].shape}  PASSED")

    for i, tta_t in enumerate(tta_ts):
        out = tta_t(image=dummy_image, mask=dummy_mask)
        assert out['image'].shape == (4, 256, 256), f"TTA[{i}] image shape wrong: {out['image'].shape}"
    print(f"  TTA transforms  : all 4 variants PASSED")

    # Verify TTA variant 0 and variant 1 differ (flip was actually applied)
    out0 = tta_ts[0](image=dummy_image, mask=dummy_mask)
    out1 = tta_ts[1](image=dummy_image, mask=dummy_mask)
    import torch
    assert not torch.equal(out0['image'], out1['image']), \
        "TTA variant 0 and 1 are identical — flip not applied!"
    print("  TTA flip verification: PASSED")

    print("transforms.py: ALL PASSED")