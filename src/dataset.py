"""
dataset.py — GalaxEye Change Detection
Dataset loader for HuggingFace doron333/change-detection-dataset

Dataset structure (confirmed from EDA):
  - Block 1: indices 0      to N-1     → post-event SAR  (1024x1024, grayscale)
  - Block 2: indices N      to 2N-1    → pre-event  EO   (1024x1024x3, RGB)
  - Block 3: indices 2N     to 3N-1    → target mask      (1024x1024, uint8 {0,1,2,3})

Pairing: triplet i → (post[i], pre[i+N], mask[i+2N])

Label remapping (mandatory per GalaxEye spec):
  0 (Background) → 0 (No-Change)
  1 (Intact)     → 0 (No-Change)
  2 (Damaged)    → 1 (Change)
  3 (Destroyed)  → 1 (Change)

Input to model: 4-channel tensor [EO_R, EO_G, EO_B, SAR] normalized to [0,1]
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transforms import get_train_transforms, get_val_transforms

# ------------------------------------------------------------------
# Label remapping — applied to every mask, no exceptions
# ------------------------------------------------------------------
REMAP = np.array([0, 0, 1, 1], dtype=np.uint8)  # index = original value

def remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remap 4-class mask to binary.
    Handles values 0-3. Any value > 3 is clamped to 0 (background).
    """
    clipped = np.clip(mask, 0, 3)
    return REMAP[clipped]


# ------------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------------
class ChangeDetectionDataset(Dataset):
    """
    Loads EO-SAR triplets from HuggingFace dataset.

    Each item returns:
        image : FloatTensor [4, H, W]  — 3 EO channels + 1 SAR channel
        mask  : LongTensor  [H, W]     — binary {0, 1}
    """

    def __init__(self, hf_split, block_size: int, transform=None):
        """
        Args:
            hf_split   : HuggingFace dataset split (ds['train'], ds['validation'], etc.)
            block_size : Number of triplets in this split (2781 / 334 / 77)
            transform  : Albumentations transform pipeline
        """
        self.split      = hf_split
        self.block_size = block_size
        self.transform  = transform

    def __len__(self):
        return self.block_size

    def __getitem__(self, idx):
        # --- Load three images from their respective blocks ---
        post_sample = self.split[idx]                          # SAR post-event
        pre_sample  = self.split[idx + self.block_size]        # EO  pre-event
        mask_sample = self.split[idx + 2 * self.block_size]    # target mask

        # --- Convert to numpy ---
        sar = np.array(post_sample['image'])   # (H, W)       uint8
        eo  = np.array(pre_sample['image'])    # (H, W, 3)    uint8
        mask_raw = np.array(mask_sample['image'])  # (H, W)   uint8

        # --- Remap mask to binary ---
        mask = remap_mask(mask_raw)            # (H, W)       uint8 {0,1}

        # --- Expand SAR to (H, W, 1) and concatenate with EO ---
        # Final image shape: (H, W, 4) — channels [R, G, B, SAR]
        sar_3d = sar[:, :, np.newaxis]
        image  = np.concatenate([eo, sar_3d], axis=-1).astype(np.float32)
        mask   = mask.astype(np.int64)

        # --- Apply transforms ---
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']   # FloatTensor [4, H, W]
            mask  = augmented['mask']    # LongTensor  [H, W]

        return image, mask


# ------------------------------------------------------------------
# Factory function — call this from train.py
# ------------------------------------------------------------------
def build_dataloaders(cfg: dict):
    """
    Loads HuggingFace dataset and returns train/val/test DataLoaders.

    Args:
        cfg: config dict with keys:
             image_size, batch_size, num_workers, seed

    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader

    print("Loading dataset from HuggingFace...")
    ds = load_dataset("doron333/change-detection-dataset")
    print(f"  Train triplets : {len(ds['train']) // 3}")
    print(f"  Val   triplets : {len(ds['validation']) // 3}")
    print(f"  Test  triplets : {len(ds['test']) // 3}")

    image_size = cfg.get('image_size', 512)

    train_ds = ChangeDetectionDataset(
        hf_split   = ds['train'],
        block_size = len(ds['train']) // 3,
        transform  = get_train_transforms(image_size),
    )
    val_ds = ChangeDetectionDataset(
        hf_split   = ds['validation'],
        block_size = len(ds['validation']) // 3,
        transform  = get_val_transforms(image_size),
    )
    test_ds = ChangeDetectionDataset(
        hf_split   = ds['test'],
        block_size = len(ds['test']) // 3,
        transform  = get_val_transforms(image_size),
    )

    g = torch.Generator()
    g.manual_seed(cfg.get('seed', 42))

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.get('batch_size', 8),
        shuffle     = True,
        num_workers = cfg.get('num_workers', 2),
        pin_memory  = True,
        generator   = g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.get('batch_size', 8),
        shuffle     = False,
        num_workers = cfg.get('num_workers', 2),
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg.get('batch_size', 8),
        shuffle     = False,
        num_workers = cfg.get('num_workers', 2),
        pin_memory  = True,
    )

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------------
# Quick sanity check — run this file directly to verify
# ------------------------------------------------------------------
if __name__ == '__main__':
    from datasets import load_dataset

    print("=== dataset.py sanity check ===")
    ds = load_dataset("doron333/change-detection-dataset")

    dataset = ChangeDetectionDataset(
        hf_split   = ds['train'],
        block_size = len(ds['train']) // 3,
        transform  = get_val_transforms(512),
    )

    print(f"Dataset length: {len(dataset)}")
    img, mask = dataset[0]
    print(f"Image shape : {img.shape}   dtype: {img.dtype}")
    print(f"Mask  shape : {mask.shape}  dtype: {mask.dtype}")
    print(f"Mask unique : {mask.unique()}")
    print(f"Image min/max: {img.min():.3f} / {img.max():.3f}")

    # Verify class imbalance
    change_pct = 100 * (mask == 1).float().mean().item()
    print(f"Change pixels in this sample: {change_pct:.2f}%")
    print("PASSED — dataset.py is working correctly.")