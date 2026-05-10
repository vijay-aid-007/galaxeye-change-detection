"""
losses.py — GalaxEye Change Detection

Loss function design rationale:
  Class imbalance is severe: 99.39% no-change vs 0.61% change.
  Standard BCE will converge to predict all-zeros and score ~99% accuracy.
  We use a combination of two losses:

  1. Focal Loss — down-weights easy negatives (the 99% no-change pixels),
     forces the model to focus on hard positives (the 0.61% change pixels).
     Introduced by Lin et al. (2017) for object detection, equally effective
     for imbalanced segmentation.

  2. Dice Loss — directly optimizes overlap between prediction and target.
     Dice is class-frequency-independent by design, making it naturally
     robust to imbalance. Complement to Focal which operates on per-pixel loss.

  Combined loss = alpha * FocalLoss + (1 - alpha) * DiceLoss
  Default alpha = 0.5 (equal weight, tunable via config.yaml)

  Why not weighted BCE alone?
  Weighted BCE requires knowing the exact positive weight ratio upfront.
  Focal + Dice is more robust — it adapts to the actual difficulty of each
  sample without manual weight tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Focal Loss
# ------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Binary Focal Loss for imbalanced segmentation.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha : weight for positive class (change pixels).
                Set > 0.5 to penalise missing change pixels more.
        gamma : focusing parameter. gamma=0 → standard BCE.
                gamma=2 is the standard recommendation (Lin et al. 2017).
        reduction : 'mean' or 'sum'
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : [B, 1, H, W] or [B, H, W] — raw model output (no sigmoid)
            targets : [B, H, W] — binary mask, dtype long or float
        """
        if logits.dim() == 4:
            logits = logits.squeeze(1)

        targets = targets.float()
        bce     = F.binary_cross_entropy_with_logits(
                      logits, targets, reduction='none')

        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ------------------------------------------------------------------
# Dice Loss
# ------------------------------------------------------------------
class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    DiceLoss = 1 - Dice

    Uses soft (probabilistic) dice during training so loss is
    differentiable. At inference we threshold to get hard predictions.

    Args:
        smooth : small constant to prevent division by zero
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : [B, 1, H, W] or [B, H, W]
            targets : [B, H, W] binary mask
        """
        if logits.dim() == 4:
            logits = logits.squeeze(1)

        probs   = torch.sigmoid(logits)
        targets = targets.float()

        # Flatten spatial dims for dice computation
        probs   = probs.view(probs.size(0), -1)    # [B, H*W]
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return 1.0 - dice.mean()


# ------------------------------------------------------------------
# Combined Loss (default)
# ------------------------------------------------------------------
class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice loss.

    total_loss = focal_weight * focal + dice_weight * dice

    This combination is widely used in medical and satellite image
    segmentation with severe class imbalance. Focal handles per-pixel
    weighting; Dice handles global overlap optimisation.

    Args:
        focal_weight : weight of focal loss term (default 0.5)
        dice_weight  : weight of dice loss term  (default 0.5)
        focal_alpha  : alpha for FocalLoss (positive class weight)
        focal_gamma  : gamma for FocalLoss (focusing strength)
    """

    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5,
                 focal_alpha: float = 0.75, focal_gamma: float = 2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight  = dice_weight
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice  = DiceLoss()

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        dice_loss  = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("=== losses.py sanity check ===")

    B, H, W = 2, 256, 256
    logits  = torch.randn(B, 1, H, W)
    targets = torch.zeros(B, H, W).long()
    targets[:, 50:100, 50:100] = 1  # small change region

    focal = FocalLoss()
    dice  = DiceLoss()
    combo = FocalDiceLoss()

    fl = focal(logits, targets)
    dl = dice(logits, targets)
    cl = combo(logits, targets)

    print(f"  FocalLoss     : {fl.item():.4f}")
    print(f"  DiceLoss      : {dl.item():.4f}")
    print(f"  FocalDiceLoss : {cl.item():.4f}")
    assert not torch.isnan(fl), "FocalLoss returned NaN"
    assert not torch.isnan(dl), "DiceLoss returned NaN"
    assert not torch.isnan(cl), "Combined loss returned NaN"

    # Test with perfect prediction
    perfect_logits = torch.full((B, 1, H, W), -10.0)
    perfect_logits[:, :, 50:100, 50:100] = 10.0
    perfect_loss = combo(perfect_logits, targets)
    print(f"  Perfect pred loss : {perfect_loss.item():.6f}  (should be near 0)")
    print("losses.py: PASSED")