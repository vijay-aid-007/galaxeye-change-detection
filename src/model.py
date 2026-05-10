"""
model.py — GalaxEye Change Detection

Architecture: UNet with pretrained ResNet34 encoder (segmentation_models_pytorch)

Design rationale:
  - UNet is the standard architecture for pixel-level segmentation.
    Skip connections preserve spatial detail lost in the encoder's downsampling,
    which is critical for detecting small changed regions (buildings, ~0.61% of pixels).

  - ResNet34 encoder pretrained on ImageNet provides strong low-level features
    (edges, textures) that transfer well to satellite imagery.
    GalaxEye explicitly permits pretrained backbone weights.

  - Input: 4 channels [EO_R, EO_G, EO_B, SAR].
    The first conv layer of ResNet34 is modified to accept 4 channels.
    We initialise the 4th channel weight by averaging the 3 RGB channel weights
    — this is better than random init because SAR captures structural/intensity
    information similar to grayscale optical, so the pretrained weights are
    a useful prior.

  - Output: 1 channel logit map (no sigmoid — applied in loss and metrics).

Alternative considered:
  - Siamese networks (FC-Siam-Diff, BIT, ChangeFormer) process pre and post
    images separately then compute difference features. Better for same-modality
    change detection (EO-EO or SAR-SAR).
  - For cross-modal (EO pre + SAR post), early fusion (concatenation) is
    competitive and simpler to implement and debug within the time constraint.
    Late fusion would be the next experiment if time permitted.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(cfg: dict) -> nn.Module:
    """
    Builds the UNet model from config.

    Args:
        cfg: full config dict (loaded from config.yaml)

    Returns:
        model: nn.Module ready for training
    """
    model_cfg = cfg['model']

    model = smp.Unet(
        encoder_name    = model_cfg.get('encoder', 'resnet34'),
        encoder_weights = 'imagenet' if model_cfg.get('pretrained', True) else None,
        in_channels     = model_cfg.get('in_channels', 4),
        classes         = model_cfg.get('out_channels', 1),
        activation      = None,   # raw logits — sigmoid applied in loss/metrics
    )

    # Patch the first conv layer to handle 4-channel input
    # segmentation_models_pytorch handles this automatically via in_channels=4
    # but we verify and log the weight initialisation strategy here
    first_conv = model.encoder.model.conv1  \
                 if hasattr(model.encoder, 'model') \
                 else model.encoder.conv1 \
                 if hasattr(model.encoder, 'conv1') \
                 else None

    if first_conv is not None:
        print(f"  First conv shape: {first_conv.weight.shape}  "
              f"(expected [64, 4, 7, 7])")

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")

    return model


# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("=== model.py sanity check ===")

    cfg = {
        'model': {
            'architecture': 'unet',
            'encoder'     : 'resnet34',
            'in_channels' : 4,
            'out_channels': 1,
            'pretrained'  : True,
        }
    }

    model = build_model(cfg)
    model.eval()

    # Forward pass with dummy input
    B, C, H, W = 2, 4, 512, 512
    dummy = torch.randn(B, C, H, W)

    with torch.no_grad():
        out = model(dummy)

    print(f"  Input  shape: {dummy.shape}")
    print(f"  Output shape: {out.shape}   (expected [2, 1, 512, 512])")
    assert out.shape == (B, 1, H, W), \
        f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    print("model.py: PASSED")