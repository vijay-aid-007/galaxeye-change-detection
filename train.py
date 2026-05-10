"""
train.py — GalaxEye Change Detection
Main training script.

Usage:
    python train.py --config config.yaml

Two-phase training strategy:
  Phase 1 (warmup_epochs): Encoder frozen, only decoder trains.
    WHY: The decoder starts with random weights. Letting random gradients
    flow into the pretrained encoder immediately corrupts the ImageNet
    features before the decoder is stable. We freeze the encoder first.

  Phase 2 (remaining epochs): Full end-to-end training.
    Encoder gets 10x lower LR than decoder.
    WHY: The encoder already has good features — it needs gentle updates.
    The decoder needs aggressive updates to learn change detection.
"""

import os
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import build_dataloaders
from src.model   import build_model
from src.losses  import FocalDiceLoss
from src.metrics import ChangeMetrics, find_best_threshold
from src.utils   import save_checkpoint, plot_training_history, plot_confusion_matrix


# ─────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    """Lock all random sources for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    """Single training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)              # [B, 1, H, W]
        loss   = criterion(logits, masks)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        # Max norm 1.0 is standard for segmentation tasks
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            avg = total_loss / (batch_idx + 1)
            print(f"  Epoch [{epoch}/{total_epochs}] "
                  f"Batch [{batch_idx+1}/{n_batches}] Loss: {avg:.4f}")

    return total_loss / n_batches


# ─────────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device, threshold=0.5):
    """Single validation pass. Returns loss, metrics dict, ChangeMetrics object."""
    model.eval()
    total_loss = 0.0
    metrics    = ChangeMetrics(threshold=threshold)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)
            logits = model(images)
            loss   = criterion(logits, masks)
            total_loss += loss.item()
            metrics.update(logits.cpu(), masks.cpu())

    return total_loss / len(loader), metrics.compute(), metrics


# ─────────────────────────────────────────────────────────────────
def main(config_path: str):

    # ── Load config ─────────────────────────────────────────────────
    cfg = load_config(config_path)

    # BUG FIX 1: Extract epochs HERE, at the top, before any reference to it.
    # Previously, `epochs` was used in a print statement before being assigned,
    # causing a NameError crash. Always assign before use.
    epochs         = cfg['training']['epochs']
    warmup_epochs  = cfg['training'].get('warmup_epochs', 5)
    lr             = cfg['training']['learning_rate']

    print(f"\n{'='*55}")
    print(f"  GalaxEye Change Detection — Training")
    print(f"{'='*55}")
    print(f"  Epochs       : {epochs}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Batch size   : {cfg['training']['batch_size']}")
    print(f"  Image size   : {cfg['dataset']['image_size']}")
    print(f"  LR           : {lr}")
    print(f"  Phase 1      : Encoder frozen for {warmup_epochs} epochs")
    print(f"  Phase 2      : Full end-to-end for {epochs - warmup_epochs} epochs")
    print(f"{'='*55}\n")

    # ── Reproducibility ─────────────────────────────────────────────
    set_seed(cfg['seed'])

    # ── Device ──────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # ── Dataloaders ─────────────────────────────────────────────────
    print("Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders({
        'image_size' : cfg['dataset']['image_size'],
        'batch_size' : cfg['training']['batch_size'],
        'num_workers': cfg['dataset']['num_workers'],
        'seed'       : cfg['seed'],
    })

    # ── Model ───────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(cfg).to(device)

    # ── Loss ────────────────────────────────────────────────────────
    criterion = FocalDiceLoss(
        focal_weight = cfg['loss']['focal_weight'],
        dice_weight  = cfg['loss']['dice_weight'],
        focal_alpha  = cfg['loss']['focal_alpha'],
        focal_gamma  = cfg['loss']['focal_gamma'],
    )

    # ── PHASE 1 SETUP: Freeze encoder ───────────────────────────────
    # Freeze all encoder parameters — they will not be updated in Phase 1
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Build optimizer — only non-frozen params are optimised in Phase 1
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = lr,
        weight_decay = cfg['training']['weight_decay'],
    )

    # BUG FIX 2: Use a SINGLE scheduler for the full training run.
    # Previously, a new scheduler was created in Phase 2, which reset the LR
    # back to the initial value — creating a sudden spike mid-training.
    # One scheduler, one smooth cosine curve, no restarts.
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max   = epochs,   # full duration — smooth decay over all epochs
        eta_min = 1e-6,
    )

    # ── Checkpoint directory ─────────────────────────────────────────
    os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['paths']['logs_dir'],       exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────
    best_f1    = 0.0
    best_epoch = 0
    history    = []

    # Store val logits for threshold search after training
    best_val_logits  = []
    best_val_targets = []

    print(f"\nStarting training...\n")

    for epoch in range(1, epochs + 1):

        # ── PHASE 2: Unfreeze encoder at warmup boundary ─────────────
        if epoch == warmup_epochs + 1:
            print(f"\n*** Phase 2 START at epoch {epoch} — Encoder UNFROZEN ***")

            # Unfreeze encoder
            for param in model.encoder.parameters():
                param.requires_grad = True

            # Rebuild optimizer with differential learning rates:
            # Encoder: 10x lower LR  — it has good features, needs gentle updates
            # Decoder: full LR       — learning change detection from scratch
            optimizer = AdamW([
                {'params': model.encoder.parameters(),
                 'lr': lr * 0.1},
                {'params': model.decoder.parameters(),
                 'lr': lr},
                {'params': model.segmentation_head.parameters(),
                 'lr': lr},
            ], weight_decay=cfg['training']['weight_decay'])

            # Continue scheduler from current point with remaining epochs
            # This gives a smooth LR curve for Phase 2 only
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max   = epochs - warmup_epochs,
                eta_min = 1e-6,
            )
            print(f"  Encoder LR : {lr * 0.1:.2e}")
            print(f"  Decoder LR : {lr:.2e}\n")

        # ── Train ────────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, epochs
        )

        # ── Validate ─────────────────────────────────────────────────
        val_loss, val_results, val_metrics_obj = validate(
            model, val_loader, criterion, device,
            threshold=cfg['evaluation']['threshold']
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Log ──────────────────────────────────────────────────────
        print(f"\n── Epoch {epoch}/{epochs} ──")
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val   Loss : {val_loss:.4f}")
        print(f"  Val   IoU  : {val_results['iou']:.4f}")
        print(f"  Val   F1   : {val_results['f1']:.4f}")
        print(f"  Val   Prec : {val_results['precision']:.4f}")
        print(f"  Val   Rec  : {val_results['recall']:.4f}")
        print(f"  LR         : {current_lr:.2e}\n")

        history.append({
            'epoch'     : epoch,
            'train_loss': train_loss,
            'val_loss'  : val_loss,
            **val_results,
        })

        # ── Save best model ───────────────────────────────────────────
        if val_results['f1'] > best_f1:
            best_f1    = val_results['f1']
            best_epoch = epoch
            save_checkpoint({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'scheduler'  : scheduler.state_dict(),
                'val_results': val_results,
                'config'     : cfg,
            }, cfg['paths']['best_model'])
            print(f"  *** Best model saved — Val F1: {best_f1:.4f} "
                  f"at epoch {best_epoch} ***\n")

            # Store val outputs for threshold search
            best_val_logits  = []
            best_val_targets = []
            model.eval()
            with torch.no_grad():
                for images, masks in val_loader:
                    logits = model(images.to(device))
                    best_val_logits.append(logits.cpu())
                    best_val_targets.append(masks.cpu())

    # ── Post-training: Threshold search on validation set ────────────
    print("\n" + "="*55)
    print("Post-training: threshold search on validation set...")
    best_thr, best_thr_f1 = find_best_threshold(
        best_val_logits, best_val_targets
    )
    print(f"Optimal threshold: {best_thr:.2f}  F1: {best_thr_f1:.4f}")

    # ── Final test evaluation with best checkpoint + best threshold ──
    print(f"\nLoading best checkpoint (epoch {best_epoch})...")
    ckpt = torch.load(cfg['paths']['best_model'], map_location=device)
    model.load_state_dict(ckpt['model_state'])

    print(f"Evaluating on VALIDATION set (threshold={best_thr:.2f})...")
    _, final_val_results, final_val_metrics = validate(
        model, val_loader, criterion, device, threshold=best_thr
    )
    final_val_metrics.print_results("Final Validation")
    val_cm = final_val_metrics.confusion_matrix()

    print(f"Evaluating on TEST set (threshold={best_thr:.2f})...")
    _, test_results, test_metrics = validate(
        model, test_loader, criterion, device, threshold=best_thr
    )
    test_metrics.print_results("Test")
    test_cm = test_metrics.confusion_matrix()

    # ── Save plots ───────────────────────────────────────────────────
    plot_training_history(history,
                          save_path='logs/training_history.png')
    plot_confusion_matrix(test_cm,
                          save_path='logs/confusion_matrix_test.png',
                          title=f'Test Confusion Matrix (thr={best_thr:.2f})')
    plot_confusion_matrix(val_cm,
                          save_path='logs/confusion_matrix_val.png',
                          title='Validation Confusion Matrix')

    # ── Save results to file ─────────────────────────────────────────
    results_path = os.path.join(cfg['paths']['logs_dir'], 'final_results.txt')
    with open(results_path, 'w') as f:
        f.write("GalaxEye Change Detection — Final Results\n")
        f.write(f"{'='*45}\n")
        f.write(f"Best epoch     : {best_epoch}/{epochs}\n")
        f.write(f"Best threshold : {best_thr}\n\n")
        f.write("Validation Results:\n")
        for k, v in final_val_results.items():
            f.write(f"  {k:12s}: {v}\n")
        f.write("\nTest Results:\n")
        for k, v in test_results.items():
            f.write(f"  {k:12s}: {v}\n")
        f.write(f"\nTraining History:\n")
        f.write(f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>9} "
                f"{'IoU':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}\n")
        for h in history:
            f.write(f"{h['epoch']:>6} {h['train_loss']:>10.4f} "
                    f"{h['val_loss']:>9.4f} "
                    f"{h['iou']:>7.4f} {h['f1']:>7.4f} "
                    f"{h['precision']:>7.4f} {h['recall']:>7.4f}\n")

    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best epoch    : {best_epoch}")
    print(f"  Best Val F1   : {best_f1:.4f}")
    print(f"  Best threshold: {best_thr:.2f}")
    print(f"  Results saved : {results_path}")
    print(f"  Model saved   : {cfg['paths']['best_model']}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    main(args.config)