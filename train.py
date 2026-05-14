"""
train.py — GalaxEye Change Detection
=====================================
Main training script for satellite image change detection.

Usage:
    python train.py --config config.yaml

Architecture:
    Encoder–Decoder segmentation model (e.g. ResNet + UNet-style decoder)
    trained on pairs of bi-temporal satellite images to produce binary
    change masks.

Two-Phase Training Strategy:
─────────────────────────────
  Phase 1 — Warmup (epochs 1 … warmup_epochs):
      Encoder is FROZEN; only decoder + segmentation head train.
      Rationale: the decoder starts from random weights. Backpropagating
      random gradients into a pretrained encoder immediately degrades the
      ImageNet feature representations before the decoder is stable enough
      to produce meaningful gradients. Freezing the encoder first lets the
      decoder converge to a reasonable solution safely.

  Phase 2 — Full fine-tuning (epochs warmup_epochs+1 … total_epochs):
      Entire network trains end-to-end with differential learning rates:
        • Encoder  → lr × 0.1  (already has good features; gentle updates)
        • Decoder  → lr        (still learning change detection; aggressive)
      A fresh CosineAnnealingLR scheduler is created for Phase 2 so it
      decays over the remaining epochs only, giving a full cosine arc in
      each phase rather than one shared curve that is already half-spent
      when Phase 2 starts.

Bug Fixes vs. original (see inline FIX comments for full rationale):
──────────────────────────────────────────────────────────────────────
  FIX-1  Scheduler/optimizer mismatch in Phase 2:
         When the optimizer is rebuilt for Phase 2, the scheduler is also
         rebuilt to point at the new optimizer. Without this, the old
         scheduler continued stepping the Phase-1 optimizer; the Phase-2
         encoder lr group was never touched.

  FIX-2  Redundant validation pass eliminated:
         validate() now accepts return_logits=True and collects logits
         during the normal val pass. Previously, a full second pass over
         the val loader was done every time a new best model was found.

  FIX-3  Pylance / type-narrowing error on logits_list.append():
         logits_list and targets_list are now declared as
         list[torch.Tensor] unconditionally instead of
         `[] if flag else None`, which kept the inferred type as
         list | None and caused Pylance to flag every .append() call.

  FIX-4  Comment accuracy:
         Phase-2 start epoch is now derived from warmup_epochs at runtime,
         not hardcoded in a comment that silently diverged from the config.
"""

import os
import yaml
import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import build_dataloaders
from src.losses  import FocalDiceLoss
from src.metrics import ChangeMetrics, find_best_threshold
from src.model   import build_model
from src.utils   import plot_confusion_matrix, plot_training_history, save_checkpoint


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Lock all random-number sources for full reproducibility.

    Covers Python's built-in random, NumPy, and both CPU / CUDA PyTorch
    generators.  deterministic=True forces cuDNN to use deterministic
    algorithms at the cost of some throughput; benchmark=False prevents
    cuDNN from auto-selecting non-deterministic convolution algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_config(path: str) -> dict:
    """Parse a YAML config file and return its contents as a dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Training / Validation
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    criterion:    nn.Module,
    device:       torch.device,
    epoch:        int,
    total_epochs: int,
) -> float:
    """
    Run one full training epoch over `loader`.

    Gradient clipping (max_norm=1.0) guards against exploding gradients,
    which is a common failure mode early in training when the decoder
    weights are still far from their optimum.

    Returns:
        Average loss over all batches in the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)          # [B, 1, H, W]
        loss   = criterion(logits, masks)
        loss.backward()

        # Clip gradient norm to 1.0 — standard ceiling for segmentation tasks.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        # Log every 50 batches and at the very last batch.
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            avg = total_loss / (batch_idx + 1)
            print(
                f"  Epoch [{epoch}/{total_epochs}]  "
                f"Batch [{batch_idx + 1}/{n_batches}]  "
                f"Loss: {avg:.4f}"
            )

    return total_loss / n_batches


def validate(
    model:         nn.Module,
    loader:        DataLoader,
    criterion:     nn.Module,
    device:        torch.device,
    threshold:     float = 0.5,
    return_logits: bool  = False,
) -> Tuple:
    """
    Run one full validation pass over `loader`.

    FIX-2: The return_logits flag lets the caller capture all raw logits
    and ground-truth masks in the same pass used for metric computation,
    removing the need for a redundant second pass whenever threshold search
    or external analysis is required.

    FIX-3: logits_list and targets_list are declared as list[torch.Tensor]
    unconditionally (not `[] if flag else None`) so Pylance can resolve the
    concrete type and stop flagging .append() calls as errors.

    Args:
        model:         The segmentation model in eval mode.
        loader:        Validation DataLoader.
        criterion:     Loss function (same as training).
        device:        Target device.
        threshold:     Binarisation threshold for predicted probabilities.
        return_logits: When True, return raw logits and targets as extra
                       elements in the tuple for downstream threshold search.

    Returns:
        (avg_loss, metrics_dict, ChangeMetrics_obj)
        — or, when return_logits=True —
        (avg_loss, metrics_dict, ChangeMetrics_obj,
         logits_list, targets_list)
    """
    model.eval()

    total_loss:   float             = 0.0
    metrics:      ChangeMetrics     = ChangeMetrics(threshold=threshold)

    # FIX-3: Always list[torch.Tensor], never Optional — Pylance can now
    # narrow the type correctly and .append() is unambiguously valid.
    logits_list:  list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, masks)
            total_loss += loss.item()

            logits_cpu = logits.cpu()
            masks_cpu  = masks.cpu()

            metrics.update(logits_cpu, masks_cpu)

            # Always collect; the lists are only surfaced when return_logits=True.
            # Overhead is a single .append() per batch regardless of the flag,
            # which is negligible compared with the forward pass.
            logits_list.append(logits_cpu)
            targets_list.append(masks_cpu)

    avg_loss = total_loss / len(loader)

    if return_logits:
        return avg_loss, metrics.compute(), metrics, logits_list, targets_list
    return avg_loss, metrics.compute(), metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    """
    Orchestrate the full training run:

        1.  Load config + set global seed.
        2.  Build dataloaders, model, loss.
        3.  Phase 1 — warmup with frozen encoder.
        4.  Phase 2 — end-to-end fine-tuning with differential LRs.
        5.  Post-training threshold search on the best validation set.
        6.  Final evaluation on val + test with the best checkpoint.
        7.  Persist plots, confusion matrices, and a results text file.
    """

    # ── Config ───────────────────────────────────────────────────────────────
    cfg = load_config(config_path)

    # Extract core scalars up front so they are available everywhere in main()
    # without repeated dict look-ups and without any risk of NameError from
    # referencing a variable before assignment (original FIX was for this).
    epochs:        int   = cfg["training"]["epochs"]
    warmup_epochs: int   = cfg["training"].get("warmup_epochs", 3)
    lr:            float = cfg["training"]["learning_rate"]

    # ── Banner ────────────────────────────────────────────────────────────────
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  GalaxEye Change Detection — Training")
    print(f"{sep}")
    print(f"  Epochs        : {epochs}")
    print(f"  Warmup epochs : {warmup_epochs}")
    print(f"  Batch size    : {cfg['training']['batch_size']}")
    print(f"  Image size    : {cfg['dataset']['image_size']}")
    print(f"  Learning rate : {lr}")
    print(f"  Phase 1       : Encoder frozen, epochs 1–{warmup_epochs}")
    # FIX-4: Phase-2 start is derived at runtime, not hardcoded in a comment.
    print(f"  Phase 2       : End-to-end, epochs {warmup_epochs + 1}–{epochs}")
    print(f"{sep}\n")

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_seed(cfg["seed"])

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM   : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    print()

    # ── Dataloaders ───────────────────────────────────────────────────────────
    print("Building dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders({
        "image_size" : cfg["dataset"]["image_size"],
        "batch_size" : cfg["training"]["batch_size"],
        "num_workers": cfg["dataset"]["num_workers"],
        "seed"       : cfg["seed"],
    })

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model …")
    model: nn.Module = build_model(cfg).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    # FocalDiceLoss combines Focal loss (handles class imbalance by down-
    # weighting easy negatives) with Dice loss (directly optimises overlap).
    criterion = FocalDiceLoss(
        focal_weight = cfg["loss"]["focal_weight"],
        dice_weight  = cfg["loss"]["dice_weight"],
        focal_alpha  = cfg["loss"]["focal_alpha"],
        focal_gamma  = cfg["loss"]["focal_gamma"],
    )

    # ── Output directories ────────────────────────────────────────────────────
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs_dir"],       exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 SETUP — Freeze encoder; train decoder + head only
    # ─────────────────────────────────────────────────────────────────────────
    # Freezing the encoder prevents random decoder gradients from corrupting
    # pretrained ImageNet features before the decoder has stabilised.
    for param in model.encoder.parameters():
        param.requires_grad = False

    # AdamW on unfrozen params only; the frozen encoder params are excluded
    # because filter() skips requires_grad=False tensors.
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = lr,
        weight_decay = cfg["training"]["weight_decay"],
    )

    # Cosine schedule for Phase 1; decays over warmup_epochs.
    # A brand-new scheduler is created for Phase 2 (see FIX-1 below),
    # so T_max here covers Phase 1 only.
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max   = warmup_epochs,
        eta_min = 1e-6,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Training bookkeeping
    # ─────────────────────────────────────────────────────────────────────────
    best_f1:    float             = 0.0
    best_epoch: int               = 0
    history:    list[dict]        = []

    # FIX-2: Logits for post-training threshold search are captured inside
    # validate() via return_logits=True and stored here.  No extra val pass.
    best_val_logits:  list[torch.Tensor] = []
    best_val_targets: list[torch.Tensor] = []

    print("\nStarting training …\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):

        # ── PHASE 2 TRANSITION ────────────────────────────────────────────────
        if epoch == warmup_epochs + 1:
            print(f"\n{'*' * 55}")
            print(f"  Phase 2 START — epoch {epoch} — Encoder UNFROZEN")
            print(f"{'*' * 55}\n")

            # Unfreeze the encoder so its weights can be updated.
            for param in model.encoder.parameters():
                param.requires_grad = True

            # Rebuild optimizer with differential learning rates:
            #   Encoder  → lr × 0.1  — gentle nudges to preserve good features.
            #   Decoder  → lr        — aggressive updates to learn change maps.
            #   Head     → lr        — same reasoning as decoder.
            optimizer = AdamW(
                [
                    {"params": model.encoder.parameters(),          "lr": lr * 0.1},
                    {"params": model.decoder.parameters(),          "lr": lr},
                    {"params": model.segmentation_head.parameters(),"lr": lr},
                ],
                weight_decay=cfg["training"]["weight_decay"],
            )

            # FIX-1: Rebuild the scheduler pointing at the NEW Phase-2 optimizer.
            # The old scheduler held a reference to the Phase-1 optimizer.
            # Calling scheduler.step() on it after the optimizer swap would step
            # the wrong parameter groups and leave the newly unfrozen encoder LR
            # group completely unscheduled for the entire second phase.
            # T_max = remaining epochs so the cosine decay reaches eta_min at
            # the very last epoch, matching the intended training curve.
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max   = epochs - warmup_epochs,   # remaining epochs
                eta_min = 1e-6,
            )

            print(f"  Encoder LR : {lr * 0.1:.2e}")
            print(f"  Decoder LR : {lr:.2e}")
            print(f"  Head    LR : {lr:.2e}\n")

        # ── Train ─────────────────────────────────────────────────────────────
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, epochs
        )

        # ── Validate ──────────────────────────────────────────────────────────
        # FIX-2: return_logits=True piggybacks logit collection onto the single
        # mandatory val pass.  Previously a full second pass was triggered every
        # time a new best model was found, doubling validation cost at each
        # checkpoint.
        val_loss, val_results, val_metrics_obj, val_logits, val_targets = validate(
            model, val_loader, criterion, device,
            threshold     = cfg["evaluation"]["threshold"],
            return_logits = True,
        )

        # Step the scheduler AFTER the optimiser step (PyTorch convention).
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Epoch summary ─────────────────────────────────────────────────────
        print(f"\n── Epoch {epoch}/{epochs} ──────────────────────────────────")
        print(f"  Train Loss : {train_loss:.4f}")
        print(f"  Val   Loss : {val_loss:.4f}")
        print(f"  Val   IoU  : {val_results['iou']:.4f}")
        print(f"  Val   F1   : {val_results['f1']:.4f}")
        print(f"  Val   Prec : {val_results['precision']:.4f}")
        print(f"  Val   Rec  : {val_results['recall']:.4f}")
        print(f"  LR         : {current_lr:.2e}\n")

        history.append({
            "epoch"     : epoch,
            "train_loss": train_loss,
            "val_loss"  : val_loss,
            **val_results,
        })

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_results["f1"] > best_f1:
            best_f1    = val_results["f1"]
            best_epoch = epoch

            # FIX-2: Swap in the logits already collected above — no second pass.
            best_val_logits  = val_logits
            best_val_targets = val_targets

            save_checkpoint(
                {
                    "epoch"      : epoch,
                    "model_state": model.state_dict(),
                    "optimizer"  : optimizer.state_dict(),
                    "scheduler"  : scheduler.state_dict(),
                    "val_results": val_results,
                    "config"     : cfg,
                },
                cfg["paths"]["best_model"],
            )
            print(
                f"  *** Best model saved — "
                f"Val F1: {best_f1:.4f}  epoch: {best_epoch} ***\n"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Post-training: threshold search on the best validation set
    # ─────────────────────────────────────────────────────────────────────────
    # Grid-search over probability thresholds to maximise F1 on the val set.
    # The optimal threshold is then applied at test time — this is the
    # standard operating-point selection for imbalanced binary segmentation.
    print(f"\n{sep}")
    print("Post-training: threshold search on validation logits …")
    best_thr, best_thr_f1 = find_best_threshold(
        best_val_logits, best_val_targets
    )
    print(f"Optimal threshold : {best_thr:.2f}   F1 @ thr : {best_thr_f1:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Final evaluation — best checkpoint + optimal threshold
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (epoch {best_epoch}) …")
    ckpt = torch.load(cfg["paths"]["best_model"], map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Validation set — confirms threshold search did not over-fit to val.
    print(f"Evaluating on VALIDATION set (threshold={best_thr:.2f}) …")
    val_out = validate(model, val_loader, criterion, device, threshold=best_thr)
    final_val_results, final_val_metrics = val_out[1], val_out[2]
    final_val_metrics.print_results("Final Validation")
    val_cm = final_val_metrics.confusion_matrix()

    # Test set — held-out; this is the number that goes in the paper / report.
    print(f"Evaluating on TEST set (threshold={best_thr:.2f}) …")
    test_out = validate(model, test_loader, criterion, device, threshold=best_thr)
    test_results, test_metrics = test_out[1], test_out[2]
    test_metrics.print_results("Test")
    test_cm = test_metrics.confusion_matrix()

    # ─────────────────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────────────────
    plot_training_history(
        history,
        save_path=os.path.join(cfg["paths"]["logs_dir"], "training_history.png"),
    )
    plot_confusion_matrix(
        test_cm,
        save_path=os.path.join(cfg["paths"]["logs_dir"], "confusion_matrix_test.png"),
        title=f"Test Confusion Matrix (thr={best_thr:.2f})",
    )
    plot_confusion_matrix(
        val_cm,
        save_path=os.path.join(cfg["paths"]["logs_dir"], "confusion_matrix_val.png"),
        title="Validation Confusion Matrix",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Persist text results
    # ─────────────────────────────────────────────────────────────────────────
    results_path = os.path.join(cfg["paths"]["logs_dir"], "final_results.txt")
    with open(results_path, "w") as fh:
        fh.write("GalaxEye Change Detection — Final Results\n")
        fh.write(f"{'=' * 45}\n")
        fh.write(f"Best epoch     : {best_epoch}/{epochs}\n")
        fh.write(f"Best threshold : {best_thr}\n\n")

        fh.write("Validation Results:\n")
        for k, v in final_val_results.items():
            fh.write(f"  {k:12s}: {v}\n")

        fh.write("\nTest Results:\n")
        for k, v in test_results.items():
            fh.write(f"  {k:12s}: {v}\n")

        fh.write("\nTraining History:\n")
        fh.write(
            f"{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>9} "
            f"{'IoU':>7} {'F1':>7} {'Prec':>7} {'Rec':>7}\n"
        )
        for h in history:
            fh.write(
                f"{h['epoch']:>6} {h['train_loss']:>10.4f} "
                f"{h['val_loss']:>9.4f} "
                f"{h['iou']:>7.4f} {h['f1']:>7.4f} "
                f"{h['precision']:>7.4f} {h['recall']:>7.4f}\n"
            )

    # ── Final banner ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best epoch     : {best_epoch}/{epochs}")
    print(f"  Best Val F1    : {best_f1:.4f}")
    print(f"  Best threshold : {best_thr:.2f}")
    print(f"  Results saved  : {results_path}")
    print(f"  Model saved    : {cfg['paths']['best_model']}")
    print(f"{sep}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GalaxEye Change Detection — training script"
    )
    parser.add_argument(
        "--config",
        type    = str,
        default = "config.yaml",
        help    = "Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)