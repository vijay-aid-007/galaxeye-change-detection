"""
utils.py — GalaxEye Change Detection
Utility functions: visualization, checkpoint management, logging.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path: str, model, optimizer=None, device='cpu'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f"  Loaded checkpoint: {path}  "
          f"(epoch={ckpt.get('epoch','?')})")
    return ckpt


def plot_training_history(history: list, save_path: str = 'logs/training_history.png'):
    """
    Plots loss and metrics curves from training history.
    Call after training completes.
    """
    epochs      = [h['epoch']      for h in history]
    train_loss  = [h['train_loss'] for h in history]
    val_loss    = [h['val_loss']   for h in history]
    val_f1      = [h['f1']         for h in history]
    val_iou     = [h['iou']        for h in history]
    val_prec    = [h['precision']  for h in history]
    val_recall  = [h['recall']     for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, train_loss, label='Train Loss', color='blue')
    axes[0].plot(epochs, val_loss,   label='Val Loss',   color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 and IoU
    axes[1].plot(epochs, val_f1,  label='Val F1',  color='green')
    axes[1].plot(epochs, val_iou, label='Val IoU', color='red')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].set_title('F1 and IoU (Change Class)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Precision vs Recall
    axes[2].plot(epochs, val_prec,   label='Precision', color='purple')
    axes[2].plot(epochs, val_recall, label='Recall',    color='teal')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Precision vs Recall (Change Class)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Training History — GalaxEye Change Detection',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training history plot saved: {save_path}")


def plot_confusion_matrix(cm: np.ndarray,
                           save_path: str = 'logs/confusion_matrix.png',
                           title: str = 'Confusion Matrix'):
    """
    Plots and saves a clean confusion matrix.

    cm format:
    [[TN, FP],
     [FN, TP]]
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)

    labels = ['No-Change (0)', 'Change (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title(title)

    # Annotate cells
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = 100 * count / (total + 1e-8)
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{count:,}\n({pct:.1f}%)',
                    ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")