"""
eval.py — GalaxEye Change Detection
Standalone evaluation script.

Usage:
    python eval.py --config config.yaml \
                   --weights checkpoints/best_model.pth \
                   --split test \
                   --threshold 0.35 \
                   --visualize

What this does:
  1. Loads trained model weights
  2. Runs inference on val or test split
  3. Reports IoU, F1, Precision, Recall, Confusion Matrix
  4. Optionally saves prediction visualizations
"""

import os
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.model   import build_model
from src.metrics import ChangeMetrics
from datasets import load_dataset, DatasetDict
from typing import cast

from src.dataset    import build_dataloaders, ChangeDetectionDataset
from src.transforms import get_val_transforms


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device, threshold):
    model.eval()
    metrics = ChangeMetrics(threshold=threshold)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            metrics.update(logits.cpu(), masks.cpu())

    return metrics


def visualize_predictions(model, hf_split, block_size,
                           device, threshold, cfg,
                           n_samples=8, save_dir='outputs/predictions'):
    """
    Saves side-by-side visualizations:
    EO | SAR | Ground Truth | Prediction | Overlay

    Covers both success cases and failure cases.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    transform = get_val_transforms(cfg['dataset']['image_size'])

    # Pick varied samples: low-change and high-change scenes
    indices = [0, 10, 50, 100, 200, 300, 310, 320][:n_samples]

    import numpy as np

    for idx in indices:
        # Raw images (no transform, for display)
        sar_raw  = np.array(hf_split[idx]['image'])
        eo_raw   = np.array(hf_split[idx + block_size]['image'])
        mask_raw = np.array(hf_split[idx + 2 * block_size]['image'])

        # Remapped ground truth
        gt = np.where(mask_raw >= 2, 1, 0).astype(np.uint8)

        # Model input (transformed)
        import torch
        sar_3d = sar_raw[:, :, np.newaxis]
        image_np = np.concatenate([eo_raw, sar_3d], axis=-1).astype(np.float32)
        augmented = transform(image=image_np, mask=gt)
        image_t = augmented['image'].unsqueeze(0).to(device)  # [1,4,H,W]

        with torch.no_grad():
            logit = model(image_t)
            prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred  = (prob >= threshold).astype(np.uint8)

        # Metrics for this sample
        tp = int(((pred == 1) & (gt[:pred.shape[0], :pred.shape[1]] == 1)).sum())
        fp = int(((pred == 1) & (gt[:pred.shape[0], :pred.shape[1]] == 0)).sum())
        fn = int(((pred == 0) & (gt[:pred.shape[0], :pred.shape[1]] == 1)).sum())
        eps = 1e-8
        sample_iou = tp / (tp + fp + fn + eps)
        change_pct = 100 * gt.mean()

        # Crop display images to match transform output size
        H, W = pred.shape
        eo_disp   = eo_raw[:H, :W]
        sar_disp  = sar_raw[:H, :W]
        gt_crop   = gt[:H, :W]

        # Overlay: TP=green, FP=red, FN=yellow, TN=black
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        overlay[(pred == 1) & (gt_crop == 1)] = [0,   200, 0  ]  # TP green
        overlay[(pred == 1) & (gt_crop == 0)] = [200, 0,   0  ]  # FP red
        overlay[(pred == 0) & (gt_crop == 1)] = [255, 215, 0  ]  # FN yellow

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))

        axes[0].imshow(eo_disp)
        axes[0].set_title('EO Pre-Event')
        axes[0].axis('off')

        axes[1].imshow(sar_disp, cmap='gray')
        axes[1].set_title('SAR Post-Event')
        axes[1].axis('off')

        axes[2].imshow(gt_crop, cmap='Reds', vmin=0, vmax=1)
        axes[2].set_title(f'Ground Truth\n({change_pct:.1f}% changed)')
        axes[2].axis('off')

        axes[3].imshow(prob, cmap='hot', vmin=0, vmax=1)
        axes[3].set_title(f'Prediction Prob\n(threshold={threshold})')
        axes[3].axis('off')

        axes[4].imshow(overlay)
        tp_patch = mpatches.Patch(color='green',  label='TP (correct change)')
        fp_patch = mpatches.Patch(color='red',    label='FP (false alarm)')
        fn_patch = mpatches.Patch(color='yellow', label='FN (missed change)')
        axes[4].legend(handles=[tp_patch, fp_patch, fn_patch],
                       loc='lower right', fontsize=7)
        axes[4].set_title(f'Error Overlay\nIoU={sample_iou:.3f}')
        axes[4].axis('off')

        plt.suptitle(f'Sample {idx}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'sample_{idx:04d}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}  "
              f"(IoU={sample_iou:.3f}, change={change_pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    type=str,   default='config.yaml')
    parser.add_argument('--weights',   type=str,   required=True,
                        help='Path to model checkpoint .pth file')
    parser.add_argument('--split',     type=str,   default='test',
                        choices=['train', 'validation', 'test'])
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override threshold from config')
    parser.add_argument('--visualize', action='store_true',
                        help='Save prediction visualizations')
    args = parser.parse_args()

    cfg = load_config(args.config)
    threshold = args.threshold or cfg['evaluation']['threshold']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load model
    print(f"Loading model weights from: {args.weights}")
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    if 'val_results' in ckpt:
        vr = ckpt['val_results']
        print(f"  Checkpoint val F1={vr.get('f1','?')}  "
              f"IoU={vr.get('iou','?')}")

    # Build loaders
    train_loader, val_loader, test_loader = build_dataloaders({
        'image_size' : cfg['dataset']['image_size'],
        'batch_size' : cfg['training']['batch_size'],
        'num_workers': cfg['dataset']['num_workers'],
        'seed'       : cfg['seed'],
    })

    loader_map = {
        'train'     : train_loader,
        'validation': val_loader,
        'test'      : test_loader,
    }
    loader = loader_map[args.split]

    # Evaluate
    print(f"\nEvaluating on [{args.split}] split "
          f"with threshold={threshold}...")
    metrics = evaluate(model, loader, device, threshold)
    metrics.print_results(args.split.capitalize())

    # Visualize
    if args.visualize:
        print("\nGenerating prediction visualizations...")
        ds = cast(DatasetDict, load_dataset("doron333/change-detection-dataset"))

        split_map = {
            'train'     : ('train',      2781),
            'validation': ('validation', 334),
            'test'      : ('test',       77),
        }
        hf_key, block_size = split_map[args.split]
        visualize_predictions(
            model       = model,
            hf_split    = ds[hf_key],
            block_size  = block_size,
            device      = device,
            threshold   = threshold,
            cfg         = cfg,
            n_samples   = 8,
            save_dir    = f'outputs/predictions_{args.split}',
        )
        print("Visualizations saved.")


if __name__ == '__main__':
    main()