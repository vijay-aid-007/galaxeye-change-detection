"""
metrics.py — GalaxEye Change Detection

All metrics computed on the CHANGE class (label=1) only.
This is mandatory per GalaxEye spec — accuracy is NOT reported
because 99.39% no-change pixels make it meaningless.

Metrics reported:
  - IoU  (Intersection over Union) on change class
  - Precision on change class
  - Recall on change class
  - F1 Score on change class
  - Confusion matrix (2x2)
"""

import torch
import numpy as np


class ChangeMetrics:
    """
    Accumulates predictions across batches, computes final metrics.

    Usage:
        metrics = ChangeMetrics()
        for batch in loader:
            preds, targets = model(batch), batch['mask']
            metrics.update(preds, targets)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.reset()

    def reset(self):
        # Confusion matrix entries for binary case
        self.tp = 0  # true positives  (predicted change, actually change)
        self.fp = 0  # false positives (predicted change, actually no-change)
        self.tn = 0  # true negatives  (predicted no-change, actually no-change)
        self.fn = 0  # false negatives (predicted no-change, actually change)

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits  : raw model output, shape [B, 1, H, W] or [B, H, W]
            targets : ground truth binary mask, shape [B, H, W], values {0,1}
        """
        # Convert logits to binary predictions
        if logits.dim() == 4:
            logits = logits.squeeze(1)  # [B, H, W]

        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).long()
        targets = targets.long()

        # Flatten for computation
        preds   = preds.view(-1)
        targets = targets.view(-1)

        self.tp += ((preds == 1) & (targets == 1)).sum().item()
        self.fp += ((preds == 1) & (targets == 0)).sum().item()
        self.tn += ((preds == 0) & (targets == 0)).sum().item()
        self.fn += ((preds == 0) & (targets == 1)).sum().item()

    def compute(self) -> dict:
        """
        Returns dict with all metrics. All computed on change class (label=1).
        """
        eps = 1e-8  # prevent division by zero

        precision = self.tp / (self.tp + self.fp + eps)
        recall    = self.tp / (self.tp + self.fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        iou       = self.tp / (self.tp + self.fp + self.fn + eps)

        # Pixel accuracy (reported but not primary metric)
        total    = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / (total + eps)

        return {
            'iou'      : round(iou, 4),
            'f1'       : round(f1, 4),
            'precision': round(precision, 4),
            'recall'   : round(recall, 4),
            'accuracy' : round(accuracy, 4),
            'tp'       : self.tp,
            'fp'       : self.fp,
            'tn'       : self.tn,
            'fn'       : self.fn,
        }

    def confusion_matrix(self) -> np.ndarray:
        """
        Returns 2x2 confusion matrix:
        [[TN, FP],
         [FN, TP]]
        """
        return np.array([
            [self.tn, self.fp],
            [self.fn, self.tp],
        ])

    def print_results(self, split_name: str = 'Validation'):
        results = self.compute()
        cm = self.confusion_matrix()
        print(f"\n{'='*45}")
        print(f"  {split_name} Results (Change Class)")
        print(f"{'='*45}")
        print(f"  IoU       : {results['iou']:.4f}")
        print(f"  F1 Score  : {results['f1']:.4f}")
        print(f"  Precision : {results['precision']:.4f}")
        print(f"  Recall    : {results['recall']:.4f}")
        print(f"  Accuracy  : {results['accuracy']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"              Pred No-Change  Pred Change")
        print(f"  GT No-Change  {cm[0,0]:>12,}  {cm[0,1]:>11,}")
        print(f"  GT Change     {cm[1,0]:>12,}  {cm[1,1]:>11,}")
        print(f"{'='*45}\n")
        return results


# ------------------------------------------------------------------
# Threshold search — find best F1 threshold on validation set
# ------------------------------------------------------------------
def find_best_threshold(logits_list, targets_list, thresholds=None):
    """
    Searches for the probability threshold that maximises F1 on val set.

    Why this matters: with 0.61% change pixels, the default threshold
    of 0.5 may not be optimal. A lower threshold increases recall
    at the cost of precision. We pick the threshold that maximises F1.

    Args:
        logits_list  : list of torch tensors (raw model output per batch)
        targets_list : list of torch tensors (ground truth per batch)
        thresholds   : list of floats to try (default: 0.1 to 0.9)

    Returns:
        best_threshold : float
        best_f1        : float
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]

    best_f1 = 0.0
    best_threshold = 0.5

    print("Threshold search:")
    for thr in thresholds:
        m = ChangeMetrics(threshold = thr)
        for logits, targets in zip(logits_list, targets_list):
            m.update(logits, targets)
        results = m.compute()
        f1 = results['f1']
        print(f"  threshold={thr:.2f}  "
              f"F1={f1:.4f}  "
              f"Prec={results['precision']:.4f}  "
              f"Rec={results['recall']:.4f}  "
              f"IoU={results['iou']:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thr

    print(f"\nBest threshold: {best_threshold:.2f}  Best F1: {best_f1:.4f}")
    return best_threshold, best_f1


# ------------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("=== metrics.py sanity check ===")

    m = ChangeMetrics(threshold=0.5)

    # Simulate perfect predictions
    perfect_logits  = torch.tensor([10.0, -10.0, 10.0, -10.0])
    perfect_targets = torch.tensor([1, 0, 1, 0])
    m.update(perfect_logits.unsqueeze(0).unsqueeze(0),
             perfect_targets.unsqueeze(0))
    r = m.compute()
    assert r['iou'] == 1.0, "Perfect IoU should be 1.0"
    assert r['f1']  == 1.0, "Perfect F1 should be 1.0"
    print("Perfect prediction test: PASSED")

    m.reset()

    # Simulate all-zero prediction (the naive baseline)
    zero_logits  = torch.full((1, 1, 4, 4), -10.0)
    zero_targets = torch.zeros(1, 4, 4).long()
    zero_targets[0, 0, 0] = 1  # one changed pixel
    m.update(zero_logits, zero_targets)
    r = m.compute()
    assert r['iou'] == 0.0, "All-zero pred should have 0 IoU"
    assert r['f1']  == 0.0, "All-zero pred should have 0 F1"
    print("All-zero prediction test: PASSED")
    print("metrics.py is working correctly.")       