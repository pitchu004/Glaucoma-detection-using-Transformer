"""
utils.py
--------
Shared helper functions:
  - Metrics (IoU, Dice, Accuracy, Precision, Recall, F1)
  - Visualisation
  - Checkpoint save/load
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2


# ─────────────────────────────────────────
#  SEGMENTATION METRICS
# ─────────────────────────────────────────

class SegmentationMetrics:
    """
    Accumulates per-batch predictions and computes
    pixel-level accuracy, precision, recall, F1, and IoU
    for multi-class segmentation (3 classes).
    """

    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes),
                                  dtype=np.int64)

    def update(self, preds, targets):
        """
        preds   : (B, H, W) integer class predictions
        targets : (B, H, W) integer ground-truth masks
        """
        preds_np   = preds.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        for t, p in zip(targets_np, preds_np):
            self.confusion[t][p] += 1

    def compute(self):
        cm = self.confusion.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)

        # Per-class
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1        = np.where(precision + recall > 0,
                             2 * precision * recall / (precision + recall), 0.0)
        iou       = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
        accuracy  = tp.sum() / cm.sum()

        return {
            "accuracy"  : float(accuracy),
            "precision" : float(np.mean(precision)),
            "recall"    : float(np.mean(recall)),
            "f1_score"  : float(np.mean(f1)),
            "mean_iou"  : float(np.mean(iou)),
            # Per-class detail
            "per_class" : {
                "precision": precision.tolist(),
                "recall"   : recall.tolist(),
                "f1"       : f1.tolist(),
                "iou"      : iou.tolist(),
            }
        }

    def print_summary(self):
        m = self.compute()
        print("-" * 45)
        print(f"  Pixel Accuracy : {m['accuracy']*100:.2f}%")
        print(f"  Mean Precision : {m['precision']*100:.2f}%")
        print(f"  Mean Recall    : {m['recall']*100:.2f}%")
        print(f"  Mean F1 Score  : {m['f1_score']*100:.2f}%")
        print(f"  Mean IoU       : {m['mean_iou']*100:.2f}%")
        print("-" * 45)
        labels = ["Background", "Optic Disc", "Optic Cup"]
        for i, lbl in enumerate(labels):
            pc = m["per_class"]
            print(f"  {lbl:12s}: "
                  f"P={pc['precision'][i]*100:.1f}%  "
                  f"R={pc['recall'][i]*100:.1f}%  "
                  f"F1={pc['f1'][i]*100:.1f}%  "
                  f"IoU={pc['iou'][i]*100:.1f}%")
        print("-" * 45)


# ─────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────

CLASS_COLORS = {
    0: (0,   0,   0),    # Background  – black
    1: (0, 255,   0),    # Optic Disc  – green
    2: (255,  0,   0),   # Optic Cup   – red
}


def mask_to_rgb(mask_np):
    """Convert (H, W) class index mask → (H, W, 3) RGB."""
    h, w = mask_np.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask_np == cls] = color
    return rgb


def visualise_prediction(image_tensor, gt_mask, pred_mask, save_path=None):
    """
    image_tensor : (3, H, W) normalised torch tensor
    gt_mask      : (H, W) numpy int array
    pred_mask    : (H, W) numpy int array
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = image_tensor.permute(1, 2, 0).cpu().numpy()
    img  = (img * std + mean).clip(0, 1)

    gt_rgb   = mask_to_rgb(gt_mask)
    pred_rgb = mask_to_rgb(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);            axes[0].set_title("Fundus Image")
    axes[1].imshow(gt_rgb / 255);   axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_rgb / 255); axes[2].set_title("Prediction")

    patches = [
        mpatches.Patch(color=np.array(c) / 255, label=l)
        for l, c in [("Background", CLASS_COLORS[0]),
                     ("Optic Disc", CLASS_COLORS[1]),
                     ("Optic Cup",  CLASS_COLORS[2])]
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualisation → {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  CHECKPOINTING
# ─────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, metric, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch"     : epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "metric"    : metric,
    }, path)
    print(f"Checkpoint saved → {path}  (epoch {epoch}, metric {metric:.4f})")


def load_checkpoint(model, optimizer, path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    print(f"Loaded checkpoint from {path}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["metric"]


def plot_training_curves(train_losses, val_losses, val_f1s, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label="Train Loss", color="blue")
    ax1.plot(val_losses,   label="Val Loss",   color="orange")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(val_f1s, label="Val F1", color="green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 Score"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
