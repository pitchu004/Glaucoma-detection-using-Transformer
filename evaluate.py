"""
evaluate.py
-----------
Full evaluation on the test set with:
  - Pixel accuracy, precision, recall, F1, IoU
  - Confusion matrix
  - Qualitative visualisations
  - CDR statistics across the test set

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
                       --data_dir   ./data/REFUGE2
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.cuda.amp import autocast

from data.prepare_dataset import get_dataloaders
from model.segformer_model import GlaucomaSegFormer
from utils import SegmentationMetrics, load_checkpoint, visualise_prediction


# ─────────────────────────────────────────
#  CDR COMPUTATION
# ─────────────────────────────────────────

def compute_cdr(pred_mask):
    """
    Compute Vertical CDR and Area CDR from a predicted segmentation mask.

    pred_mask : (H, W) numpy array with values {0, 1, 2}
                 1 = Optic Disc, 2 = Optic Cup

    Returns:
        vcdr (float) : Vertical Cup-to-Disc Ratio
        acdr (float) : Area   Cup-to-Disc Ratio
    """
    od_mask = (pred_mask == 1).astype(np.uint8)   # Optic Disc
    oc_mask = (pred_mask == 2).astype(np.uint8)   # Optic Cup

    # ── Area CDR ──────────────────────────────
    od_area = od_mask.sum()
    oc_area = oc_mask.sum()
    acdr = oc_area / od_area if od_area > 0 else 0.0

    # ── Vertical CDR ──────────────────────────
    # Find bounding box rows for OD and OC
    def vertical_diameter(mask):
        rows = np.where(mask.any(axis=1))[0]
        return (rows[-1] - rows[0] + 1) if len(rows) > 0 else 0

    od_vdiam = vertical_diameter(od_mask)
    oc_vdiam = vertical_diameter(oc_mask)
    vcdr = oc_vdiam / od_vdiam if od_vdiam > 0 else 0.0

    return float(vcdr), float(acdr)


def glaucoma_risk(vcdr):
    """Clinical threshold-based risk classification."""
    if vcdr < 0.5:
        return "Low Risk"
    elif vcdr < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk (Suspect Glaucoma)"


# ─────────────────────────────────────────
#  CONFUSION MATRIX PLOT
# ─────────────────────────────────────────

def plot_confusion_matrix(cm, save_path=None):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    labels  = ["Background", "Optic Disc", "Optic Cup"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 12},
    )
    ax.set_xlabel("Predicted",   fontsize=12)
    ax.set_ylabel("True",        fontsize=12)
    ax.set_title("Normalised Confusion Matrix (%)", fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved → {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  MAIN EVALUATION
# ─────────────────────────────────────────

def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}")

    # ── Data ──────────────────────────────
    _, _, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────
    model = GlaucomaSegFormer().to(device)
    load_checkpoint(model, None, args.checkpoint, device)
    model.eval()

    # ── Evaluation Loop ───────────────────
    metrics = SegmentationMetrics(num_classes=3)
    all_vcdr, all_acdr = [], []
    vis_count = 0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(
                tqdm(test_loader, desc="Evaluating")):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with autocast():
                logits = model(images)

            preds = logits.argmax(dim=1)   # (B, H, W)
            metrics.update(preds, masks)

            # CDR computation per image
            for i in range(preds.shape[0]):
                pred_np = preds[i].cpu().numpy()
                vcdr, acdr = compute_cdr(pred_np)
                all_vcdr.append(vcdr)
                all_acdr.append(acdr)

            # Save a few qualitative visualisations
            if vis_count < args.num_vis:
                for i in range(min(preds.shape[0], args.num_vis - vis_count)):
                    vcdr, acdr = compute_cdr(preds[i].cpu().numpy())
                    save_p = f"{args.output_dir}/vis/sample_{vis_count}.png"
                    visualise_prediction(
                        images[i], masks[i].cpu().numpy(),
                        preds[i].cpu().numpy(), save_path=save_p,
                    )
                    vis_count += 1

    # ── Print Results ─────────────────────
    print("\n" + "="*50)
    print("  TEST SET RESULTS")
    print("="*50)
    metrics.print_summary()

    print(f"\n  CDR Statistics over {len(all_vcdr)} images:")
    print(f"    Mean vCDR : {np.mean(all_vcdr):.3f} ± {np.std(all_vcdr):.3f}")
    print(f"    Mean aCDR : {np.mean(all_acdr):.3f} ± {np.std(all_acdr):.3f}")

    # Risk distribution
    risks = [glaucoma_risk(v) for v in all_vcdr]
    from collections import Counter
    risk_counts = Counter(risks)
    print("\n  Glaucoma Risk Distribution:")
    for k, v in risk_counts.items():
        pct = v / len(risks) * 100
        print(f"    {k:<35} : {v:4d} ({pct:.1f}%)")

    # ── Confusion Matrix ──────────────────
    plot_confusion_matrix(
        metrics.confusion,
        save_path=f"{args.output_dir}/confusion_matrix.png",
    )


# ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, default="checkpoints/best_model.pth")
    p.add_argument("--data_dir",    type=str, default="./data/REFUGE2")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--img_size",    type=int, default=512)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_vis",     type=int, default=10)
    p.add_argument("--output_dir",  type=str, default="./results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate(args)
