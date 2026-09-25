"""
predict.py
----------
Single-image inference with:
  - Optic disc & cup segmentation overlay
  - vCDR and aCDR computation
  - Glaucoma risk classification
  - Overlay visualisation saved to disk

Usage:
    python predict.py --image fundus.jpg --checkpoint checkpoints/best_model.pth
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast

from model.segformer_model import GlaucomaSegFormer
from utils import load_checkpoint, mask_to_rgb, CLASS_COLORS


# ─────────────────────────────────────────
#  PRE-PROCESS A SINGLE IMAGE
# ─────────────────────────────────────────

def preprocess(image_path, img_size=512):
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    img = np.array(Image.open(image_path).convert("RGB"))
    return transform(image=img)["image"].unsqueeze(0), img   # (1,3,H,W), orig_np


# ─────────────────────────────────────────
#  CDR COMPUTATION  (same as evaluate.py)
# ─────────────────────────────────────────

def compute_cdr(pred_mask):
    od_mask = (pred_mask == 1).astype(np.uint8)
    oc_mask = (pred_mask == 2).astype(np.uint8)

    od_area = od_mask.sum()
    oc_area = oc_mask.sum()
    acdr    = oc_area / od_area if od_area > 0 else 0.0

    def vdiam(m):
        rows = np.where(m.any(axis=1))[0]
        return (rows[-1] - rows[0] + 1) if len(rows) else 0

    vcdr = vdiam(oc_mask) / vdiam(od_mask) if vdiam(od_mask) > 0 else 0.0
    return float(vcdr), float(acdr)


def glaucoma_risk(vcdr):
    if vcdr < 0.5:
        return "Low Risk", "green"
    elif vcdr < 0.7:
        return "Moderate Risk", "orange"
    else:
        return "High Risk - Suspect Glaucoma", "red"


# ─────────────────────────────────────────
#  OVERLAY CREATION
# ─────────────────────────────────────────

def create_overlay(orig_img_np, pred_mask, alpha=0.45):
    """Blend colour-coded segmentation mask over the original fundus image."""
    resized = cv2.resize(orig_img_np, (pred_mask.shape[1], pred_mask.shape[0]))
    seg_rgb = mask_to_rgb(pred_mask)
    overlay = cv2.addWeighted(resized, 1 - alpha, seg_rgb, alpha, 0)
    return overlay


# ─────────────────────────────────────────
#  MAIN INFERENCE
# ─────────────────────────────────────────

def predict(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load model ────────────────────────
    model = GlaucomaSegFormer().to(device)
    load_checkpoint(model, None, args.checkpoint, device)
    model.eval()

    # ── Preprocess ────────────────────────
    tensor, orig_np = preprocess(args.image, img_size=args.img_size)
    tensor = tensor.to(device)

    # ── Predict ───────────────────────────
    with torch.no_grad(), autocast():
        logits = model(tensor)
    pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)

    # ── CDR ───────────────────────────────
    vcdr, acdr  = compute_cdr(pred_mask)
    risk, color = glaucoma_risk(vcdr)

    print("\n" + "-" * 45)
    print(f"  Image   : {os.path.basename(args.image)}")
    print(f"  vCDR    : {vcdr:.3f}")
    print(f"  aCDR    : {acdr:.3f}")
    print(f"  Risk    : {risk}")
    print("-" * 45)

    # ── Visualise ─────────────────────────
    overlay = create_overlay(orig_np, pred_mask)
    seg_rgb = mask_to_rgb(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(cv2.resize(orig_np, (args.img_size, args.img_size)))
    axes[0].set_title("Original Fundus Image", fontsize=12)

    axes[1].imshow(seg_rgb)
    axes[1].set_title("Segmentation Mask", fontsize=12)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12)

    # Add CDR info as figure text
    info = (f"vCDR: {vcdr:.3f}   |   aCDR: {acdr:.3f}   |   "
            f"Risk: {risk}")
    fig.text(0.5, 0.02, info, ha="center", fontsize=12,
             color=color, fontweight="bold")

    # Legend
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=np.array(CLASS_COLORS[1])/255, label="Optic Disc"),
        mpatches.Patch(color=np.array(CLASS_COLORS[2])/255, label="Optic Cup"),
    ]
    fig.legend(handles=patches, loc="lower right", fontsize=10)
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    stem      = os.path.splitext(os.path.basename(args.image))[0]
    save_path = os.path.join(args.output_dir, f"{stem}_prediction.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved prediction → {save_path}")
    plt.show()
    plt.close()

    return {"vcdr": vcdr, "acdr": acdr, "risk": risk, "mask": pred_mask}


# ─────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, default=None,
                   help="Path to the fundus image to segment. Defaults to the first sample image in data/test/images if omitted.")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--output_dir", type=str, default="./predictions")
    args = p.parse_args()

    if args.image is None:
        for candidate in [
            os.path.join("data", "test", "images", "T0001.jpg"),
            os.path.join("data", "val", "images", "T0001.jpg"),
            os.path.join("data", "train", "images", "T0001.jpg"),
        ]:
            if os.path.exists(candidate):
                args.image = candidate
                break

    if args.image is None:
        raise SystemExit(
            "No image provided. Use --image /path/to/image.jpg or place an image in data/test/images."
        )

    return args


if __name__ == "__main__":
    predict(parse_args())
