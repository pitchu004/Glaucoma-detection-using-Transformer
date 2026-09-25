"""
show_preprocessing.py
---------------------
Visually shows what preprocessing and augmentation does to fundus images.
Saves comparison images to results/preprocessing/ folder.

Run:
    python show_preprocessing.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

os.makedirs("results/preprocessing", exist_ok=True)

# ─────────────────────────────────────────
#  PICK A SAMPLE IMAGE FROM YOUR DATASET
# ─────────────────────────────────────────

IMAGE_PATH = "./data/train/images/n0001.jpg"   # change if needed
MASK_PATH  = "./data/train/mask/n0001.bmp"     # change if needed

# Auto find first image if above path doesn't exist
if not os.path.exists(IMAGE_PATH):
    img_dir = "./data/train/images"
    files   = sorted(os.listdir(img_dir))
    IMAGE_PATH = os.path.join(img_dir, files[0])
    stem = os.path.splitext(files[0])[0]
    mask_dir   = "./data/train/mask"
    mask_files = sorted(os.listdir(mask_dir))
    MASK_PATH  = os.path.join(mask_dir, mask_files[0])

print(f"Using image : {IMAGE_PATH}")
print(f"Using mask  : {MASK_PATH}")

# Load original
orig_image = np.array(Image.open(IMAGE_PATH).convert("RGB"))
orig_mask  = np.array(Image.open(MASK_PATH).convert("L"))


# ─────────────────────────────────────────
#  GRAPH 1 — PREPROCESSING STEPS
#  Shows: Original → Resized → Normalized
# ─────────────────────────────────────────

def show_preprocessing_steps():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Preprocessing Pipeline", fontsize=15, fontweight="bold")

    # Step 1 — Original
    axes[0].imshow(orig_image)
    axes[0].set_title(f"Step 1: Original Image\nSize: {orig_image.shape[1]}×{orig_image.shape[0]} px", fontsize=11)
    axes[0].axis("off")

    # Step 2 — Resized
    resized = cv2.resize(orig_image, (512, 512))
    axes[1].imshow(resized)
    axes[1].set_title("Step 2: Resized\n512×512 px", fontsize=11)
    axes[1].axis("off")

    # Step 3 — Normalized (de-normalize for display)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    normalized = (resized / 255.0 - mean) / std
    # Clip to 0-1 for display
    display_norm = np.clip((normalized - normalized.min()) /
                           (normalized.max() - normalized.min()), 0, 1)
    axes[2].imshow(display_norm)
    axes[2].set_title("Step 3: Normalized\n(pixel values scaled to 0-1)", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    path = "results/preprocessing/1_preprocessing_steps.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  GRAPH 2 — AUGMENTATION SAMPLES
#  Shows 8 different augmented versions
# ─────────────────────────────────────────

def show_augmentation_samples():
    resized = cv2.resize(orig_image, (512, 512))

    augmentations = [
        ("Original",              A.Compose([A.Resize(512,512)])),
        ("Horizontal Flip",       A.Compose([A.Resize(512,512), A.HorizontalFlip(p=1.0)])),
        ("Vertical Flip",         A.Compose([A.Resize(512,512), A.VerticalFlip(p=1.0)])),
        ("Rotation 90°",          A.Compose([A.Resize(512,512), A.RandomRotate90(p=1.0)])),
        ("Shift Scale Rotate",    A.Compose([A.Resize(512,512), A.ShiftScaleRotate(
                                     shift_limit=0.1, scale_limit=0.2,
                                     rotate_limit=30, p=1.0)])),
        ("Brightness/Contrast",   A.Compose([A.Resize(512,512),
                                     A.RandomBrightnessContrast(
                                     brightness_limit=0.4, contrast_limit=0.4, p=1.0)])),
        ("Hue Saturation",        A.Compose([A.Resize(512,512),
                                     A.HueSaturationValue(
                                     hue_shift_limit=20, sat_shift_limit=40,
                                     val_shift_limit=20, p=1.0)])),
        ("CLAHE Enhancement",     A.Compose([A.Resize(512,512), A.CLAHE(clip_limit=4.0, p=1.0)])),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Data Augmentation Techniques Applied During Training",
                 fontsize=15, fontweight="bold")
    axes = axes.flatten()

    for i, (name, transform) in enumerate(augmentations):
        augmented = transform(image=orig_image)["image"]
        axes[i].imshow(augmented)
        axes[i].set_title(name, fontsize=11, fontweight="bold" if i==0 else "normal")
        axes[i].axis("off")
        if i == 0:
            axes[i].set_title(f"{name}\n(before augmentation)", fontsize=11)

    plt.tight_layout()
    path = "results/preprocessing/2_augmentation_samples.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  GRAPH 3 — IMAGE AND MASK PAIR
#  Shows original image with its GT mask
# ─────────────────────────────────────────

def show_image_mask_pair():
    resized_img  = cv2.resize(orig_image, (512, 512))
    resized_mask = cv2.resize(orig_mask,  (512, 512), interpolation=cv2.INTER_NEAREST)

    # Convert mask to colour
    color_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    color_mask[resized_mask == 255] = [20,  20,  20]    # background
    color_mask[resized_mask == 128] = [0,  200,  80]    # optic disc  - green
    color_mask[resized_mask == 0]   = [255, 80,  80]    # optic cup   - red

    # Overlay
    overlay = cv2.addWeighted(resized_img, 0.6, color_mask, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Image — Ground Truth Mask — Overlay", fontsize=14, fontweight="bold")

    axes[0].imshow(resized_img)
    axes[0].set_title("Original Fundus Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title("Ground Truth Mask\n(Expert Annotated)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    patches = [
        mpatches.Patch(color=np.array([0,200,80])/255,  label="Optic Disc"),
        mpatches.Patch(color=np.array([255,80,80])/255, label="Optic Cup"),
        mpatches.Patch(color=np.array([20,20,20])/255,  label="Background"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    path = "results/preprocessing/3_image_mask_pair.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  GRAPH 4 — PIXEL DISTRIBUTION
#  Shows histogram before and after normalization
# ─────────────────────────────────────────

def show_pixel_distribution():
    resized    = cv2.resize(orig_image, (512, 512))
    mean       = np.array([0.485, 0.456, 0.406])
    std        = np.array([0.229, 0.224, 0.225])
    normalized = (resized / 255.0 - mean) / std

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Pixel Value Distribution Before and After Normalization",
                 fontsize=14, fontweight="bold")

    colors = ["#E74C3C", "#27AE60", "#2E86C1"]
    labels = ["Red Channel", "Green Channel", "Blue Channel"]

    # Before normalization
    for i, (col, lbl) in enumerate(zip(colors, labels)):
        axes[0].hist(resized[:,:,i].flatten(), bins=50,
                     alpha=0.6, color=col, label=lbl)
    axes[0].set_title("Before Normalization\n(pixel values 0–255)", fontsize=11)
    axes[0].set_xlabel("Pixel Value")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # After normalization
    for i, (col, lbl) in enumerate(zip(colors, labels)):
        axes[1].hist(normalized[:,:,i].flatten(), bins=50,
                     alpha=0.6, color=col, label=lbl)
    axes[1].set_title("After Normalization\n(scaled using ImageNet mean & std)", fontsize=11)
    axes[1].set_xlabel("Pixel Value")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = "results/preprocessing/4_pixel_distribution.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  RUN ALL
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\nGenerating preprocessing visualizations...\n")
    show_preprocessing_steps()
    show_augmentation_samples()
    show_image_mask_pair()
    show_pixel_distribution()
    print("\n✅ All saved in results/preprocessing/")
    print("   1_preprocessing_steps.png  — Original → Resize → Normalize")
    print("   2_augmentation_samples.png — 8 augmentation techniques")
    print("   3_image_mask_pair.png      — Image with ground truth mask")
    print("   4_pixel_distribution.png   — Pixel histogram before/after")