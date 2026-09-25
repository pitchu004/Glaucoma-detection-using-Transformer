"""
data/prepare_dataset.py
-----------------------
Dataset loading for REFUGE2 with pre-split train/val/test folders.

Classes:
  0 - Background
  1 - Optic Disc (OD)
  2 - Optic Cup  (OC)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


# ─────────────────────────────────────────
#  AUGMENTATION PIPELINES
# ─────────────────────────────────────────

def get_train_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=20,
                             val_shift_limit=10, p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────
#  MASK CONVERSION  (REFUGE2 encoding)
# ─────────────────────────────────────────

def convert_mask(mask_pil):
    """
    REFUGE2 mask pixel encoding:
      128 (gray)  → Optic Disc  (class 1)
        0 (black) → Optic Cup   (class 2)
      255 (white) → Background  (class 0)
    """
    mask_np = np.array(mask_pil.convert("L"))

    class_mask = np.zeros_like(mask_np, dtype=np.uint8)  # default = background
    class_mask[mask_np == 128] = 1   # Optic Disc
    class_mask[mask_np == 0]   = 2   # Optic Cup
    return class_mask


# ─────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────

class GlaucomaDataset(Dataset):
    """
    Loads images and masks from one split folder (train / val / test).

    
    Args:
        data_dir : root data directory (contains train/, val/, test/)
        split    : one of "train", "val", "test"
        img_size : resize target
    """

    def __init__(self, data_dir, split="train", img_size=512):
        self.img_size   = img_size
        self.transforms = (get_train_transforms(img_size)
                           if split == "train"
                           else get_val_transforms(img_size))

        img_dir  = os.path.join(data_dir, split, "images")
        mask_dir = os.path.join(data_dir, split, "mask")

        # Collect and sort so images[i] matches masks[i]
        img_files  = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])
        mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith((".png", ".bmp"))
        ])

        self.img_paths  = [os.path.join(img_dir,  f) for f in img_files]
        self.mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

        assert len(self.img_paths) == len(self.mask_paths), (
            f"[{split}] Mismatch: {len(self.img_paths)} images "
            f"vs {len(self.mask_paths)} masks.\n"
            f"  images folder : {img_dir}\n"
            f"  mask folder   : {mask_dir}"
        )

        print(f"[{split:5s}] {len(self.img_paths)} image-mask pairs loaded.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask  = convert_mask(Image.open(self.mask_paths[idx]))

        augmented = self.transforms(image=image, mask=mask)
        return augmented["image"], augmented["mask"].long()


# ─────────────────────────────────────────
#  DATALOADER FACTORY
# ─────────────────────────────────────────

def get_dataloaders(data_dir, img_size=512, batch_size=8, num_workers=4):
    """
    Returns train, val, and test DataLoaders using the
    pre-existing split folders in data_dir.
    """
    train_dataset = GlaucomaDataset(data_dir, split="train", img_size=img_size)
    val_dataset   = GlaucomaDataset(data_dir, split="val",   img_size=img_size)
    test_dataset  = GlaucomaDataset(data_dir, split="test",  img_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, test_loader