"""
model/segformer_model.py
------------------------
SegFormer-based semantic segmentation model for glaucoma detection.

Uses the Hugging Face `transformers` library which provides:
  - SegFormerForSemanticSegmentation (encoder + MLP decoder head)
  - Pre-trained weights from ImageNet / REFUGE2

Reference architecture from the journal:
  "Transformer Based Semantic Segmentation for Glaucoma Diagnosis
   using Morphological Biomarker Analysis"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerConfig,
)


# ─────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────

class GlaucomaSegFormer(nn.Module):
    """
    SegFormer-B2 backbone with a 3-class segmentation head.

    Classes:
        0 → Background
        1 → Optic Disc (OD)
        2 → Optic Cup  (OC)

    The model outputs logits at 1/4 resolution which are
    bilinearly upsampled to the input size during forward pass.
    """

    NUM_CLASSES = 3

    def __init__(self, pretrained_name="nvidia/mit-b2",
                 num_labels=3, id2label=None, label2id=None):
        super().__init__()

        if id2label is None:
            id2label = {0: "background", 1: "optic_disc", 2: "optic_cup"}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,   # head re-initialised for 3 classes
        )

    def forward(self, pixel_values, labels=None):
        """
        Args:
            pixel_values : (B, 3, H, W) normalised tensor
            labels       : (B, H, W) long tensor (optional, for training)

        Returns:
            During training  : loss (scalar)
            During inference : logits (B, NUM_CLASSES, H, W)
        """
        outputs = self.segformer(
            pixel_values=pixel_values,
            labels=labels,
        )

        if labels is not None:
            return outputs.loss

        # Upsample from (B, C, H/4, W/4) → (B, C, H, W)
        logits = F.interpolate(
            outputs.logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits


# ─────────────────────────────────────────
#  LOSS: Dice + Cross-Entropy combo
# ─────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, num_classes=3):
        super().__init__()
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, logits, targets):
        """
        logits  : (B, C, H, W) raw scores
        targets : (B, H, W) class indices
        """
        probs = F.softmax(logits, dim=1)                       # (B, C, H, W)
        one_hot = F.one_hot(targets, self.num_classes)         # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()         # (B, C, H, W)

        intersection = (probs * one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Weighted Cross-Entropy + Dice loss."""

    def __init__(self, ce_weight=0.5, dice_weight=0.5,
                 class_weights=None, num_classes=3):
        super().__init__()
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.dice_loss   = DiceLoss(num_classes=num_classes)
        self.class_weights = class_weights   # tensor of shape (num_classes,)

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.class_weights)
        dl = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dl


# ─────────────────────────────────────────
#  QUICK MODEL TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GlaucomaSegFormer().to(device)
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    dummy_labels = torch.randint(0, 3, (2, 512, 512)).to(device)

    # Training mode
    loss = model(dummy_input, labels=dummy_labels)
    print(f"Training loss (dummy): {loss.item():.4f}")

    # Inference mode
    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
    print(f"Logits shape: {logits.shape}")   # Should be (2, 3, 512, 512)
