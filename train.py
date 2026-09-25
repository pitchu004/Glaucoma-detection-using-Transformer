"""
train.py
--------
Training loop for the SegFormer-based glaucoma detection model.

Usage:
    python train.py --data_dir ./data/REFUGE2 --epochs 50 --batch_size 8
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from data.prepare_dataset import get_dataloaders
from model.segformer_model import GlaucomaSegFormer, CombinedLoss
from utils import SegmentationMetrics, save_checkpoint, plot_training_curves


# ─────────────────────────────────────────
#  ARGUMENT PARSING
# ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Glaucoma SegFormer")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--lr",          type=float, default=6e-5)
    parser.add_argument("--img_size",    type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path to resume checkpoint")
    return parser.parse_args()


# ─────────────────────────────────────────
#  TRAINING STEP
# ─────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device, epoch):
    model.train()
    metrics   = SegmentationMetrics(num_classes=3)
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)             # (B, 3, H, W)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)
        total_loss += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    m = metrics.compute()
    return avg_loss, m


# ─────────────────────────────────────────
#  VALIDATION STEP
# ─────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    metrics    = SegmentationMetrics(num_classes=3)
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        with autocast():
            logits = model(images)
            loss   = criterion(logits, masks)

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    m = metrics.compute()
    return avg_loss, m


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*50}")
    print(f"  Glaucoma Detection - SegFormer Training")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch Size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print(f"{'='*50}\n")

    # ── Data ──────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────
    model = GlaucomaSegFormer(
        pretrained_name="nvidia/mit-b2",
        num_labels=3,
    ).to(device)

    # Class weights: OC and OD are small regions → upweight them
    class_weights = torch.tensor([0.5, 1.5, 2.0]).to(device)
    criterion = CombinedLoss(
        ce_weight=0.5, dice_weight=0.5,
        class_weights=class_weights,
        num_classes=3,
    )

    # ── Optimizer & Scheduler ─────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler()

    # ── Resume ────────────────────────────
    start_epoch = 1
    best_f1     = 0.0
    if args.resume:
        from utils import load_checkpoint
        start_epoch, best_f1 = load_checkpoint(
            model, optimizer, args.resume, device)
        start_epoch += 1

    # ── Training Loop ─────────────────────
    train_losses, val_losses, val_f1s = [], [], []

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_m = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch)
        vl_loss, vl_m = validate(
            model, val_loader, criterion, device, epoch)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_f1s.append(vl_m["f1_score"])

        print(f"\nEpoch [{epoch:3d}/{args.epochs}] "
              f"Train Loss: {tr_loss:.4f}  |  "
              f"Val Loss: {vl_loss:.4f}  |  "
              f"Val Acc: {vl_m['accuracy']*100:.2f}%  |  "
              f"Val F1: {vl_m['f1_score']*100:.2f}%  |  "
              f"Val IoU: {vl_m['mean_iou']*100:.2f}%")

        # Save best model
        if vl_m["f1_score"] > best_f1:
            best_f1 = vl_m["f1_score"]
            save_checkpoint(
                model, optimizer, epoch, best_f1,
                path=f"{args.checkpoint_dir}/best_model.pth",
            )

        # Periodic checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, vl_m["f1_score"],
                path=f"{args.checkpoint_dir}/epoch_{epoch}.pth",
            )

    print(f"\n✅ Training complete. Best Val F1: {best_f1*100:.2f}%")

    # ── Plot curves ───────────────────────
    plot_training_curves(
        train_losses, val_losses, val_f1s,
        save_path=f"{args.checkpoint_dir}/training_curves.png",
    )


if __name__ == "__main__":
    main()
