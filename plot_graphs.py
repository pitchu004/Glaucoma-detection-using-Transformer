"""
plot_graphs.py
--------------
Generates training graphs using REAL data from your 50 epoch training.

Graphs generated:
  1. Training vs Validation Loss
  2. Training vs Validation Accuracy (F1 Score)
  3. Training vs Validation IoU
  4. All metrics combined in one figure

Run:
    python plot_graphs.py
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ─────────────────────────────────────────
#  YOUR REAL TRAINING DATA (50 epochs)
# ─────────────────────────────────────────

epochs = list(range(1, 51))

train_loss = [
    0.3628, 0.1041, 0.0765, 0.0645, 0.0598, 0.0616, 0.0596, 0.0569, 0.0558, 0.0555,
    0.0547, 0.0521, 0.0523, 0.0535, 0.0520, 0.0517, 0.0505, 0.0526, 0.0488, 0.0487,
    0.0484, 0.0479, 0.0470, 0.0470, 0.0454, 0.0449, 0.0459, 0.0457, 0.0443, 0.0451,
    0.0432, 0.0454, 0.0433, 0.0447, 0.0436, 0.0427, 0.0438, 0.0432, 0.0426, 0.0429,
    0.0421, 0.0425, 0.0421, 0.0415, 0.0417, 0.0407, 0.0417, 0.0401, 0.0424, 0.0425
]

val_loss = [
    0.1565, 0.1079, 0.0763, 0.0751, 0.0638, 0.0816, 0.0706, 0.0746, 0.0648, 0.0724,
    0.0679, 0.0683, 0.0677, 0.0649, 0.0634, 0.0662, 0.0594, 0.0708, 0.0682, 0.0631,
    0.0768, 0.0752, 0.0703, 0.0742, 0.0662, 0.0684, 0.0631, 0.0699, 0.0662, 0.0630,
    0.0640, 0.0657, 0.0661, 0.0603, 0.0660, 0.0633, 0.0638, 0.0645, 0.0639, 0.0665,
    0.0664, 0.0668, 0.0675, 0.0652, 0.0653, 0.0667, 0.0666, 0.0652, 0.0662, 0.0664
]

val_f1 = [
    85.89, 85.57, 89.68, 89.79, 91.16, 88.52, 90.02, 89.43, 90.81, 89.63,
    90.39, 90.49, 90.32, 90.92, 90.92, 90.58, 91.49, 90.08, 90.26, 91.00,
    89.08, 89.44, 90.08, 89.51, 90.66, 90.31, 91.13, 90.08, 90.75, 91.09,
    90.98, 90.86, 90.73, 91.46, 90.75, 91.13, 91.01, 91.01, 91.04, 90.72,
    90.71, 90.64, 90.60, 90.84, 90.84, 90.67, 90.69, 90.87, 90.73, 90.71
]

val_acc = [
    99.55, 99.67, 99.67, 99.69, 99.75, 99.63, 99.66, 99.65, 99.73, 99.65,
    99.70, 99.72, 99.69, 99.73, 99.73, 99.70, 99.72, 99.70, 99.67, 99.69,
    99.65, 99.67, 99.68, 99.68, 99.70, 99.71, 99.72, 99.66, 99.72, 99.72,
    99.72, 99.72, 99.71, 99.73, 99.70, 99.73, 99.72, 99.73, 99.72, 99.71,
    99.71, 99.71, 99.71, 99.71, 99.71, 99.71, 99.71, 99.72, 99.71, 99.71
]

val_iou = [
    76.72, 76.78, 82.14, 82.32, 84.45, 80.42, 82.63, 81.75, 83.88, 82.05,
    83.21, 83.39, 83.10, 84.04, 84.06, 83.50, 84.92, 82.75, 83.00, 84.14,
    81.24, 81.79, 82.74, 81.89, 83.62, 83.10, 84.36, 82.72, 83.78, 84.29,
    84.14, 83.94, 83.74, 84.86, 83.76, 84.37, 84.18, 84.17, 84.21, 83.73,
    83.71, 83.60, 83.55, 83.90, 83.91, 83.65, 83.68, 83.96, 83.74, 83.70
]

# Best epoch marker
best_epoch = 17
best_f1    = 91.49


# ─────────────────────────────────────────
#  STYLE SETTINGS
# ─────────────────────────────────────────

plt.rcParams.update({
    'font.family'  : 'sans-serif',
    'font.size'    : 11,
    'axes.titlesize' : 13,
    'axes.titleweight' : 'bold',
    'axes.labelsize' : 11,
    'legend.fontsize': 10,
    'grid.alpha'   : 0.4,
    'grid.linestyle': '--',
})

COLORS = {
    'train_loss' : '#E74C3C',   # red
    'val_loss'   : '#2E86C1',   # blue
    'f1'         : '#27AE60',   # green
    'acc'        : '#8E44AD',   # purple
    'iou'        : '#E67E22',   # orange
    'best'       : '#F39C12',   # yellow/gold for best epoch line
}

os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────
#  GRAPH 1 — LOSS CURVE
# ─────────────────────────────────────────

def plot_loss():
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, train_loss, 'o-', label='Training Loss',
            color=COLORS['train_loss'], linewidth=2, markersize=3)
    ax.plot(epochs, val_loss, 's--', label='Validation Loss',
            color=COLORS['val_loss'], linewidth=2, markersize=3)

    ax.axvline(x=best_epoch, color=COLORS['best'],
               linestyle=':', linewidth=1.5, label=f'Best Epoch ({best_epoch})')

    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xlim(1, 50)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    path = 'results/loss_curve.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved → {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  GRAPH 2 — F1 SCORE CURVE
# ─────────────────────────────────────────

def plot_f1():
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, val_f1, 'o-', label='Validation F1 Score',
            color=COLORS['f1'], linewidth=2, markersize=3)

    # Mark best point
    ax.scatter([best_epoch], [best_f1], color=COLORS['best'],
               s=120, zorder=5, label=f'Best F1: {best_f1}% (Epoch {best_epoch})')
    ax.axvline(x=best_epoch, color=COLORS['best'],
               linestyle=':', linewidth=1.5)

    ax.set_title('Validation F1 Score over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score (%)')
    ax.set_xlim(1, 50)
    ax.set_ylim(84, 93)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    path = 'results/f1_curve.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved → {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  GRAPH 3 — ACCURACY CURVE
# ─────────────────────────────────────────

def plot_accuracy():
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, val_acc, 'o-', label='Validation Accuracy',
            color=COLORS['acc'], linewidth=2, markersize=3)

    ax.axvline(x=best_epoch, color=COLORS['best'],
               linestyle=':', linewidth=1.5, label=f'Best Epoch ({best_epoch})')

    ax.set_title('Validation Accuracy over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(1, 50)
    ax.set_ylim(99.5, 99.85)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    path = 'results/accuracy_curve.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved → {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  GRAPH 4 — IoU CURVE
# ─────────────────────────────────────────

def plot_iou():
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(epochs, val_iou, 'o-', label='Validation IoU',
            color=COLORS['iou'], linewidth=2, markersize=3)

    ax.axvline(x=best_epoch, color=COLORS['best'],
               linestyle=':', linewidth=1.5, label=f'Best Epoch ({best_epoch})')

    ax.set_title('Validation IoU over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU (%)')
    ax.set_xlim(1, 50)
    ax.set_ylim(75, 86)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    path = 'results/iou_curve.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved → {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  GRAPH 5 — COMBINED (for report)
# ─────────────────────────────────────────

def plot_combined():
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Loss ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, 'o-', label='Train Loss',
             color=COLORS['train_loss'], linewidth=2, markersize=2)
    ax1.plot(epochs, val_loss, 's--', label='Val Loss',
             color=COLORS['val_loss'], linewidth=2, markersize=2)
    ax1.axvline(x=best_epoch, color=COLORS['best'], linestyle=':', linewidth=1.5)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # ── F1 ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, val_f1, 'o-', label='Val F1 Score',
             color=COLORS['f1'], linewidth=2, markersize=2)
    ax2.scatter([best_epoch], [best_f1], color=COLORS['best'],
                s=100, zorder=5, label=f'Best: {best_f1}%')
    ax2.axvline(x=best_epoch, color=COLORS['best'], linestyle=':', linewidth=1.5)
    ax2.set_title('Validation F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score (%)')
    ax2.set_ylim(84, 93)
    ax2.legend()
    ax2.grid(True)

    # ── Accuracy ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, val_acc, 'o-', label='Val Accuracy',
             color=COLORS['acc'], linewidth=2, markersize=2)
    ax3.axvline(x=best_epoch, color=COLORS['best'], linestyle=':', linewidth=1.5)
    ax3.set_title('Validation Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_ylim(99.5, 99.85)
    ax3.legend()
    ax3.grid(True)

    # ── IoU ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(epochs, val_iou, 'o-', label='Val IoU',
             color=COLORS['iou'], linewidth=2, markersize=2)
    ax4.axvline(x=best_epoch, color=COLORS['best'], linestyle=':', linewidth=1.5)
    ax4.set_title('Validation IoU')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('IoU (%)')
    ax4.set_ylim(75, 86)
    ax4.legend()
    ax4.grid(True)

    fig.suptitle(
        'SegFormer Glaucoma Detection — Training Results (50 Epochs)',
        fontsize=15, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    path = 'results/combined_graphs.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved → {path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────
#  RUN ALL
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Generating training graphs...\n")
    plot_loss()
    plot_f1()
    plot_accuracy()
    plot_iou()
    plot_combined()
    print("\n✅ All graphs saved in results/ folder!")
    print("   results/loss_curve.png")
    print("   results/f1_curve.png")
    print("   results/accuracy_curve.png")
    print("   results/iou_curve.png")
    print("   results/combined_graphs.png")
