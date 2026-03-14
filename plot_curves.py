"""
plot_curves.py — Graphe Accuracy + Loss (valeurs lues depuis ton graphe)
Lance : python plot_curves.py
"""

import os
import matplotlib.pyplot as plt

# Valeurs lues visuellement depuis ton graphe (30 époques)

train_acc = [
    0.37, 0.52, 0.60, 0.64, 0.66,
    0.67, 0.68, 0.69, 0.70, 0.71,
    0.72, 0.73, 0.73, 0.74, 0.75,
    0.75, 0.76, 0.76, 0.77, 0.77,
    0.77, 0.77, 0.77, 0.77, 0.78,
    0.78, 0.78, 0.78, 0.78, 0.78,
]

val_acc = [
    0.52, 0.60, 0.63, 0.65, 0.70,
    0.68, 0.66, 0.71, 0.72, 0.70,
    0.73, 0.72, 0.74, 0.75, 0.71,
    0.75, 0.74, 0.75, 0.76, 0.76,
    0.79, 0.76, 0.79, 0.77, 0.79,
    0.78, 0.80, 0.80, 0.78, 0.82,
]

train_loss = [
    1.82, 1.55, 1.35, 1.18, 1.08,
    1.00, 0.95, 0.91, 0.88, 0.85,
    0.83, 0.82, 0.80, 0.79, 0.78,
    0.77, 0.76, 0.75, 0.74, 0.73,
    0.72, 0.72, 0.71, 0.70, 0.70,
    0.69, 0.68, 0.68, 0.67, 0.63,
]

val_loss = [
    1.35, 1.10, 0.98, 0.92, 0.83,
    0.90, 0.95, 0.88, 0.84, 0.90,
    0.82, 0.83, 0.69, 0.77, 0.83,
    0.75, 0.78, 0.74, 0.76, 0.73,
    0.66, 0.72, 0.67, 0.70, 0.67,
    0.65, 0.63, 0.60, 0.64, 0.52,
]

# ── Graphe ──────────────────────────────
epochs_range = range(len(train_acc))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_range, train_acc, color="#4C72B0", linewidth=2, label="train_acc")
ax1.plot(epochs_range, val_acc,   color="#DD8452", linewidth=2, label="val_acc")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoques")
ax1.set_ylabel("Accuracy")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, train_loss, color="#4C72B0", linewidth=2, label="train_loss")
ax2.plot(epochs_range, val_loss,   color="#DD8452", linewidth=2, label="val_loss")
ax2.set_title("Loss")
ax2.set_xlabel("Epoques")
ax2.set_ylabel("Loss")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graphe sauvegarde -> outputs/training_curves.png")
