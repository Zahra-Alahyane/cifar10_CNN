"""
ÉTAPE 1 — Chargement & Préparation des données CIFAR-10
========================================================
Run: python step1_data.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# ── Classes CIFAR-10 ─────────────────────────────────────
CLASS_NAMES = [
    'avion', 'automobile', 'oiseau', 'chat', 'cerf',
    'chien', 'grenouille', 'cheval', 'bateau', 'camion'
]

def load_and_preprocess():
    print("📦 Chargement du dataset CIFAR-10...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print(f"  Train brut : {X_train.shape}  |  dtype: {X_train.dtype}")
    print(f"  Test brut  : {X_test.shape}   |  dtype: {X_test.dtype}")

    # ── Normalisation [0,255] → [0.0, 1.0] ──────────────
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0

    # ── One-hot encoding ─────────────────────────────────
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test,  10)

    print(f"\n✅ Après normalisation :")
    print(f"  X_train : {X_train.shape}  min={X_train.min():.2f}  max={X_train.max():.2f}")
    print(f"  X_test  : {X_test.shape}   min={X_test.min():.2f}   max={X_test.max():.2f}")
    print(f"  y_train_cat shape : {y_train_cat.shape}")

    return X_train, y_train, X_test, y_test, y_train_cat, y_test_cat


def visualize_samples(X_train, y_train):
    """Afficher une grille d'exemples par classe"""
    print("\n🖼️  Génération de la grille d'exemples...")
    fig, axes = plt.subplots(10, 8, figsize=(14, 18))
    fig.suptitle('Exemples CIFAR-10 — 8 images par classe', fontsize=14, y=1.01)

    for class_idx in range(10):
        indices = np.where(y_train.flatten() == class_idx)[0][:8]
        for j, idx in enumerate(indices):
            axes[class_idx, j].imshow(X_train[idx])
            axes[class_idx, j].axis('off')
            if j == 0:
                axes[class_idx, j].set_ylabel(
                    CLASS_NAMES[class_idx], fontsize=9,
                    rotation=0, labelpad=55, va='center'
                )

    plt.tight_layout()
    plt.savefig('outputs/cifar10_samples.png', dpi=150, bbox_inches='tight')
    print("  → Sauvegardé : outputs/cifar10_samples.png")
    plt.close()


def visualize_distribution(y_train, y_test):
    """Distribution des classes"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, y, title in zip(axes, [y_train, y_test], ['Train (50 000)', 'Test (10 000)']):
        unique, counts = np.unique(y, return_counts=True)
        bars = ax.bar(CLASS_NAMES, counts, color='steelblue', edgecolor='white', linewidth=0.5)
        ax.set_title(f'Distribution des classes — {title}')
        ax.set_ylabel("Nombre d'images")
        ax.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    str(count), ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('outputs/class_distribution.png', dpi=150)
    print("  → Sauvegardé : outputs/class_distribution.png")
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)

    X_train, y_train, X_test, y_test, y_train_cat, y_test_cat = load_and_preprocess()
    visualize_samples(X_train, y_train)
    visualize_distribution(y_train, y_test)

    print("\n✅ Étape 1 terminée !")
    print("   → Passez à : python step2_augmentation.py")
