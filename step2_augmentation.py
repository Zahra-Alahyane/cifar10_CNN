"""
ÉTAPE 2 — Augmentation de données
===================================
Run: python step2_augmentation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from step1_data import load_and_preprocess, CLASS_NAMES


def build_datagen():
    """Crée et retourne le générateur d'augmentation"""
    datagen = ImageDataGenerator(
        rotation_range=15,        # Rotation aléatoire ±15°
        width_shift_range=0.1,    # Décalage horizontal ±10%
        height_shift_range=0.1,   # Décalage vertical ±10%
        horizontal_flip=True,     # Miroir gauche/droite
        zoom_range=0.1,           # Zoom ±10%
        fill_mode='nearest'       # Remplissage des bords créés
    )
    return datagen


def visualize_augmentation(datagen, X_train):
    """Visualiser 7 versions augmentées d'une même image"""
    print("\n🔄 Visualisation des augmentations...")

    fig, axes = plt.subplots(3, 8, figsize=(16, 7))
    fig.suptitle('Augmentation de données — 3 images originales × 7 variantes', fontsize=12)

    for row, img_idx in enumerate([0, 100, 500]):
        sample = X_train[img_idx:img_idx+1]

        # Image originale
        axes[row, 0].imshow(sample[0])
        axes[row, 0].set_title('Original', fontsize=8)
        axes[row, 0].axis('off')

        # 7 variantes augmentées
        gen = datagen.flow(sample, batch_size=1)
        for col in range(1, 8):
            batch = next(gen)
            axes[row, col].imshow(batch[0])
            axes[row, col].set_title(f'Aug {col}', fontsize=8)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/augmentation_examples.png', dpi=150)
    print("  → Sauvegardé : outputs/augmentation_examples.png")
    plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)

    print("📦 Chargement des données...")
    X_train, y_train, X_test, y_test, y_train_cat, y_test_cat = load_and_preprocess()

    print("⚙️  Construction du générateur d'augmentation...")
    datagen = build_datagen()
    datagen.fit(X_train)
    print("  ✅ Générateur prêt")
    print("  → Paramètres : rotation=15°, shift=10%, flip=True, zoom=10%")

    visualize_augmentation(datagen, X_train)

    print("\n✅ Étape 2 terminée !")
    print("   → Passez à : python step3_model.py")
