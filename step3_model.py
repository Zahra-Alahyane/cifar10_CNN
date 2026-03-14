"""
ÉTAPE 3 — Architecture du CNN Profond
=======================================
Run: python step3_model.py
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, Flatten, Dense
)


def build_model(input_shape=(32, 32, 3), num_classes=10):
    """
    CNN Profond avec 3 blocs convolutifs + classifieur dense.
    Chaque bloc : Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout
    """
    model = Sequential([

        # ════════════════════════════════════════
        # BLOC 1 — 32 filtres  |  sortie: 16×16×32
        # ════════════════════════════════════════
        Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # ════════════════════════════════════════
        # BLOC 2 — 64 filtres  |  sortie: 8×8×64
        # ════════════════════════════════════════
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.35),

        # ════════════════════════════════════════
        # BLOC 3 — 128 filtres  |  sortie: 4×4×128
        # ════════════════════════════════════════
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),

        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.45),

        # ════════════════════════════════════════
        # CLASSIFIEUR DENSE
        # ════════════════════════════════════════
        Flatten(),
        Dense(256),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ], name='CIFAR10_CNN')

    return model


if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)

    print("🏗️  Construction du modèle CNN...")
    model = build_model()

    # ── Afficher l'architecture ──────────────────
    model.summary()

    # ── Compter les paramètres ───────────────────
    total     = model.count_params()
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\n📊 Paramètres totaux      : {total:,}")
    print(f"📊 Paramètres entraînables : {trainable:,}")

    # ── Sauvegarder le diagramme ─────────────────
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(
            model,
            to_file='outputs/model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=96
        )
        print("\n🖼️  Diagramme sauvegardé : outputs/model_architecture.png")
    except Exception as e:
        print(f"\n⚠️  plot_model non disponible ({e})")
        print("   Installez : pip install pydot graphviz")

    print("\n✅ Étape 3 terminée !")
    print("   → Passez à : python step4_train.py")
