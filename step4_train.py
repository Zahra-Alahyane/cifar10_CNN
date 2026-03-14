"""
ÉTAPE 4 — Compilation & Entraînement
======================================
Run: python step4_train.py

⏱️  Temps estimé :
  - CPU : ~5-10 min/époque  (50-80 époques → 4-8h)
  - GPU : ~30-60 sec/époque (50-80 époques → 30-60 min)
  
💡 Recommandé : Google Colab (GPU T4 gratuit)
               ou Kaggle Notebooks (GPU P100 gratuit)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('outputs', exist_ok=True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
)

from step1_data import load_and_preprocess, CLASS_NAMES
from step2_augmentation import build_datagen
from step3_model import build_model

# ══════════════════════════════════════════════
# HYPERPARAMÈTRES
# ══════════════════════════════════════════════
BATCH_SIZE = 64
EPOCHS     = 100      # EarlyStopping s'arrêtera bien avant
LR         = 0.001


def compile_model(model):
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"✅ Modèle compilé — Adam(lr={LR}), loss=categorical_crossentropy")
    return model


def get_callbacks():
    return [
        # Réduit LR × 0.5 si val_loss stagne 5 époques
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        ),
        # Arrêt si val_loss ne s'améliore pas pendant 15 époques
        EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        ),
        # Sauvegarde du meilleur checkpoint
        ModelCheckpoint(
            filepath='outputs/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True, verbose=1
        )
    ]


def plot_history(history):
    """Tracer les 4 courbes : loss, val_loss, accuracy, val_accuracy"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Courbes d\'entraînement — CIFAR-10 CNN', fontsize=14)

    epochs = range(1, len(history.history['loss']) + 1)

    # ── Loss ──────────────────────────────────────
    axes[0].plot(epochs, history.history['loss'],     label='Train Loss',
                 color='royalblue', linewidth=2)
    axes[0].plot(epochs, history.history['val_loss'], label='Val Loss',
                 color='tomato', linewidth=2, linestyle='--')
    axes[0].set_title('Loss (Perte)')
    axes[0].set_xlabel('Époques')
    axes[0].set_ylabel('Categorical Crossentropy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────
    axes[1].plot(epochs, history.history['accuracy'],     label='Train Acc',
                 color='royalblue', linewidth=2)
    axes[1].plot(epochs, history.history['val_accuracy'], label='Val Acc',
                 color='tomato', linewidth=2, linestyle='--')
    axes[1].set_title('Accuracy (Précision)')
    axes[1].set_xlabel('Époques')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/training_curves.png', dpi=150)
    print("  → Sauvegardé : outputs/training_curves.png")
    plt.close()


if __name__ == '__main__':
    # ── Chargement données ───────────────────────
    print("📦 Chargement des données...")
    X_train, y_train, X_test, y_test, y_train_cat, y_test_cat = load_and_preprocess()

    # ── Augmentation ────────────────────────────
    print("🔄 Préparation de l'augmentation...")
    datagen = build_datagen()
    datagen.fit(X_train)

    # ── Modèle ──────────────────────────────────
    print("🏗️  Construction du modèle...")
    model = build_model()
    model = compile_model(model)
    model.summary()

    # ── Entraînement ─────────────────────────────
    print(f"\n🚀 Début de l'entraînement — {EPOCHS} époques max, batch={BATCH_SIZE}")
    print("   (EarlyStopping arrêtera automatiquement)\n")

    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test_cat),
        callbacks=get_callbacks(),
        verbose=1
    )

    # ── Sauvegarde finale ────────────────────────
    model.save('outputs/final_model.keras')
    print("\n💾 Modèle final sauvegardé : outputs/final_model.keras")

    # ── Courbes ──────────────────────────────────
    print("📊 Génération des courbes d'entraînement...")
    plot_history(history)

    # ── Score rapide ─────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n🎯 Test Accuracy : {test_acc*100:.2f}%")
    print(f"🎯 Test Loss     : {test_loss:.4f}")

    print("\n✅ Étape 4 terminée !")
    print("   → Passez à : python step5_evaluate.py")
