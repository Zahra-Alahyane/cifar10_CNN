"""
ÉTAPE 5 — Évaluation complète du modèle
=========================================
Run: python step5_evaluate.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from step1_data import load_and_preprocess, CLASS_NAMES

os.makedirs('outputs', exist_ok=True)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── Matrice brute ────────────────────────────
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title('Matrice de Confusion (valeurs brutes)', fontsize=12)
    axes[0].set_ylabel('Vraie classe')
    axes[0].set_xlabel('Classe prédite')
    axes[0].tick_params(axis='x', rotation=45)

    # ── Matrice normalisée ───────────────────────
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title('Matrice de Confusion (normalisée par classe)', fontsize=12)
    axes[1].set_ylabel('Vraie classe')
    axes[1].set_xlabel('Classe prédite')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("  → Sauvegardé : outputs/confusion_matrix.png")
    plt.close()


def plot_per_class_accuracy(y_true, y_pred):
    """Bar chart de l'accuracy par classe"""
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    colors = ['#22c55e' if a >= 0.85 else '#f59e0b' if a >= 0.75 else '#ef4444'
              for a in per_class_acc]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(CLASS_NAMES, per_class_acc * 100, color=colors, edgecolor='white')
    plt.axhline(y=np.mean(per_class_acc) * 100, color='royalblue',
                linestyle='--', label=f'Moyenne : {np.mean(per_class_acc)*100:.1f}%')
    plt.title('Accuracy par classe')
    plt.ylabel('Accuracy (%)')
    plt.ylim([0, 105])
    plt.legend()
    plt.xticks(rotation=45)

    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('outputs/per_class_accuracy.png', dpi=150)
    print("  → Sauvegardé : outputs/per_class_accuracy.png")
    plt.close()


def show_wrong_predictions(X_test, y_true, y_pred, n=20):
    """Afficher des exemples mal classifiés"""
    wrong_idx = np.where(y_true != y_pred)[0][:n]

    fig, axes = plt.subplots(4, 5, figsize=(14, 12))
    fig.suptitle('Exemples mal classifiés', fontsize=14)

    for i, idx in enumerate(wrong_idx):
        ax = axes[i // 5, i % 5]
        ax.imshow(X_test[idx])
        ax.set_title(
            f'Réel: {CLASS_NAMES[y_true[idx]]}\nPrédit: {CLASS_NAMES[y_pred[idx]]}',
            fontsize=8, color='red'
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('outputs/wrong_predictions.png', dpi=150)
    print("  → Sauvegardé : outputs/wrong_predictions.png")
    plt.close()


if __name__ == '__main__':
    # ── Charger les données ──────────────────────
    print("📦 Chargement des données...")
    X_train, y_train, X_test, y_test, y_train_cat, y_test_cat = load_and_preprocess()

    # ── Charger le modèle entraîné ───────────────
    model_path = 'outputs/best_model.keras'
    if not os.path.exists(model_path):
        model_path = 'outputs/final_model.keras'
    print(f"📂 Chargement du modèle : {model_path}")
    model = load_model(model_path)

    # ── Score global ─────────────────────────────
    print("\n🧪 Évaluation sur le test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=1)
    print(f"\n{'='*40}")
    print(f"  ✅ Test Accuracy : {test_acc*100:.2f}%")
    print(f"  ✅ Test Loss     : {test_loss:.4f}")
    print(f"{'='*40}\n")

    # ── Prédictions ──────────────────────────────
    print("🔮 Génération des prédictions...")
    y_pred_proba = model.predict(X_test, verbose=1)
    y_pred  = np.argmax(y_pred_proba, axis=1)
    y_true  = y_test.flatten()

    # ── Rapport de classification ─────────────────
    print("\n📋 Rapport de classification :")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ── Visualisations ────────────────────────────
    print("📊 Génération des visualisations...")
    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_accuracy(y_true, y_pred)
    show_wrong_predictions(X_test, y_true, y_pred)

    print("\n✅ Étape 5 terminée !")
    print("   → Passez à : python step6_predict.py")
    print("   (interface de prédiction sur vos propres images)")
