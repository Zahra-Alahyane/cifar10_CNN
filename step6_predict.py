"""
ÉTAPE 6 — Prédiction sur vos propres images (CLI)
===================================================
Run: python step6_predict.py --image path/to/image.jpg
  ou python step6_predict.py --image path/to/image.jpg --top 3

Le modèle accepte toute image (redimensionnée automatiquement en 32×32).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model

CLASS_NAMES = [
    'avion', 'automobile', 'oiseau', 'chat', 'cerf',
    'chien', 'grenouille', 'cheval', 'bateau', 'camion'
]

CLASS_EMOJIS = {
    'avion': '✈️', 'automobile': '🚗', 'oiseau': '🐦',
    'chat': '🐱', 'cerf': '🦌', 'chien': '🐶',
    'grenouille': '🐸', 'cheval': '🐴', 'bateau': '⛵',
    'camion': '🚛'
}


def load_and_prepare_image(image_path):
    """Charge une image, la redimensionne en 32×32 et la normalise"""
    img = Image.open(image_path).convert('RGB')
    img_original = img.copy()

    # Redimensionner en 32×32 (taille CIFAR-10)
    img_resized = img.resize((32, 32), Image.LANCZOS)

    # Convertir en array et normaliser
    img_array = np.array(img_resized).astype('float32') / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)

    return img_batch, img_original, img_resized


def predict(model, image_path, top_k=5):
    """Retourne les top_k prédictions pour une image"""
    img_batch, img_original, img_resized = load_and_prepare_image(image_path)

    # Prédiction
    probs = model.predict(img_batch, verbose=0)[0]

    # Top-k résultats
    top_indices = np.argsort(probs)[::-1][:top_k]
    results = [(CLASS_NAMES[i], probs[i]) for i in top_indices]

    return results, img_original, img_resized, probs


def visualize_prediction(image_path, results, img_original, img_resized, probs):
    """Visualisation complète de la prédiction"""
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('#0a0c10')

    # ── Image originale ──────────────────────────
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img_original)
    ax1.set_title('Image originale', color='white', fontsize=11)
    ax1.axis('off')
    ax1.set_facecolor('#0a0c10')

    # ── Image 32×32 ──────────────────────────────
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img_resized)
    ax2.set_title('Redimensionnée 32×32\n(entrée du modèle)', color='white', fontsize=11)
    ax2.axis('off')
    ax2.set_facecolor('#0a0c10')

    # ── Bar chart des probabilités ────────────────
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor('#111318')

    colors = ['#00e5ff' if i == 0 else '#1e2330' for i in range(len(CLASS_NAMES))]
    sorted_idx = np.argsort(probs)
    bar_colors = ['#00e5ff' if CLASS_NAMES[i] == results[0][0] else '#334155'
                  for i in sorted_idx]

    bars = ax3.barh(
        [CLASS_NAMES[i] for i in sorted_idx],
        [probs[i] * 100 for i in sorted_idx],
        color=bar_colors, edgecolor='none', height=0.7
    )

    ax3.set_xlabel('Probabilité (%)', color='#94a3b8')
    ax3.set_title('Probabilités par classe', color='white', fontsize=11)
    ax3.tick_params(colors='#94a3b8')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_color('#1e2330')
    ax3.spines['left'].set_color('#1e2330')
    ax3.set_xlim([0, 105])

    for bar, idx in zip(bars, sorted_idx):
        prob = probs[idx] * 100
        if prob > 2:
            ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f'{prob:.1f}%', va='center', color='#94a3b8', fontsize=8)

    # ── Titre principal ───────────────────────────
    pred_class = results[0][0]
    pred_prob  = results[0][1]
    emoji = CLASS_EMOJIS.get(pred_class, '🔍')
    fig.suptitle(
        f'{emoji}  Prédiction : {pred_class.upper()}  ({pred_prob*100:.1f}%)',
        color='white', fontsize=14, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    out_path = 'outputs/prediction_result.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0c10')
    print(f"  → Visualisation sauvegardée : {out_path}")
    plt.show()
    plt.close()


def print_results(results, image_path):
    """Affichage console des résultats"""
    print(f"\n{'═'*45}")
    print(f"  🔍 Image analysée : {os.path.basename(image_path)}")
    print(f"{'═'*45}")
    print(f"\n  {'Rang':<6} {'Classe':<15} {'Confiance':>10}  {'Barre'}")
    print(f"  {'-'*50}")

    for rank, (cls, prob) in enumerate(results, 1):
        bar_len = int(prob * 30)
        bar = '█' * bar_len + '░' * (30 - bar_len)
        emoji = CLASS_EMOJIS.get(cls, '  ')
        marker = '← RÉSULTAT' if rank == 1 else ''
        print(f"  #{rank:<5} {emoji} {cls:<12} {prob*100:>8.2f}%  {bar}  {marker}")

    print(f"\n{'═'*45}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prédiction CIFAR-10 sur une image')
    parser.add_argument('--image', required=True, help='Chemin vers l\'image')
    parser.add_argument('--top',   type=int, default=5, help='Nombre de top prédictions (défaut: 5)')
    parser.add_argument('--model', default='outputs/best_model.keras',
                        help='Chemin vers le modèle sauvegardé')
    args = parser.parse_args()

    # ── Vérifications ────────────────────────────
    if not os.path.exists(args.image):
        print(f"❌ Image introuvable : {args.image}")
        sys.exit(1)

    model_path = args.model
    if not os.path.exists(model_path):
        model_path = 'outputs/final_model.keras'
    if not os.path.exists(model_path):
        print("❌ Aucun modèle trouvé. Entraînez d'abord : python step4_train.py")
        sys.exit(1)

    os.makedirs('outputs', exist_ok=True)

    # ── Chargement ───────────────────────────────
    print(f"📂 Chargement du modèle : {model_path}")
    model = load_model(model_path)
    print(f"🖼️  Analyse de l'image : {args.image}")

    # ── Prédiction ───────────────────────────────
    results, img_original, img_resized, probs = predict(model, args.image, top_k=args.top)

    # ── Affichage ────────────────────────────────
    print_results(results, args.image)
    visualize_prediction(args.image, results, img_original, img_resized, probs)
