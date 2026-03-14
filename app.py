"""
app.py — Serveur Flask : connecte l'interface web au modèle Keras
=================================================================
Run: python app.py
Puis ouvrir : http://localhost:5000

Installation Flask : pip install flask flask-cors pillow
"""

import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

app = Flask(__name__, static_folder='.')
CORS(app)

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

# ── Chargement du modèle (une seule fois au démarrage) ───
model = None

def load_model_once():
    global model
    if model is None:
        from tensorflow.keras.models import load_model as keras_load
        model_path = 'outputs/best_model.keras'
        if not os.path.exists(model_path):
            model_path = 'outputs/final_model.keras'
        if os.path.exists(model_path):
            print(f"📂 Chargement du modèle : {model_path}")
            model = keras_load(model_path)
            print("✅ Modèle chargé !")
        else:
            print("⚠️  Aucun modèle trouvé. Entraînez d'abord : python step4_train.py")
    return model


@app.route('/')
def index():
    return send_from_directory('.', 'app_ui.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: { "image": "<base64 string>" }
    Returns: { "predictions": [...], "top_class": "...", "top_prob": 0.xx }
    """
    try:
        m = load_model_once()
        if m is None:
            return jsonify({'error': 'Modèle non disponible. Entraînez le modèle d\'abord.'}), 503

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Champ "image" manquant'}), 400

        # Décoder l'image base64
        img_data = data['image']
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Prétraitement
        img_resized = img.resize((32, 32), Image.LANCZOS)
        img_array  = np.array(img_resized).astype('float32') / 255.0
        img_batch  = np.expand_dims(img_array, axis=0)

        # Prédiction
        probs = m.predict(img_batch, verbose=0)[0]

        # Formater les résultats
        predictions = [
            {
                'class':      CLASS_NAMES[i],
                'emoji':      CLASS_EMOJIS[CLASS_NAMES[i]],
                'probability': float(probs[i]),
                'rank':        int(np.argsort(probs)[::-1].tolist().index(i)) + 1
            }
            for i in range(10)
        ]

        top_idx = int(np.argmax(probs))

        return jsonify({
            'predictions': predictions,
            'top_class':   CLASS_NAMES[top_idx],
            'top_emoji':   CLASS_EMOJIS[CLASS_NAMES[top_idx]],
            'top_prob':    float(probs[top_idx]),
            'all_probs':   probs.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Informations sur le modèle chargé"""
    m = load_model_once()
    if m is None:
        return jsonify({'status': 'not_loaded'})

    return jsonify({
        'status':      'loaded',
        'total_params': m.count_params(),
        'input_shape':  list(m.input_shape),
        'output_shape': list(m.output_shape),
        'classes':      CLASS_NAMES
    })


if __name__ == '__main__':
    print("🚀 Démarrage du serveur Flask...")
    print("   Interface : http://localhost:5000")
    print("   API       : http://localhost:5000/predict  (POST)")
    print("   Info      : http://localhost:5000/model-info (GET)")
    app.run(debug=True, host='0.0.0.0', port=5000)
