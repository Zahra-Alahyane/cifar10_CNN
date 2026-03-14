# 🧠 Projet #17 — Reconnaissance d'objets CIFAR-10 (CNN Profond)

## 📁 Structure du projet

```
cifar10_cnn/
├── step1_data.py          # Étape 1 : Chargement & exploration des données
├── step2_augmentation.py  # Étape 2 : Augmentation de données
├── step3_model.py         # Étape 3 : Architecture CNN
├── step4_train.py         # Étape 4 : Entraînement complet
├── step5_evaluate.py      # Étape 5 : Évaluation & visualisations
├── step6_predict.py       # Étape 6 : Prédiction CLI sur une image
├── app.py                 # Serveur Flask (API + interface web)
├── app_ui.html            # Interface web de démonstration
├── requirements.txt
└── outputs/               # (créé automatiquement)
    ├── best_model.keras
    ├── final_model.keras
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── cifar10_samples.png
    └── ...
```

---

## 🚀 Guide d'exécution étape par étape

### Prérequis
```bash
pip install -r requirements.txt
```

---

### ÉTAPE 1 — Chargement & visualisation des données
```bash
python step1_data.py
```
**Sorties :**
- `outputs/cifar10_samples.png` — Grille d'exemples par classe
- `outputs/class_distribution.png` — Distribution équilibrée des classes

---

### ÉTAPE 2 — Augmentation de données
```bash
python step2_augmentation.py
```
**Sorties :**
- `outputs/augmentation_examples.png` — Variantes augmentées d'images

---

### ÉTAPE 3 — Architecture du modèle
```bash
python step3_model.py
```
**Sorties :**
- Résumé du modèle dans la console (`model.summary()`)
- `outputs/model_architecture.png` — Diagramme de l'architecture
- Nombre de paramètres : ~1.2 million

---

### ÉTAPE 4 — Entraînement ⏱️
```bash
python step4_train.py
```
> ⚠️ **Durée :** ~5-10 min/époque sur CPU, ~30-60 sec/époque sur GPU  
> 💡 **Recommandé :** Google Colab (GPU T4 gratuit) ou Kaggle Notebooks

**Sorties :**
- `outputs/best_model.keras` — Meilleur modèle (checkpoint)
- `outputs/final_model.keras` — Modèle final
- `outputs/training_curves.png` — Courbes loss/accuracy

---

### ÉTAPE 5 — Évaluation complète
```bash
python step5_evaluate.py
```
**Sorties :**
- Score Test Accuracy (~82-88%)
- Rapport de classification par classe
- `outputs/confusion_matrix.png` — Matrice de confusion (brute + normalisée)
- `outputs/per_class_accuracy.png` — Accuracy par classe
- `outputs/wrong_predictions.png` — Exemples mal classifiés

---

### ÉTAPE 6 — Prédiction sur une image (CLI) 🔍
```bash
# Prédire sur une image
python step6_predict.py --image mon_image.jpg

# Afficher les top 3 classes
python step6_predict.py --image mon_image.jpg --top 3
```
**Sorties :**
- Résultat console avec barre de confiance
- `outputs/prediction_result.png` — Visualisation complète

---

### ÉTAPE 7 — Interface Web 🌐
```bash
# Installer Flask
pip install flask flask-cors

# Lancer le serveur
python app.py

# Ouvrir dans le navigateur
http://localhost:5000
```
L'interface permet d'uploader une image et voir la prédiction en temps réel avec toutes les probabilités.

---

## 📊 Résultats attendus

| Métrique | Valeur |
|---|---|
| Test Accuracy | 82 – 88 % |
| Test Loss | 0.45 – 0.60 |
| Paramètres totaux | ~1.2 million |
| Époques effectives | 50 – 80 |

## 🔧 Architecture CNN

```
Input (32×32×3)
  ↓
[Conv2D(32) → BN → ReLU] × 2 → MaxPool → Dropout(0.25)
  ↓
[Conv2D(64) → BN → ReLU] × 2 → MaxPool → Dropout(0.35)
  ↓
[Conv2D(128) → BN → ReLU] × 2 → MaxPool → Dropout(0.45)
  ↓
Flatten → Dense(256) → BN → ReLU → Dropout(0.5)
  ↓
Dense(10) → Softmax
```
