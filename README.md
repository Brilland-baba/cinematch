#  🎬 BrillantCiné — High-Scale Recommender System🎬 

> **Moteur de recommandation de films** basé sur l'algorithme ALS (Alternating Least Squares) entraîné sur le dataset MovieLens 1M.

[![Streamlit App](https://brillantcine.streamlit.app/)

---

## 🚀 Démo live

**👉 [https://alsfilms.streamlit.app](https://alsfilms.streamlit.app)**

---

## 📋 Description

CineMatch est une plateforme de recommandation personnalisée qui :

- 🎬 Recommande des films en temps réel via **ALS (Alternating Least Squares)**
- ⭐ Met à jour les recommandations **instantanément** après chaque notation (< 2ms)
- 🌐 Supporte les films hors-catalogue via l'**API TMDB**
- 🧊 Gère le **cold start** avec un scoring bayésien pour les nouveaux utilisateurs
- 📊 Expose les **paramètres du modèle** et l'analyse des prédictions

---

## 🧠 Architecture technique

```
MovieLens 1M Dataset
    │
    ▼
ALS Factorization (implicit)
    │  k=50 facteurs latents
    │  α=40 (confiance)
    │  λ=0.01 (régularisation)
    ▼
Vecteurs latents U (users) × I (items)
    │
    ▼
score(u,i) = uᵀ · i  →  Top-K recommandations
    │
    ▼
Interface Streamlit  +  TMDB API (posters)
```

---

## 📊 Dataset — MovieLens 1M

| Métrique | Valeur |
|----------|--------|
| Notations | 1 000 209 |
| Utilisateurs | 6 040 |
| Films | 3 706 |
| Sparsité | **95.53 %** |
| Période | Films sortis avant 2001 |

---

## ⚙️ Stack technique

| Composant | Technologie |
|-----------|-------------|
| Modèle ALS | `implicit` (Python) |
| Interface | Streamlit 1.33+ |
| Posters | TMDB API |
| Hébergement | Streamlit Community Cloud |
| Données | MovieLens 1M |

---

## 🗂️ Structure du projet

```
cinematch/
├── app.py                  # Application Streamlit principale
├── als_recommender.py      # Moteur ALS (classe + méthodes)
├── requirements.txt        # Dépendances Python
├── model/
│   ├── als_model.pkl       # Modèle ALS entraîné
│   ├── matrix_conf.npz     # Matrice de confiance
│   ├── matrix_seen.npz     # Films vus par utilisateur
│   └── meta.pkl            # Encodeurs user/item
└── ml-1m/
    ├── movies.dat          # 3 706 films
    ├── ratings.dat         # 1 000 209 notations
    └── users.dat           # 6 040 utilisateurs
```

---

## 🔧 Lancer en local

```bash
# Cloner le repo
git clone https://github.com/Brilland-baba/1M_Recommand.git
cd 1M_Recommand

# Installer les dépendances
pip install -r requirements.txt

# Configurer la clé TMDB
mkdir -p .streamlit
echo 'TMDB_API_KEY = "votre_cle_tmdb"' > .streamlit/secrets.toml

# Lancer l'application
streamlit run app.py
```

---

## 🧪 Évaluation live

L'enseignant peut se connecter sur **https://brillantcine.streamlit.app/**, entrer n'importe quel prénom, noter un film et observer les recommandations se mettre à jour en temps réel.

---

## 👤 Auteur

**BABA C.F. Brilland**
ISE — Promotion 2024-2025
ENEAM — Advanced Machine Learning
Enseignant : Rodéo Oswald Y. TOHA (Engineer in CV & GenAI)

---

## 📄 Licence

Projet académique — ENEAM / ISE 2025
