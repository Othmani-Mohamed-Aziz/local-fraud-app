APPLICATION LOCALE — DETECTION DE FRAUDE

1. Installer Python 3.10 ou 3.11 (64 bits).
2. Ouvrir un terminal dans ce dossier.
3. Créer un environnement : python -m venv .venv
4. Windows : .venv\Scripts\activate
5. Installer : python -m pip install --upgrade pip
6. Installer les dépendances : pip install -r requirements.txt
7. Lancer : streamlit run app.py
8. Ouvrir http://localhost:8501 dans le navigateur.

Les modèles entraînés sont dans models/models.joblib. Le dashboard charge le modèle XGBoost et les artefacts dans artifacts/.
Aucun serveur externe, tunnel ou service API n'est requis.
