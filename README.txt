# Détection de fraude — application locale

Application Streamlit locale permettant d’explorer et de prioriser des transactions potentiellement frauduleuses à l’aide d’un modèle de machine learning.

Le dashboard présente les résultats du modèle, les alertes à traiter selon un budget donné et une explication des facteurs qui contribuent au score de fraude.

## Fonctionnalités

- visualisation des transactions et du taux de fraude ;
- comparaison des modèles selon plusieurs métriques ;
- réglage du seuil de score, du budget d’alertes et du coût d’analyse ;
- calcul de la précision, du rappel, des fraudes manquées et du coût total ;
- classement des alertes par score de risque ;
- explication locale des prédictions à l’aide des contributions SHAP ;
- fonctionnement entièrement local, sans serveur externe ni API distante.

## Prérequis

- Python 3.10 ou 3.11 en 64 bits ;
- Windows, macOS ou Linux ;
- les fichiers du modèle et les artefacts présents dans `models/` et `artifacts/`.

## Installation

Depuis le dossier du projet :

```bash
python -m venv .venv
```

Activation de l’environnement virtuel sous Windows PowerShell :

```powershell
.venv\Scripts\Activate.ps1
```

Activation sous Windows avec l’invite de commandes :

```bat
.venv\Scripts\activate.bat
```

Installation des dépendances :

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

Ouvrir ensuite l’adresse affichée par Streamlit, généralement :

```text
http://localhost:8501
```

Sous Windows, le fichier `run_app.bat` peut également être utilisé pour lancer l’application.

## Organisation du projet

```text
app.py                    Interface Streamlit et dashboard
fraud_features.py         Variables et libellés utilisés par le modèle
models/models.joblib      Modèles entraînés
artifacts/                Données, métriques, configuration et explications
requirements.txt          Dépendances Python
run_app.bat               Script de lancement Windows
Challenge2_local.ipynb    Notebook d’analyse et d’expérimentation
```

## Utilisation du dashboard

Les paramètres disponibles dans la barre latérale permettent de simuler une politique de traitement des alertes :

- **Seuil de score** : score minimal pour considérer une transaction comme risquée ;
- **Budget d’alertes** : part maximale des transactions pouvant être analysées ;
- **Coût d’analyse** : coût associé au traitement d’une alerte.

Les onglets permettent ensuite de consulter les données, le comparatif des modèles, les alertes et l’explication détaillée d’une prédiction.

## Remarques

L’application utilise les fichiers déjà présents dans le dépôt. Les chemins vers les données et les artefacts sont relatifs au dossier racine du projet ; il est donc recommandé de lancer Streamlit depuis ce dossier.
