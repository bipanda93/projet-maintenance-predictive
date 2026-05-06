# Maintenance Predictive Industrielle — NASA CMAPSS

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)
![MLflow](https://img.shields.io/badge/MLflow-2.19-blue?logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Production-green)
![API](https://img.shields.io/badge/API-REST-brightgreen)

> Prediction de la duree de vie restante (RUL) de turbines avion
> et detection d anomalies sur le dataset NASA CMAPSS FD001.
> 5 modeles Deep Learning compares + MLflow Model Registry
> + API REST deployee + interface Streamlit 3 pages.

## Structure du projet
projet-maintenance-predictive/
├── notebooks/
│   └── projet4_maintenance.ipynb  ← pipeline complet
├── app/
│   └── app.py                     ← interface Streamlit
├── results/
│   └── comparaison_finale.png     ← graphiques comparatifs
├── requirements.txt
└── README.md

## Resultats — Regression RUL

| Modele | RMSE  | MAE   | R2     | Duree |
|--------|-------|-------|--------|-------|
| GRU    | 54.56 | 42.86 | -0.868 | 382s  |
| MLP    | 55.11 | 43.60 | -0.906 | 104s  |
| LSTM   | 55.15 | 43.63 | -0.909 | 703s  |
| CNN1D  | 55.48 | 44.30 | -0.932 | 157s  |

## Resultats — Detection Anomalies

| Methode            | Precision | Recall | F1   | ROC-AUC |
|--------------------|-----------|--------|------|---------|
| Seuil GRU (RUL<75) | 0.97      | 0.25   | 0.40 | 0.30    |
| Autoencoder        | 0.53      | 0.05   | 0.09 | 0.54    |
| Isolation Forest   | 0.52      | 0.03   | 0.06 | 0.51    |

## Deploiement MLflow

### Model Registry — 5 modeles · 2 versions chacun
| Modele           | v1          | v2                    | Statut     |
|------------------|-------------|-----------------------|------------|
| Maintenance_GRU  | sans sign.  | avec signature ✅     | Production |
| Maintenance_MLP  | sans sign.  | avec signature ✅     | Staging    |
| Maintenance_CNN1D| sans sign.  | avec signature ✅     | Archived   |
| Maintenance_LSTM | sans sign.  | avec signature ✅     | Archived   |
| Maintenance_AE   | sans sign.  | avec signature ✅     | Archived   |

### Lancer MLflow UI
mlflow ui --backend-store-uri file:///path/to/mlruns --port 5004
→ http://127.0.0.1:5004

### API REST — GRU en production
export MLFLOW_TRACKING_URI="file:///path/to/mlruns"
mlflow models serve --model-uri "models:/Maintenance_GRU/2" --port 5003 --no-conda
→ http://127.0.0.1:5003/invocations

### Test API
python3 -c "
import requests, numpy as np, json
data = np.random.randn(1, 30, 14).tolist()
r = requests.post('http://127.0.0.1:5003/invocations',
    headers={'Content-Type': 'application/json'},
    data=json.dumps({'inputs': data}))
print(r.json())
# {'predictions': [[124.24]]}  → RUL predit en cycles
"

## Interface Streamlit
streamlit run app/app.py
→ http://localhost:8501

Pages :
- Dashboard           : KPIs, capteurs, jauge sante turbine
- Detection Anomalies : turbines a risque, alertes, pie chart
- Comparaison Modeles : tableaux et graphiques comparatifs

## Dataset
NASA CMAPSS FD001
- 100 turbines avion
- 20 631 echantillons train · 13 096 test
- 14 features capteurs selectionnes
- 3 approches RUL : lineaire, plafonne (125), piecewise
- Fenetre temporelle : 30 cycles (sliding window)

## Stack technique
- Python 3.10
- TensorFlow 2.13 · Keras
- MLflow 2.19 · Model Registry · Signatures · API REST
- Streamlit · Plotly
- Scikit-learn · Pandas · NumPy · Matplotlib

## Installation
git clone https://github.com/bipanda93/projet-maintenance-predictive.git
cd projet-maintenance-predictive
conda create -n maintenance_env python=3.10 -y
conda activate maintenance_env
pip install -r requirements.txt
streamlit run app/app.py

## Auteur
Bipanda Franck Ulrich
Mastere Data Engineering — Digital School de Paris — 2026
github.com/bipanda93 · datascienceportfol.io/bipandaf
