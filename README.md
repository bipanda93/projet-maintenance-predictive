# Maintenance Predictive Industrielle — NASA CMAPSS

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?logo=tensorflow)
![MLflow](https://img.shields.io/badge/MLflow-2.19-blue?logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Status](https://img.shields.io/badge/Status-Production-green)

> Prediction de la duree de vie restante (RUL) de turbines avion
> et detection d anomalies sur le dataset NASA CMAPSS FD001.
> 5 modeles Deep Learning compares + interface Streamlit deployee.

## Resultats

| Modele | RMSE | MAE | R2 | Duree |
|--------|------|-----|----|-------|
| GRU | 54.56 | 42.86 | -0.868 | 382s |
| MLP | 55.11 | 43.60 | -0.906 | 104s |
| LSTM | 55.15 | 43.63 | -0.909 | 703s |
| CNN1D | 55.48 | 44.30 | -0.932 | 157s |

| Methode | Precision | Recall | F1 | ROC-AUC |
|---------|-----------|--------|----|---------|
| Seuil GRU (RUL<75) | 0.97 | 0.25 | 0.40 | 0.30 |
| Autoencoder | 0.53 | 0.05 | 0.09 | 0.54 |
| Isolation Forest | 0.52 | 0.03 | 0.06 | 0.51 |

## Stack technique
- Python 3.10 · TensorFlow 2.13 · Keras
- MLflow 2.19 · Streamlit · Plotly
- Scikit-learn · Pandas · NumPy
- NASA CMAPSS FD001

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
