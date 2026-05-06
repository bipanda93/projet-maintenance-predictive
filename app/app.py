import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")

# ── Configuration page ────────────────────────────
st.set_page_config(
    page_title="Maintenance Prédictive — NASA CMAPSS",
    page_icon="🔧",
    layout="wide"
)

# ── Chemins ───────────────────────────────────────
DATA_PATH  = "/Users/macbook/Projet_maintenance/data/"
MODEL_PATH = "/Users/macbook/Projet_maintenance/app/"

# ── Chargement données ────────────────────────────
@st.cache_data
def load_data():
    cols = ["unit","cycle","opt1","opt2","opt3"] + \
           [f"s{i}" for i in range(1, 22)]
    train = pd.read_csv(DATA_PATH+"train_FD001.txt",
                        sep="\s+", header=None, names=cols)
    test  = pd.read_csv(DATA_PATH+"test_FD001.txt",
                        sep="\s+", header=None, names=cols)
    rul   = pd.read_csv(DATA_PATH+"RUL_FD001.txt",
                        sep="\s+", header=None, names=["RUL"])
    return train, test, rul

# ── Chargement modèle GRU ─────────────────────────
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH+"model_gru.h5")

train, test, rul = load_data()

# ── Sidebar ───────────────────────────────────────
st.sidebar.title("🔧 Maintenance Prédictive")
st.sidebar.markdown("**NASA CMAPSS FD001**")
page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Detection Anomalies",
    "Comparaison Modeles"
])

unit_id = st.sidebar.selectbox(
    "Sélectionner une turbine",
    options=list(range(1, 101))
)

# ── PAGE 1 — DASHBOARD ────────────────────────────
if page == "Dashboard":

    st.title("🔧 Maintenance Prédictive Industrielle")
    st.markdown("**Dataset : NASA CMAPSS FD001 — 100 turbines avion**")
    st.markdown("---")

    # KPIs globaux
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Turbines totales", "100")
    col2.metric("Features capteurs", "14")
    col3.metric("Meilleur modèle", "GRU")
    col4.metric("RMSE", "54.56 cycles")

    st.markdown("---")

    # Données turbine sélectionnée
    turbine_data = train[train["unit"] == unit_id].copy()
    max_cycle    = turbine_data["cycle"].max()
    st.subheader(f"Turbine {unit_id} — {max_cycle} cycles de vie")

    # Graphique évolution capteurs
    feature_cols = ["s2","s3","s4","s7","s11","s12","s15","s17"]
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Evolution des capteurs**")
        fig = px.line(turbine_data, x="cycle", y=feature_cols,
                      title=f"Capteurs — Turbine {unit_id}")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**RUL réel**")
        turbine_data["RUL"] = max_cycle - turbine_data["cycle"]
        turbine_data["RUL_clipped"] = turbine_data["RUL"].clip(upper=125)
        fig2 = px.line(turbine_data, x="cycle", y="RUL_clipped",
                       title=f"RUL plafonné — Turbine {unit_id}",
                       color_discrete_sequence=["coral"])
        st.plotly_chart(fig2, use_container_width=True)

    # Jauge santé
    st.markdown("---")
    st.subheader("Etat de santé de la turbine")

    rul_value = int(rul["RUL"].iloc[unit_id - 1])
    sante     = min(100, int(rul_value / 125 * 100))

    if sante > 60:
        color = "green"
        status = "✅ Normale"
    elif sante > 30:
        color = "orange"
        status = "⚠️ Surveillance"
    else:
        color = "red"
        status = "🚨 Critique"

    col_gauge, col_info = st.columns(2)
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sante,
            title={"text": f"Santé Turbine {unit_id}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30],  "color": "#ffcccc"},
                    {"range": [30, 60], "color": "#fff3cc"},
                    {"range": [60, 100],"color": "#ccffcc"},
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_info:
        st.markdown(f"### Statut : {status}")
        st.markdown(f"**RUL restant :** {rul_value} cycles")
        st.markdown(f"**Santé :** {sante}%")
        if rul_value < 30:
            st.error("🚨 ALERTE — Maintenance urgente requise !")
        elif rul_value < 75:
            st.warning("⚠️ Planifier une maintenance prochainement")
        else:
            st.success("✅ Turbine en bon état")
            
# ── PAGE 2 — DETECTION ANOMALIES ─────────────────
elif page == "Detection Anomalies":

    st.title("🚨 Détection d'Anomalies")
    st.markdown("**Turbines à risque — Seuil RUL < 75 cycles**")
    st.markdown("---")

    # Tableau turbines à risque
    rul_df = pd.DataFrame({
        "Turbine"    : list(range(1, 101)),
        "RUL_restant": rul["RUL"].values,
    })
    rul_df["Sante_%"]  = (rul_df["RUL_restant"] / 125 * 100).clip(0, 100).round(1)
    rul_df["Statut"]   = rul_df["RUL_restant"].apply(
        lambda x: "🚨 Critique" if x < 30 else
                  "⚠️ Surveillance" if x < 75 else
                  "✅ Normale"
    )

    # KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Turbines critiques (RUL<30)",
                len(rul_df[rul_df["RUL_restant"] < 30]))
    col2.metric("Turbines surveillance (RUL<75)",
                len(rul_df[rul_df["RUL_restant"] < 75]))
    col3.metric("Turbines normales",
                len(rul_df[rul_df["RUL_restant"] >= 75]))

    st.markdown("---")

    # Graphique RUL toutes turbines
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**RUL par turbine**")
        colors = rul_df["RUL_restant"].apply(
            lambda x: "red" if x < 30 else
                      "orange" if x < 75 else "green"
        )
        fig = px.bar(rul_df, x="Turbine", y="RUL_restant",
                     color="Statut",
                     color_discrete_map={
                         "🚨 Critique"    : "red",
                         "⚠️ Surveillance": "orange",
                         "✅ Normale"      : "green"
                     },
                     title="RUL restant par turbine")
        fig.add_hline(y=75, line_dash="dash",
                      line_color="orange", annotation_text="Seuil 75")
        fig.add_hline(y=30, line_dash="dash",
                      line_color="red", annotation_text="Seuil critique 30")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Distribution santé turbines**")
        fig2 = px.pie(rul_df, names="Statut",
                      title="Répartition état des turbines",
                      color="Statut",
                      color_discrete_map={
                          "🚨 Critique"    : "red",
                          "⚠️ Surveillance": "orange",
                          "✅ Normale"      : "green"
                      })
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Tableau détaillé
    st.subheader("Tableau détaillé — Turbines à surveiller")
    risque = rul_df[rul_df["RUL_restant"] < 75].sort_values("RUL_restant")
    st.dataframe(risque, use_container_width=True)

# ── PAGE 3 — COMPARAISON MODELES ──────────────────
elif page == "Comparaison Modeles":

    st.title("📊 Comparaison des Modèles")
    st.markdown("**Régression RUL + Détection Anomalies**")
    st.markdown("---")

    # Tableau régression
    st.subheader("Régression — Prédiction RUL")
    df_reg = pd.DataFrame({
        "Modele"  : ["GRU", "MLP", "LSTM", "CNN1D"],
        "RMSE"    : [54.5579, 55.1139, 55.1456, 55.4816],
        "MAE"     : [42.8553, 43.6014, 43.6313, 44.3044],
        "R2"      : [-0.8682, -0.9065, -0.9087, -0.9320],
        "Duree_s" : [382.0, 104.35, 703.35, 157.17],
        "Params"  : [96769, 148993, 126849, 126017]
    })
    st.dataframe(df_reg, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df_reg, x="Modele", y="RMSE",
                     color="Modele", title="RMSE par modele")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(df_reg, x="Modele", y="Duree_s",
                      color="Modele", title="Duree entrainement (s)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Tableau anomalies
    st.subheader("Détection Anomalies")
    df_anomaly = pd.DataFrame({
        "Methode"  : ["Seuil GRU (RUL<75)", "Seuil GRU (RUL<30)",
                      "Autoencoder", "Isolation Forest"],
        "Precision": [0.9714, 0.9801, 0.5263, 0.5205],
        "Recall"   : [0.2511, 0.0828, 0.0477, 0.0320],
        "F1"       : [0.3990, 0.1528, 0.0875, 0.0603],
        "ROC_AUC"  : [0.2952, 0.3148, 0.5385, 0.5081]
    })
    st.dataframe(df_anomaly, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.bar(df_anomaly, x="Methode", y="F1",
                      color="Methode", title="F1-score Detection Anomalies")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.bar(df_anomaly, x="Methode", y="Precision",
                      color="Methode", title="Precision Detection Anomalies")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # Conclusions
    st.subheader("Conclusions scientifiques")
    col5, col6 = st.columns(2)

    with col5:
        st.info("""
        **Meilleur modèle régression : GRU**
        - RMSE : 54.56 cycles
        - MAE  : 42.86 cycles
        - R²   : -0.868
        - Durée : 382s
        """)

    with col6:
        st.success("""
        **Meilleure détection anomalies : Seuil GRU (RUL<75)**
        - Precision : 97.14%
        - Recall    : 25.11%
        - F1        : 39.90%
        - Fiable en production industrielle
        """)