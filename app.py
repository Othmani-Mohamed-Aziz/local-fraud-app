import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from fraud_features import FEATURES, LIBELLES, codes_motifs

st.set_page_config(page_title='Détection de fraude explicable', layout='wide')

st.markdown("""
<style>
    :root {
        --navy: #102a43;
        --blue: #2563eb;
        --muted: #627d98;
        --line: #d9e2ec;
        --surface: #f7f9fc;
    }

    .block-container {
        max-width: 1440px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(115deg, #102a43 0%, #174a73 58%, #2563eb 100%);
        border-radius: 20px;
        padding: 1.8rem 2rem 1.7rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 12px 30px rgba(16, 42, 67, .16);
    }

    .hero-eyebrow {
        color: #9ed8ff;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }

    .hero h1 {
        color: white;
        font-size: clamp(1.8rem, 3vw, 2.65rem);
        line-height: 1.1;
        margin: 0 0 .65rem;
    }

    .hero p {
        color: #d9efff;
        margin: 0;
        font-size: .95rem;
    }

    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 12px rgba(16, 42, 67, .05);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--navy);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: .35rem;
        border-bottom: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: .65rem 1rem;
        color: var(--muted);
    }

    .stTabs [aria-selected="true"] {
        color: var(--blue);
        font-weight: 700;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] h2 {
        color: var(--navy);
        font-size: 1.05rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def charger():
    d = pd.read_parquet('artifacts/app_data.parquet')
    c = np.load('artifacts/contribs.npy')
    cfg = json.load(open('artifacts/config.json'))
    comp = pd.read_csv('artifacts/comparatif.csv')
    mois = pd.read_csv('artifacts/eda_monthly.csv')
    ordre = np.argsort(-d['score'].values)
    return d, c, cfg, comp, mois, ordre


d, contribs, cfg, comp, mois, ordre = charger()
y, amt, score = d['is_fraud'].values, d['amt'].values, d['score'].values

st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Pilotage du risque · Analyse explicable</div>
    <h1>Détection de fraude</h1>
    <p>Modèle <strong>{cfg['modele']}</strong> · période de test : {d['ts'].min():%d/%m/%Y} → {d['ts'].max():%d/%m/%Y}</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Paramètres d'alerte")
st.sidebar.caption("Ajustez les règles de priorisation des transactions à analyser.")
seuil = st.sidebar.slider('Seuil de score', 0.0, 1.0, float(min(cfg['seuil'], 1.0)), 0.005)
budget = st.sidebar.slider("Budget d'alertes (% des transactions)", 0.1, 5.0, float(cfg['budget_pct']), 0.1)
c_alerte = st.sidebar.slider("Coût d'analyse par alerte ($)", 1.0, 100.0, float(cfg['cout_analyse']), 1.0)

k = max(1, int(budget / 100 * len(d)))
n_alertes = int(min(k, (score >= seuil).sum()))
sel = ordre[:n_alertes]
tp = int(y[sel].sum())
precision = tp / n_alertes if n_alertes else 0.0
rappel = tp / max(int(y.sum()), 1)
base = amt[y == 1].sum()
manque = base - amt[sel][y[sel] == 1].sum()
cout = manque + c_alerte * n_alertes

t1, t2, t3, t4 = st.tabs(['📊 Données', '🏁 Comparatif des modèles', '🚨 Alertes & seuil', '🔍 Explication'])

with t1:
    a, b, c = st.columns(3)
    a.metric('Transactions (test)', f'{len(d):,}')
    b.metric('Fraudes (test)', f'{int(y.sum()):,}')
    c.metric('Taux de fraude (test)', f'{y.mean():.2%}')
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(pd.to_datetime(mois['mois'].astype(str)), mois['taux'] * 100, marker='o')
    ax.set(title='Taux de fraude par mois (tout le jeu)', ylabel='%'); ax.grid(alpha=.3)
    st.pyplot(fig)

with t2:
    st.dataframe(comp.round(4), width='stretch')
    st.caption('PR-AUC et ROC-AUC mesurent le classement ; F2, MCC et coût sont calculés au seuil choisi sur la validation (minimisation du coût métier).')

with t3:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Alertes', f'{n_alertes:,}', f'budget max {k:,}')
    c2.metric('Précision', f'{precision:.1%}')
    c3.metric('Rappel', f'{rappel:.1%}')
    c4.metric('Fraudes manquées', f'{manque:,.0f} $')
    c5.metric('Coût total', f'{cout:,.0f} $', f'{cout - base:,.0f} $ vs sans alerte', delta_color='inverse')
    cum = np.cumsum((amt * y)[ordre])
    nn = np.arange(1, len(y) + 1)
    courbe = base - cum + c_alerte * nn
    lim = max(int(0.05 * len(y)), 50)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(nn[:lim:max(lim // 400, 1)], courbe[:lim:max(lim // 400, 1)])
    ax.axvline(max(n_alertes, 1), color='r', ls='--')
    ax.set(title="Coût total selon le nombre d'alertes traitées", xlabel="nombre d'alertes", ylabel='coût ($)'); ax.grid(alpha=.3)
    st.pyplot(fig)
    tab = d.iloc[sel[:200]][['ts', 'category', 'amt', 'score', 'is_fraud']].copy()
    tab.insert(0, 'rang', np.arange(1, len(tab) + 1))
    st.dataframe(tab, width='stretch', hide_index=True)

with t4:
    if n_alertes == 0:
        st.info("Aucune alerte avec ces paramètres : baissez le seuil ou augmentez le budget.")
    else:
        options = list(range(min(n_alertes, 200)))
        choix = st.selectbox('Alerte à expliquer (classée par score décroissant)', options,
                             format_func=lambda j: f"#{j+1} — score {score[sel[j]]:.3f} — {amt[sel[j]]:,.0f} $ — {d['category'].iloc[sel[j]]} — {'FRAUDE' if y[sel[j]] else 'légitime'}")
        i = sel[choix]
        mot = codes_motifs(contribs[i], d[FEATURES].iloc[i].values, n=5)
        st.subheader('Codes motifs')
        st.dataframe(pd.DataFrame(mot), width='stretch', hide_index=True)
        contrib_i = pd.Series(contribs[i][:-1], index=FEATURES)
        top = contrib_i.reindex(contrib_i.abs().sort_values(ascending=False).index).head(10)[::-1]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh([LIBELLES[v][:50] for v in top.index], top.values, color=['tab:red' if v > 0 else 'tab:blue' for v in top.values])
        ax.set(title='Contributions SHAP (rouge = pousse vers « fraude »)', xlabel='contribution au log-odds'); ax.grid(alpha=.3)
        st.pyplot(fig)
    import os
    if os.path.exists('artifacts/shap_beeswarm.png'):
        st.subheader('Explication globale')
        st.image('artifacts/shap_beeswarm.png')
