# -*- coding: utf-8 -*-
"""
app.py – Film Sentiment Analyzer
Stack: RoBERTa (HuggingFace) + Streamlit + Plotly
Launch: streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from transformers import pipeline

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CineScope · Sentiment AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
#  CSS – dark cinema / IMDB amber accent / glassmorphism
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* =========================================
   THÈME CLAIR — LISIBILITÉ MAXIMALE
   ========================================= */

/* Base resets & Typography */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #f5f7fa !important;
    color: #1a1a2e !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stHeader"] {
    background: #ffffff !important;
    border-bottom: 1px solid #e2e8f0 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #1a1a2e !important;
    -webkit-text-fill-color: #1a1a2e !important;
    letter-spacing: -0.3px;
    text-shadow: none !important;
    background: none !important;
    animation: none !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 10px rgba(0,0,0,0.06) !important;
}

[data-testid="stSidebar"] * {
    color: #1a1a2e !important;
}

/* Glass card (light version) */
.glass-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.1);
}

/* Streamlit Native Elements (Metrics) */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
    transition: all 0.3s ease !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    border-color: #c084fc !important;
    box-shadow: 0 6px 20px rgba(192,132,252,0.15) !important;
}

[data-testid="stMetricValue"] {
    color: #7c3aed !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #4b5563 !important;
    font-weight: 600 !important;
}

/* Inputs */
div[data-baseweb="textarea"] > div, div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}
div[data-baseweb="textarea"]:focus-within > div, div[data-baseweb="select"]:focus-within > div {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
}
textarea, input { color: #1a1a2e !important; font-size: 1rem !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    box-shadow: 0 4px 14px rgba(124,58,237,0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.45) !important;
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-bottom: 2px solid #e2e8f0 !important;
    border-radius: 12px 12px 0 0;
    padding: 0.4rem 0.4rem 0 0.4rem;
    gap: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    color: #6b7280 !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.5rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #f3f4f6 !important;
    color: #1a1a2e !important;
}
.stTabs [aria-selected="true"] {
    background: #ede9fe !important;
    color: #7c3aed !important;
    border-bottom: 3px solid #7c3aed !important;
    font-weight: 600 !important;
}

/* Result Cards */
.result-positive, .result-negative, .result-neutral {
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    border: 1px solid;
    border-left: 5px solid;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.result-positive {
    background: linear-gradient(135deg, #f0fdf4, #dcfce7);
    border-color: #bbf7d0;
    border-left-color: #16a34a;
}
.result-negative {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border-color: #fecaca;
    border-left-color: #dc2626;
}
.result-neutral {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border-color: #fde68a;
    border-left-color: #d97706;
}
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

/* Badges */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 1.2rem;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.badge-pos { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
.badge-neg { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
.badge-neu { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }

/* Dataframe & Scrollbar */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #7c3aed; }

/* Paragraph text */
p, li, label, .stMarkdown { color: #374151 !important; }

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  LOAD MODEL (cached once)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_roberta():
    """Load RoBERTa sentiment model — cached across reruns."""
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None,
        truncation=True,
        max_length=512,
    )

with st.spinner("Chargement du modèle RoBERTa..."):
    sentiment_pipe = load_roberta()

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
LABEL_MAP = {
    "positive": ("Positif", "#16a34a", "result-positive", "badge-pos"),
    "negative": ("Négatif", "#dc2626", "result-negative", "badge-neg"),
    "neutral":  ("Neutre",  "#d97706", "result-neutral",  "badge-neu"),
}

def analyse(text: str):
    """Run inference; return (top_label, scores_dict)."""
    results = sentiment_pipe(text[:512])[0]
    scores = {r["label"]: r["score"] for r in results}
    top = max(scores, key=scores.get)
    return top, scores

def make_gauge(scores: dict):
    """Plotly radial gauge for top sentiment confidence."""
    top = max(scores, key=scores.get)
    val = scores[top] * 100
    color = LABEL_MAP[top][1]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"suffix": "%", "font": {"size": 36, "color": "#1a1a2e"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9ca3af", "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#f3f4f6",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "rgba(220,38,38,0.08)"},
                {"range": [33, 66], "color": "rgba(217,119,6,0.08)"},
                {"range": [66, 100], "color": "rgba(22,163,74,0.08)"},
            ],
        },
    ))
    fig.update_layout(
        height=240, margin=dict(t=30, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#374151",
    )
    return fig

def make_bar(scores: dict):
    """Horizontal bar chart of all 3 class probabilities."""
    labels = ["Négatif", "Neutre", "Positif"]
    keys = ["negative", "neutral", "positive"]
    vals = [scores.get(k, 0) * 100 for k in keys]
    colors = ["#f44336", "#F5C518", "#4caf50"]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=colors, text=[f"{v:.1f}%" for v in vals],
        textposition="auto", textfont=dict(color="#ffffff", size=13),
    ))
    fig.update_layout(
        height=200, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(243,244,246,0.8)",
        xaxis=dict(range=[0, 100], showgrid=False, color="#6b7280"),
        yaxis=dict(color="#374151"),
        font=dict(family="Inter", color="#374151"),
    )
    return fig

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 1rem;">
        <div style="font-size:3rem;">🎬</div>
        <div style="font-family:'Playfair Display',serif; font-size:1.4rem;
                    color:#7c3aed; font-weight:900;">CineScope</div>
        <div style="font-size:0.75rem; color:#6b7280; letter-spacing:2px;
                    text-transform:uppercase; margin-top:0.3rem;">Sentiment AI</div>
    </div>
    <hr style="border-color:#e2e8f0;">
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio(
        label="Navigation Menu",
        options=["🔮 Prédiction", "📊 Exploration", "ℹ️ À propos"],
        label_visibility="collapsed",
    )
    st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)
    st.success("✅ RoBERTa chargé")
    st.markdown("""
    <div style="text-align:center; font-size:0.7rem; color:#6b7280; margin-top:1rem;">
        cardiffnlp/twitter-roberta-base-sentiment-latest<br>
        3 classes · HuggingFace Transformers
    </div>
    <hr style="border-color:#e2e8f0; margin-top:1.5rem;">
    <div style="padding: 0.8rem 0;">
        <div style="font-size:0.7rem; font-weight:700; color:#7c3aed; text-transform:uppercase;
                    letter-spacing:1.5px; margin-bottom:0.8rem;">👩‍💻 Réalisé par</div>
        <div style="font-size:0.85rem; color:#374151; line-height:2;">
            • OUKMI Fatima Zahra<br>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  PAGE 1 — PRÉDICTION
# ═══════════════════════════════════════════════════════════════
if page == "🔮 Prédiction":
    st.markdown("# 🔮 Analyse de sentiment")
    st.markdown("Colle un avis de film — le modèle RoBERTa détecte instantanément le ton "
                "(positif · négatif · neutre) avec les scores de confiance.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_in, col_out = st.columns([1.1, 0.9], gap="large")

    with col_in:
        st.markdown("#### ✍️ Ton avis")
        review = st.text_area(
            label="Saisis ton avis de film", height=220, label_visibility="collapsed",
            placeholder="Ex: This movie was absolutely brilliant! The performances "
                        "were outstanding and the story kept me on the edge of my seat...",
        )
        c1, c2 = st.columns(2)
        with c1:
            go_btn = st.button("🎯 Analyser", use_container_width=True)
        with c2:
            if st.button("🗑️ Effacer", use_container_width=True):
                st.rerun()

    with col_out:
        st.markdown("#### 🎬 Résultat")
        if go_btn and review.strip():
            with st.spinner("Analyse RoBERTa..."):
                top, scores = analyse(review)

            fr, color, card_cls, badge_cls = LABEL_MAP[top]
            emoji = {"positive": "🌟", "negative": "💀", "neutral": "😐"}[top]
            msg = {"positive": "L'avis est enthousiaste !",
                   "negative": "L'avis est critique.",
                   "neutral": "L'avis est neutre."}[top]

            st.markdown(f"""
            <div class="{card_cls}">
                <span class="badge {badge_cls}">{emoji} {fr}</span>
                <p class="result-title" style="color:{color}; margin-top:0.8rem;">{msg}</p>
                <p class="result-sub">
                    Confiance : <strong style="color:#F5C518;">{scores[top]:.0%}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(make_gauge(scores), use_container_width=True)
            st.markdown("##### Distribution des scores")
            st.plotly_chart(make_bar(scores), use_container_width=True)

        elif go_btn:
            st.warning("Saisis un avis avant d'analyser.")
        else:
            st.markdown("""
            <div style="border: 2px dashed #e2e8f0; border-radius: 14px; padding: 2.5rem;
                        text-align: center; color: #9ca3af; margin-top: 0.5rem; background:#fafafa;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎭</div>
                <div style="font-weight:500;">Le résultat apparaîtra ici</div>
            </div>
            """, unsafe_allow_html=True)

    # Quick examples
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💡 Exemples rapides")
    examples = {
        "👍 Enthousiaste": "This film is a masterpiece! The director's vision is breathtaking, "
                           "the actors deliver stunning performances, and the soundtrack is mesmerizing.",
        "👎 Déçu": "What a waste of time. The plot was predictable, the acting was terrible, "
                   "and I nearly fell asleep. Completely disappointed.",
        "😐 Mitigé": "The movie had its moments but overall felt flat. Some scenes were "
                     "brilliant while others dragged on for too long.",
    }
    cols = st.columns(3)
    for col, (lbl, txt) in zip(cols, examples.items()):
        with col:
            if st.button(lbl, use_container_width=True):
                st.info(txt[:140] + "...")

# ═══════════════════════════════════════════════════════════════
#  PAGE 2 — EXPLORATION
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Exploration":
    st.markdown("# 📊 Exploration du dataset")

    @st.cache_data
    def load_data():
        df = pd.read_csv("IMDB_Dataset.csv")
        df["review_length"] = df["review"].apply(len)
        return df

    try:
        df = load_data()
    except FileNotFoundError:
        st.error("IMDB_Dataset.csv introuvable.")
        st.stop()

    total = len(df)
    pos = (df["sentiment"] == "positive").sum()
    neg = (df["sentiment"] == "negative").sum()
    avg_len = int(df["review_length"].mean())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📋 Total avis", f"{total:,}")
    k2.metric("🟢 Positifs", f"{pos:,}", f"{pos/total:.0%}")
    k3.metric("🔴 Négatifs", f"{neg:,}", f"{neg/total:.0%}")
    k4.metric("📏 Longueur moy.", f"{avg_len:,} car.")

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📏 Longueur", "🔍 Aperçu"])

    with tab1:
        counts = df["sentiment"].value_counts()
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(
                x=counts.index.tolist(), y=counts.values.tolist(),
                marker_color=["#4caf50", "#f44336"],
                text=counts.values.tolist(), textposition="auto",
                textfont=dict(color="#e8e0d0", size=14),
            ))
            fig.update_layout(
                title="Répartition des sentiments",
                height=380, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#f9fafb",
                font=dict(color="#374151"),
                title_font_color="#7c3aed",
                xaxis=dict(color="#6b7280"), yaxis=dict(color="#6b7280", gridcolor="#e5e7eb"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure(go.Pie(
                labels=counts.index.tolist(), values=counts.values.tolist(),
                marker=dict(colors=["#4caf50", "#f44336"],
                            line=dict(color="#0d0d0f", width=2)),
                hole=0.45, textinfo="label+percent",
                textfont=dict(color="#e8e0d0", size=13),
            ))
            fig.update_layout(
                title="Proportion",
                height=380, paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#374151"),
                title_font_color="#7c3aed",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Histogram(
                x=df["review_length"], nbinsx=60,
                marker_color="#F5C518", opacity=0.85,
            ))
            fig.update_layout(
                title="Distribution des longueurs",
                height=380, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#f9fafb",
                font=dict(color="#374151"),
                title_font_color="#7c3aed",
                xaxis=dict(title="Caractères", color="#6b7280"),
                yaxis=dict(title="Nombre", color="#6b7280", gridcolor="#e5e7eb"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure()
            for sent, color in [("positive", "#4caf50"), ("negative", "#f44336")]:
                fig.add_trace(go.Box(
                    y=df[df["sentiment"] == sent]["review_length"],
                    name=sent.capitalize(), marker_color=color,
                ))
            fig.update_layout(
                title="Longueur par sentiment",
                height=380, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#f9fafb",
                font=dict(color="#374151"),
                title_font_color="#7c3aed",
                yaxis=dict(title="Caractères", color="#6b7280", gridcolor="#e5e7eb"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        n = st.slider("Lignes à afficher", 5, 50, 10)
        filt = st.selectbox("Filtrer", ["Tous", "positive", "negative"])
        df_show = df if filt == "Tous" else df[df["sentiment"] == filt]
        st.dataframe(
            df_show[["sentiment", "review_length", "review"]].head(n),
            use_container_width=True, hide_index=True,
            column_config={
                "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                "review_length": st.column_config.NumberColumn("Longueur", width="small"),
                "review": st.column_config.TextColumn("Avis", width="large"),
            },
        )

# ═══════════════════════════════════════════════════════════════
#  PAGE 3 — À PROPOS
# ═══════════════════════════════════════════════════════════════
elif page == "ℹ️ À propos":
    st.markdown("# ℹ️ À propos du projet")

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.markdown("""
        ## Objectif
        Prédire automatiquement le **sentiment** (positif, négatif ou neutre)
        d'un avis de film à partir de son texte brut, en utilisant un modèle
        pré-entraîné RoBERTa.

        ## Modèle
        **cardiffnlp/twitter-roberta-base-sentiment-latest** — un modèle
        RoBERTa fine-tuné sur ~124M tweets, capable de classifier en 3 classes
        (positive / negative / neutral) avec une excellente précision.

        ## Dataset d'exploration
        Le dataset IMDB contient **50 000 avis** de films, parfaitement
        équilibrés (25 000 positifs / 25 000 négatifs).

        ## Avantages vs SVM+TF-IDF
        | Critère | SVM + TF-IDF | RoBERTa |
        |---------|-------------|---------|
        | **Contexte** | Bag-of-words | Attention bidirectionnelle |
        | **Classes** | 2 (pos/neg) | 3 (pos/neg/neutre) |
        | **Ironie** | ❌ Mal gérée | ✅ Mieux captée |
        | **Entraînement** | Requis | Pré-entraîné (zero-shot) |
        """)

    with c2:
        st.markdown("""
        <div class="glass-card">
            <div style="font-family:'Playfair Display',serif; font-size:1.1rem;
                        color:#7c3aed; margin-bottom:1rem;">🛠️ Stack technique</div>
        """, unsafe_allow_html=True)

        stack = [
            ("🐍", "Python 3.x"),
            ("🤗", "HuggingFace Transformers"),
            ("🧠", "RoBERTa (cardiffnlp)"),
            ("🎈", "Streamlit"),
            ("📊", "Plotly"),
            ("📋", "Pandas / NumPy"),
        ]
        for icon, name in stack:
            st.markdown(f"""
            <div style="background:#f3f4f6; border-radius:8px; padding:0.5rem 0.8rem;
                        display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;
                        border: 1px solid #e5e7eb;">
                <span>{icon}</span>
                <span style="font-size:0.9rem; color:#374151; font-weight:500;">{name}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
