# CineScope · Sentiment AI

**CineScope** est une application web interactive de Traitement du Langage Naturel (NLP) permettant d'analyser instantanément le sentiment des critiques de films. 

Propulsée par le modèle de deep learning **RoBERTa**, l'application classe les commentaires en trois catégories (Positif, Négatif, Neutre) avec un haut niveau de précision en captant les nuances contextuelles et l'ironie.

---

## Stack Technique
- **Modèle de Langage :** `cardiffnlp/twitter-roberta-base-sentiment-latest` (Hugging Face Transformers)
- **Interface Utilisateur :** Streamlit
- **Visualisation de Données :** Plotly (Jauges radiales et graphiques de distribution)
- **Analyse de Données :** Pandas, NumPy

---

## Fonctionnalités
- **Mode Prédiction :** Analyse en temps réel d'un texte personnalisé avec distribution des scores de confiance.
- **Mode Exploration :** Visualisation dynamique et interactive du dataset IMDB (50 000 avis équilibrés).
- **Indicateurs Métriques :** Analyse de la longueur des avis et répartition statistique globale.

---
