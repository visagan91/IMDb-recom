import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import joblib
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

st.set_page_config(page_title="IMDb 2024 Storyline Recommender", page_icon="🎬", layout="wide")
st.title("🎬 IMDb 2024 — Storyline-based Movie Recommender")

@st.cache_resource
def load_assets(kind: str = "tfidf"):
    df = pd.read_csv("data/clean_movies.csv")
    if kind == "tfidf":
        vec = joblib.load("data/tfidf_vectorizer.pkl")
        mat = load_npz("data/tfidf_matrix.npz")
    else:
        vec = joblib.load("data/count_vectorizer.pkl")
        mat = load_npz("data/count_matrix.npz")
    return df, vec, mat

with st.sidebar:
    st.header("Settings")
    vect_kind = st.radio("Vectorizer", ["TF-IDF", "Count"], index=0)
    top_n = st.slider("Top N", min_value=3, max_value=20, value=5, step=1)
    min_chars = st.slider("Min input length", 20, 500, 80, 10)
    st.markdown("---")
    st.caption("Tip: Build indices with `python build_index.py` after scraping.")

df, vectorizer, matrix = load_assets("tfidf" if vect_kind == "TF-IDF" else "count")

st.write("Paste a short storyline/plot. We'll compute cosine similarity against all 2024 movies and show the most similar ones.")

user_text = st.text_area("Your storyline", height=160, placeholder="e.g., A young wizard begins his journey at a magical school...")

colA, colB = st.columns([1, 3])
with colA:
    go = st.button("Recommend")
with colB:
    st.caption("Cosine similarity over " + vect_kind + " features; bigrams enabled; English stopwords removed.")

def recommend(query: str, top_k: int = 5):
    q_vec = vectorizer.transform([query])
    # For TF-IDF, linear_kernel is equivalent to cosine similarity on normalized vectors
    sims = linear_kernel(q_vec, matrix).ravel()
    top_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in top_idx]

if go:
    if not user_text or len(user_text) < min_chars:
        st.warning(f"Please enter at least {min_chars} characters of storyline for better results.")
    else:
        results = recommend(user_text, top_n)
        st.subheader(f"Top {top_n} recommendations")
        for rank, (idx, score) in enumerate(results, start=1):
            row = df.iloc[idx]
            title = row.get("title", "(unknown)")
            url = row.get("url", "")
            storyline = row.get("storyline", "")
            with st.container(border=True):
                st.markdown(f"### {rank}. [{title}]({url})  \n*Similarity:* `{score:.3f}`")
                st.write(storyline)

st.markdown("---")
with st.expander("About"):
    st.write