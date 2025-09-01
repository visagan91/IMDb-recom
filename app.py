#!/usr/bin/env python3
from pathlib import Path
import json, pickle, io
import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse

# ---------- Page config (must be first Streamlit call; only once) ----------
st.set_page_config(
    page_title="IMDb 2024 Storyline Recommender",
    page_icon="🎬",
    layout="wide",
)

ART_DIR = Path("artifacts")

# ---------- Cached loaders ----------
@st.cache_resource
def load_artifacts():
    with open(ART_DIR / "config.json") as f:
        cfg = json.load(f)
    df = pd.read_csv(ART_DIR / "movies_meta.csv")
    X = sparse.load_npz(ART_DIR / "tfidf_2024.npz")
    with open(ART_DIR / "vectorizer.pkl", "rb") as f:
        vec = pickle.load(f)
    return cfg, df, X, vec

@st.cache_data
def prep_df(df: pd.DataFrame, rating_col="Rating", duration_col="Duration"):
    df = df.copy()
    if rating_col in df.columns:
        def _to_float(x):
            try:
                return float(str(x).strip())
            except:
                return np.nan
        df["rating_float"] = df[rating_col].map(_to_float)
    else:
        df["rating_float"] = np.nan

    if duration_col in df.columns:
        df["duration_str"] = df[duration_col].astype(str)
    else:
        df["duration_str"] = ""
    return df

# ---------- Similarity helpers ----------
def cosine_topk(X, q, k):
    # TF-IDF from sklearn is L2-normalized → cosine == dot
    sims = (X @ q.T).toarray().ravel()
    k = min(k, len(sims)) if len(sims) else 1
    order = np.argpartition(-sims, kth=k-1)[:k]
    order = order[np.argsort(-sims[order])]
    return order, sims

def explain_overlap(q, x_row, vec, topn=8):
    prod = x_row.multiply(q)  # elementwise product in sparse
    if prod.nnz == 0:
        return []
    data, idxs = prod.data, prod.indices
    if len(data) <= topn:
        top_i = np.argsort(-data)
    else:
        top_i = np.argpartition(-data, kth=topn-1)[:topn]
        top_i = top_i[np.argsort(-data[top_i])]
    vocab = vec.get_feature_names_out()
    return [vocab[idxs[i]] for i in top_i]

# ---------- Load data once ----------
cfg, df_raw, X, vec = load_artifacts()
TITLE_COL = cfg["title_col"]
STORY_COL = cfg["story_col"]
df = prep_df(df_raw)

# ---------- UI ----------
st.title("🎬 IMDb 2024 — Storyline Recommender (Enhanced)")

with st.sidebar:
    st.header("Choose how to recommend")
    mode = st.radio("Recommendation mode", ["Type a storyline", "More like this movie"], index=0)
    k = st.slider("How many recommendations?", 1, 20, 5)
    min_sim = st.slider("Similarity", 0.0, 1.0, 0.05, 0.01)
    min_rating = st.slider("Min IMDb rating (optional)", 0.0, 10.0, 0.0, 0.1)
    show_explain = st.checkbox("Show why it matched (top terms)", value=True)

    st.markdown("---")
    st.caption("TF-IDF (1–2 grams, English stopwords). Cosine similarity via dot product.")

# ---------- Mode: text query ----------
def run_text_mode():
    st.subheader("Type a storyline")
    user_text = st.text_area(
        "Paste a short plot or concept:",
        height=140,
        placeholder="Military unit in hostile territory must fight for survival..."
    )
    go = st.button("Recommend", type="primary", use_container_width=True)
    if not (go and user_text.strip()):
        return

    q = vec.transform([user_text])
    order, sims = cosine_topk(X, q, k=max(k*4, 50))  # oversample; we’ll filter

    res = df.iloc[order].copy()
    res["similarity"] = sims[order]

    # thresholds
    res = res[res["similarity"] >= min_sim]
    if min_rating > 0:
        res = res[(res["rating_float"].isna()) | (res["rating_float"] >= min_rating)]

    res = res.sort_values("similarity", ascending=False).head(k)

    if res.empty:
        st.warning("No matches over the current thresholds. Try lowering ‘Min similarity’ or entering a bit more detail.")
        return

    for i, (_, r) in enumerate(res.iterrows(), start=1):
        st.markdown(f"### {i}. {r[TITLE_COL]}")
        st.markdown(
            f"**Similarity:** `{r['similarity']:.3f}`"
            + (f"  ·  **Rating:** `{r['rating_float']:.1f}`" if pd.notna(r['rating_float']) else "")
        )
        story = r.get(STORY_COL)
        if isinstance(story, str) and story.strip():
            with st.expander("Storyline", expanded=True):
                st.write(story)

        url = r.get("URL")
        if isinstance(url, str) and url.startswith("http"):
            st.link_button("Open on IMDb", url, use_container_width=False)

        if show_explain:
            idx = r.name
            terms = explain_overlap(q, X[idx], vec, topn=8)
            if terms:
                st.caption("Why it matched:")
                st.write(" · ".join(f"`{t}`" for t in terms))
        st.divider()

    # download
    csv_buf = io.StringIO()
    res[[TITLE_COL, "similarity", "rating_float", STORY_COL, "URL"]].rename(
        columns={"rating_float": "Rating"}
    ).to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download results (CSV)", data=csv_buf.getvalue(),
                       file_name="recommendations.csv", mime="text/csv")

# ---------- Mode: similar to a movie ----------
def run_item_mode():
    st.subheader("More like this movie")
    pick = st.selectbox("Choose a reference movie", options=df[TITLE_COL].tolist())
    go = st.button("Find similar", type="primary", use_container_width=True)
    if not (go and pick):
        return

    base_idx = df.index[df[TITLE_COL] == pick][0]
    q = X[base_idx]

    order, sims = cosine_topk(X, q, k=max(k*10, 500))  # oversample; we will filter
    res = df.iloc[order].copy()
    res["similarity"] = sims[order]

    if exclude_same_title:
        res = res[df[TITLE_COL] != pick]

    res = res[res["similarity"] >= min_sim]
    if min_rating > 0:
        res = res[(res["rating_float"].isna()) | (res["rating_float"] >= min_rating)]

    res = res.sort_values("similarity", ascending=False).take(range(min(k, len(res))))

    if res.empty:
        st.warning("No similar movies over the current thresholds. Try lowering ‘Min similarity’.")
        return

    # reference movie card
    with st.expander("Reference movie", expanded=False):
        r0 = df.iloc[base_idx]
        st.markdown(
            f"**{r0[TITLE_COL]}**"
            + (f"  ·  Rating: `{r0['rating_float']:.1f}`" if pd.notna(r0['rating_float']) else "")
        )
        st.write(r0.get(STORY_COL, ""))
        url0 = r0.get("URL")
        if isinstance(url0, str) and url0.startswith("http"):
            st.link_button("Open on IMDb", url0)

    for i, (_, r) in enumerate(res.iterrows(), start=1):
        st.markdown(f"### {i}. {r[TITLE_COL]}")
        st.markdown(
            f"**Similarity:** `{r['similarity']:.3f}`"
            + (f"  ·  **Rating:** `{r['rating_float']:.1f}`" if pd.notna(r['rating_float']) else "")
        )
        story = r.get(STORY_COL)
        if isinstance(story, str) and story.strip():
            with st.expander("Storyline", expanded=True):
                st.write(story)

        url = r.get("URL")
        if isinstance(url, str) and url.startswith("http"):
            st.link_button("Open on IMDb", url)

        if show_explain:
            idx = r.name
            terms = explain_overlap(q, X[idx], vec, topn=8)
            if terms:
                st.caption("Why it matched:")
                st.write(" · ".join(f"`{t}`" for t in terms))
        st.divider()

    # download
    csv_buf = io.StringIO()
    res[[TITLE_COL, "similarity", "rating_float", STORY_COL, "URL"]].rename(
        columns={"rating_float": "Rating"}
    ).to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download results (CSV)", data=csv_buf.getvalue(),
                       file_name=f"similar_to_{pick.replace(' ','_')}.csv", mime="text/csv")

# ---------- Route ----------
if mode == "Type a storyline":
    run_text_mode()
else:
    run_item_mode()

with st.expander("About this app"):
    st.markdown(
        "- TF-IDF on 2024 storylines (1–2 grams, English stopwords)\n"
        "- Cosine similarity = dot product on normalized vectors\n"
        "- Optional rating filter & similarity floor\n"
        "- ‘Why it matched’ shows highest-weight overlapping terms\n"
    )
