#!/usr/bin/env python3
# build_index.py
from pathlib import Path
import re, json, pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

# Point this at your CSV (yours is imdb_2024_list_all.csv)
DATA_CSV = Path("imdb_2024_list_all.csv")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

TITLE_COL = "Movie Name"
STORY_COL = "Storyline (list blurb)"   # <== your dataset’s storyline column

def clean_story(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u200b", "").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    df = pd.read_csv(DATA_CSV)
    # Keep useful columns, de-dup movies by IMDb ID if present
    keep_cols = [c for c in [TITLE_COL, "IMDb ID", "URL", "Rating", "Duration", STORY_COL] if c in df.columns]
    df = df[keep_cols].drop_duplicates(subset=["IMDb ID"], keep="first")

    # Clean story text and drop very short/missing
    df[STORY_COL] = df[STORY_COL].astype(str).map(clean_story)
    df = df[df[STORY_COL].str.len() >= 20].reset_index(drop=True)

    # TF-IDF on the storyline (bigrams help a lot for plots)
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=200_000,
        dtype=np.float32,
    )
    X = vec.fit_transform(df[STORY_COL].values)

    # Persist artifacts
    sparse.save_npz(ART_DIR / "tfidf_2024.npz", X)
    with open(ART_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)

    # Save metadata for the app
    df.to_csv(ART_DIR / "movies_meta.csv", index=False)
    with open(ART_DIR / "config.json", "w") as f:
        json.dump({"title_col": TITLE_COL, "story_col": STORY_COL}, f, indent=2)

    print(f"✅ Indexed {len(df):,} movies | TF-IDF shape: {X.shape} | Artifacts -> {ART_DIR}/")

if __name__ == "__main__":
    main()
