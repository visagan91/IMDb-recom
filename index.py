import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz
import joblib

DEF_CSV = "data/imdb_2024_movies.csv"

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["storyline"] = df["storyline"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    # Basic sanity filter
    df = df[df["storyline"].str.len() > 30]
    df = df.drop_duplicates(subset=["imdb_id"]).reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEF_CSV, help="Path to CSV with columns [title, storyline, url, imdb_id]")
    parser.add_argument("--outdir", default="data", help="Where to save vectorizers and matrices")
    parser.add_argument("--min_df", type=int, default=2, help="Min doc freq for vectorizers")
    parser.add_argument("--max_features", type=int, default=100000, help="Max features for vectorizers")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = clean_df(df)
    print(f"Loaded {len(df)} rows from {args.csv}")

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=args.min_df, max_features=args.max_features, strip_accents="unicode")
    X_tfidf = tfidf.fit_transform(df["storyline"])
    joblib.dump(tfidf, os.path.join(args.outdir, "tfidf_vectorizer.pkl"))
    save_npz(os.path.join(args.outdir, "tfidf_matrix.npz"), X_tfidf)
    print(f"Saved TF-IDF: {X_tfidf.shape}")

    # Count
    count = CountVectorizer(stop_words="english", ngram_range=(1,2), min_df=args.min_df, max_features=args.max_features, strip_accents="unicode")
    X_count = count.fit_transform(df["storyline"])
    joblib.dump(count, os.path.join(args.outdir, "count_vectorizer.pkl"))
    save_npz(os.path.join(args.outdir, "count_matrix.npz"), X_count)
    print(f"Saved Count: {X_count.shape}")

    # Save the cleaned dataframe to ensure app uses same ordering
    df.to_csv(os.path.join(args.outdir, "clean_movies.csv"), index=False)

if __name__ == "__main__":
    main()