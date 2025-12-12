# src/model/build_tfidf_index.py

import os
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from src.dataCleaningProcessing.tokenizer_utils import simple_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "student_affairs_knowledge_base.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# NEW NAMES (v2)
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer_v2.pkl"
MATRIX_PATH = MODELS_DIR / "kb_tfidf_matrix_v2.npz"

os.makedirs(MODELS_DIR, exist_ok=True)

print("PROJECT_ROOT :", PROJECT_ROOT)
print("DATA_PATH    :", DATA_PATH)
print("MODELS_DIR   :", MODELS_DIR)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"KB CSV not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("KB shape:", df.shape)

    if "question" not in df.columns or "answer" not in df.columns:
        raise ValueError("CSV must have 'question' and 'answer' columns")

    docs = (df["question"].fillna("") + " " + df["answer"].fillna("")).tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
        max_df=0.8,
        min_df=1,
    )

    print(f"Fitting TF-IDF on {len(docs)} documents...")
    tfidf_matrix = vectorizer.fit_transform(docs)
    print("TF-IDF matrix shape:", tfidf_matrix.shape)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Saved vectorizer to:", VECTORIZER_PATH)

    sparse.save_npz(MATRIX_PATH, tfidf_matrix)
    print("Saved TF-IDF matrix to:", MATRIX_PATH)


if __name__ == "__main__":
    main()
