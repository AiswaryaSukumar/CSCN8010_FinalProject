"""
retrieval_service.py

Provides TF-IDF based FAQ retrieval for the Student Affairs chatbot.

- Loads:
    * student_affairs_knowledge_base.csv
    * tfidf_vectorizer.pkl
    * kb_tfidf_matrix.npz

- Exposes:
    * load_resources()
    * retrieve_top_k(query, k=5)
    * answer_query(query, k=3)
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import joblib

# ---------------- CONFIG ----------------

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH   = os.path.join(BASE_DIR, "data", "student_affairs_knowledge_base.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH     = os.path.join(MODELS_DIR, "kb_tfidf_matrix.npz")

kb_df = None
tfidf_vectorizer = None
kb_tfidf_matrix = None


def load_resources():
    """
    Load KB + TF-IDF vectorizer + TF-IDF matrix from disk.
    Call this once at app startup.
    """
    global kb_df, tfidf_vectorizer, kb_tfidf_matrix

    # Load KB
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"KB CSV not found at {DATA_PATH}")
    kb_df = pd.read_csv(DATA_PATH)
    kb_df["question"] = kb_df["question"].fillna("").astype(str).str.strip()
    kb_df["answer"]   = kb_df["answer"].fillna("").astype(str).str.strip()

    # Load vectorizer
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer file not found at {VECTORIZER_PATH}")
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

    # Load sparse matrix
    if not os.path.exists(MATRIX_PATH):
        raise FileNotFoundError(f"TF-IDF matrix file not found at {MATRIX_PATH}")
    kb_tfidf_matrix = sparse.load_npz(MATRIX_PATH)

    print("Resources loaded:")
    print("  KB shape        :", kb_df.shape)
    print("  TF-IDF matrix   :", kb_tfidf_matrix.shape)
    print("  Vectorizer vocab:", len(tfidf_vectorizer.get_feature_names_out()))


def retrieve_top_k(query: str, k: int = 5) -> pd.DataFrame:
    """
    Retrieve top-k most similar KB entries for a text query.
    """
    if kb_df is None or tfidf_vectorizer is None or kb_tfidf_matrix is None:
        raise RuntimeError("Resources not loaded. Call load_resources() first.")

    q_vec = tfidf_vectorizer.transform([query])
    sims = cosine_similarity(q_vec, kb_tfidf_matrix)[0]

    top_idx = np.argsort(sims)[::-1][:k]
    results = kb_df.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]

    return results


def answer_query(query: str, k: int = 3) -> dict:
    """
    Return the best answer + some metadata for the query.
    Designed to be used by Streamlit / API / CLI.

    Returns:
        {
          "query": ...,
          "matched_question": ...,
          "answer": ...,
          "similarity": float,
          "source_type": ...,
          "source_url": ...
        }
    """
    results = retrieve_top_k(query, k=k)
    best = results.iloc[0]

    return {
        "query": query,
        "matched_question": best["question"],
        "answer": best["answer"],
        "similarity": float(best["similarity"]),
        "source_type": best.get("source_type", ""),
        "source_url": best.get("source_url", "")
    }


if __name__ == "__main__":
    # Small manual test if you run: python retrieval_service.py
    load_resources()
    test = answer_query("How do I access campus Wi-Fi?")
    from pprint import pprint
    pprint(test)
