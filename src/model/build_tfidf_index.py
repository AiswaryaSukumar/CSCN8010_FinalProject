"""
BUILD & SAVE TF-IDF INDEX FOR STUDENT AFFAIRS CHATBOT

WHAT THIS SCRIPT DOES (OFFLINE STEP):
1. Loads unified knowledge base CSV
2. Builds TF-IDF vectors using FAQ QUESTIONS ONLY
3. Saves vectorizer + matrix to disk

RUN THIS SCRIPT:
- ONCE initially
- AGAIN only if knowledge base changes

DO NOT run this from Streamlit or app.py
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


# ---------------------------------------------------------
# CONFIGURATION (absolute project paths)
# ---------------------------------------------------------

# src/model/build_tfidf_index.py → project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "student_affairs_knowledge_base.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(MODELS_DIR, "kb_tfidf_matrix.npz")


# ---------------------------------------------------------
# MAIN BUILD FUNCTION
# ---------------------------------------------------------

def build_tfidf_index():

    print("Project root :", BASE_DIR)
    print("KB path      :", DATA_PATH)
    print("Models dir   :", MODELS_DIR)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # -----------------------------------------------------
    # Load Knowledge Base
    # -----------------------------------------------------
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"KB CSV not found at {DATA_PATH}")

    kb_df = pd.read_csv(DATA_PATH)

    expected_cols = {"question", "answer", "source_url", "source_type"}
    missing = expected_cols - set(kb_df.columns)
    if missing:
        raise ValueError(f"KB missing columns: {missing}")

    print("KB loaded. Rows:", len(kb_df))

    # -----------------------------------------------------
    # Prepare text (QUESTION ONLY)
    # -----------------------------------------------------
    kb_df["question"] = kb_df["question"].fillna("").astype(str).str.strip()
    kb_df = kb_df[kb_df["question"] != ""]

    documents = kb_df["question"].tolist()

    print("Documents used for indexing:", len(documents))

    # -----------------------------------------------------
    # Build TF-IDF
    # -----------------------------------------------------
    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    print("TF-IDF built")
    print(" - Documents :", tfidf_matrix.shape[0])
    print(" - Vocabulary:", tfidf_matrix.shape[1])

    # -----------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------
    joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
    sparse.save_npz(MATRIX_PATH, tfidf_matrix)

    print("Saved vectorizer →", VECTORIZER_PATH)
    print("Saved matrix     →", MATRIX_PATH)
    print("\nDONE: Retrieval index persisted to disk.")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    build_tfidf_index()
