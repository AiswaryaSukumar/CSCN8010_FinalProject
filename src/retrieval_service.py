"""
retrieval_service.py

TF-IDF based retrieval + intent classification service for the Student Affairs chatbot.

Includes:
- LLM-based intent classifier (student_affairs / small_talk / out_of_scope / serious_issue)
- similarity threshold handling for TF-IDF retrieval
- basic invalid / out-of-scope handling
- keyword & intent-based serious/critical query escalation
"""

import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import joblib

# Import your intent classifier (OpenAI-based)
from intent_classifier import classify_intent

# ---------------- CONFIG ----------------

# project root is one level above src/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "student_affairs_knowledge_base.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(MODELS_DIR, "kb_tfidf_matrix.npz")

kb_df = None
tfidf_vectorizer = None
kb_tfidf_matrix = None

# cosine similarity threshold – tune if you want
MIN_SIMILARITY_DIRECT = 0.35

# keyword-based serious/sensitive detection (backup safety net)
SENSITIVE_KEYWORDS = [
    "suicide", "kill myself", "self harm", "self-harm",
    "want to die", "end my life",
    "assault", "sexual assault", "rape", "harassment",
    "violence", "abuse", "bullying",
    "mental health", "depressed", "depression", "anxiety",
    "panic attack", "panic attacks", "crisis", "emergency",
]
EMOTION_WORDS = [
    "stressed", "overwhelmed", "hopeless", "worthless",
    "burned out", "burnt out", "lonely", "scared",
    "depressed", "anxious",
]

ADVISOR_BOOKING_URL = "https://successportal.conestogac.on.ca/"

# quick heuristic for greetings (fast path, no LLM call needed)
SMALL_TALK = {"hi", "hello", "hey", "heyy", "heyyy", "yo"}


# ---------------- LOADERS ----------------

def load_resources():
    """
    Load KB + TF-IDF vectorizer + TF-IDF matrix from disk.
    Call this once at app startup (app.py already does this).
    """
    global kb_df, tfidf_vectorizer, kb_tfidf_matrix

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"KB CSV not found at {DATA_PATH}")
    kb_df = pd.read_csv(DATA_PATH)
    kb_df["question"] = kb_df["question"].fillna("").astype(str).str.strip()
    kb_df["answer"] = kb_df["answer"].fillna("").astype(str).str.strip()

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer file not found at {VECTORIZER_PATH}")
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

    if not os.path.exists(MATRIX_PATH):
        raise FileNotFoundError(f"TF-IDF matrix file not found at {MATRIX_PATH}")
    kb_tfidf_matrix = sparse.load_npz(MATRIX_PATH)

    print("Resources loaded:")
    print("  KB shape        :", kb_df.shape)
    print("  TF-IDF matrix   :", kb_tfidf_matrix.shape)
    print("  Vocab size      :", len(tfidf_vectorizer.get_feature_names_out()))


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


# ---------------- HELPERS ----------------

def _is_small_talk_fast(q: str) -> bool:
    """Fast small-talk check before calling the LLM intent classifier."""
    return q.lower().strip() in SMALL_TALK


def _is_query_too_short_or_garbage(q: str) -> bool:
    """
    Very simple check for nonsense / extremely short queries.

    We *allow* single-word queries like 'wifi', 'depression', 'fees'
    (they are common and should be handled), so we only reject
    tiny / random strings.
    """
    q_clean = "".join(ch for ch in q.lower() if ch.isalpha() or ch.isspace()).strip()
    # reject if fewer than 3 characters AFTER cleaning
    return len(q_clean) < 3


def _is_sensitive_keyword_based(q: str) -> bool:
    """
    Keyword-based sensitive / personal query detection.
    Used as a backup on top of the intent classifier.
    """
    q_low = q.lower()

    # 1) direct strong keywords
    if any(kw in q_low for kw in SENSITIVE_KEYWORDS):
        return True

    # 2) softer emotional language combined with "i feel / i am feeling"
    if ("i feel" in q_low or "i am feeling" in q_low or "i'm feeling" in q_low):
        if any(word in q_low for word in EMOTION_WORDS):
            return True

    return False


def _intent_branch(query: str) -> Dict[str, str]:
    """
    Use the LLM-based intent classifier to decide high-level handling.

    Returns a tiny dict with:
        {"intent": ...}
    (score is always 1.0 in the current implementation, so we ignore it.)
    """
    out = classify_intent(query)
    return {"intent": out["label"]}


# ---------------- MAIN API ----------------

def answer_query(
    query: str,
    k: int = 3,
    min_similarity: float = MIN_SIMILARITY_DIRECT,
) -> dict:
    """
    Main entry point used by Streamlit.

    Returns a dict with:
        mode: "direct" | "low_confidence" | "escalate" | "invalid"
        answer: text to show to the user
        matched_question: (optional) best FAQ question
        similarity: best similarity score (or 0.0)
        source_type, source_url: where the answer came from
    """
    query = query.strip()

    # --- empty query --------------------------------------------------------
    if not query:
        return {
            "mode": "invalid",
            "query": query,
            "answer": "Please enter a question so I can try to help.",
            "matched_question": "",
            "similarity": 0.0,
            "source_type": "",
            "source_url": "",
        }

    # --- very fast small-talk shortcut (no LLM intent call) -----------------
    if _is_small_talk_fast(query):
        return {
            "mode": "invalid",  # or a separate 'small_talk' mode if you ever want
            "query": query,
            "answer": (
                "Hi! I’m your Student Success Assistant. "
                "You can ask me questions about orientation, career services, "
                "student rights & responsibilities, and other Conestoga support."
            ),
            "matched_question": "",
            "similarity": 0.0,
            "source_type": "",
            "source_url": "",
        }

    # --- obvious garbage ----------------------------------------------------
    if _is_query_too_short_or_garbage(query):
        return {
            "mode": "invalid",
            "query": query,
            "answer": (
                "I’m not sure how to interpret that. "
                "Please try asking a full question related to Conestoga student services, "
                "orientation, career centre, or student rights."
            ),
            "matched_question": "",
            "similarity": 0.0,
            "source_type": "",
            "source_url": "",
        }

    # --- intent classification (OpenAI-based) --------------------------------
    intent_info = _intent_branch(query)
    intent_label = intent_info["intent"]  # one of: student_affairs, small_talk, out_of_scope, serious_issue

    # 1) serious_issue OR keyword-based safety net → escalate
    if intent_label == "serious_issue" or _is_sensitive_keyword_based(query):
        escalation_answer = (
            "Your question sounds important and personal. For your safety and privacy, "
            "it’s best to speak directly with a Student Success Advisor or support service.\n\n"
            f"Please book an appointment through the Student Success Portal: {ADVISOR_BOOKING_URL}"
        )
        return {
            "mode": "escalate",
            "query": query,
            "answer": escalation_answer,
            "matched_question": "",
            "similarity": 0.0,
            "source_type": "escalation",
            "source_url": ADVISOR_BOOKING_URL,
        }

    # 2) out_of_scope → explain bot scope, no RAG / no FAQ search
    if intent_label == "out_of_scope":
        return {
            "mode": "invalid",
            "query": query,
            "answer": (
                "I’m designed to help with Conestoga student services: orientation, "
                "student success, career services, and student rights & responsibilities. "
                "Your question seems outside this scope, so I may not be able to answer it accurately."
            ),
            "matched_question": "",
            "similarity": 0.0,
            "source_type": "",
            "source_url": "",
        }

    # 3) small_talk (caught by the LLM but not by the fast check) ------------
    if intent_label == "small_talk":
        return {
            "mode": "invalid",
            "query": query,
            "answer": (
                "Hello! I’m here to help with Student Affairs questions. "
                "Try asking about orientation, career centre, student rights, or campus support."
            ),
            "matched_question": "",
            "similarity": 0.0,
            "source_type": "",
            "source_url": "",
        }

    # 4) student_affairs (or anything not caught above) → normal TF-IDF retrieval
    # ---------------------------------------------------------------------------
    results = retrieve_top_k(query, k=k)
    best = results.iloc[0]
    sim = float(best["similarity"])

    # Low-confidence match
    if sim < min_similarity:
        fallback_answer = (
            "I’m not confident I have a strong match in the current Student Affairs FAQs for your question.\n\n"
            "You can try rephrasing your question, or contact a Student Success Advisor directly "
            f"through the portal: {ADVISOR_BOOKING_URL}"
        )
        return {
            "mode": "low_confidence",
            "query": query,
            "answer": fallback_answer,
            "matched_question": best["question"],
            "similarity": sim,
            "source_type": best.get("source_type", ""),
            "source_url": best.get("source_url", ""),
        }

    # High-confidence FAQ answer
    return {
        "mode": "direct",
        "query": query,
        "answer": best["answer"],
        "matched_question": best["question"],
        "similarity": sim,
        "source_type": best.get("source_type", ""),
        "source_url": best.get("source_url", ""),
    }


if __name__ == "__main__":
    # quick manual tests (python src/retrieval_service.py)
    load_resources()
    from pprint import pprint

    pprint(answer_query("hi"))
    pprint(answer_query("How do I access wifi on campus?"))
    pprint(answer_query("I feel very depressed"))
    pprint(answer_query("tell me a joke"))
