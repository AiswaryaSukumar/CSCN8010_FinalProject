# --------------------------------------------------------------------
# retrieval_service.py
# Patched Version – HF Emotion, Hybrid Intent, Logging, Safe Routing
# --------------------------------------------------------------------

import os
import logging
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from src.llm_service import generate_llm_answer
from src.intent_classifier import classify_intent
from src.hf_models import hf_detect_emotion

# Logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_affairs_knowledge_base.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(MODELS_DIR, "kb_tfidf_matrix.npz")

ADVISOR_BOOKING_URL = "https://successportal.conestogac.on.ca/"

# Threshold
MIN_SIMILARITY_DIRECT = 0.35

# Small talk
SMALL_TALK = {"hi", "hello", "hey", "yo", "heyy", "good morning", "good evening"}

# Sensitive terms (backup safety)
SENSITIVE_KEYWORDS = [
    "suicide", "kill myself", "self harm", "self-harm",
    "end my life", "harassment", "sexual assault", "abuse",
    "violence", "panic attack", "depressed", "hopeless",
    "anxious", "scared", "overwhelmed"
]

EMOTION_CRISIS_LABELS = {"sadness", "fear", "anxiety", "despair"}

kb_df = None
tfidf_vectorizer = None
kb_tfidf_matrix = None


# =============================================================
# LOAD DATA + MODELS
# =============================================================
def load_resources():
    global kb_df, tfidf_vectorizer, kb_tfidf_matrix

    try:
        kb_df = pd.read_csv(DATA_PATH)
        kb_df["question"] = kb_df["question"].fillna("").astype(str)
        kb_df["answer"] = kb_df["answer"].fillna("").astype(str)

        tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
        kb_tfidf_matrix = sparse.load_npz(MATRIX_PATH)

        logger.info("Resources loaded successfully")

    except Exception as e:
        logger.error(f"Failed loading resources: {e}")
        raise


# =============================================================
# Helpful Checks
# =============================================================
def _is_small_talk(query: str):
    return query.lower().strip() in SMALL_TALK


def _is_too_short(query: str):
    cleaned = "".join(c for c in query if c.isalpha()).strip()
    return len(cleaned) < 3


def _keyword_crisis(query: str):
    qlow = query.lower()
    return any(kw in qlow for kw in SENSITIVE_KEYWORDS)


# =============================================================
# HuggingFace Emotion + Hybrid Crisis Detection
# =============================================================
def is_crisis_query(query: str, intent_label: str):
    query_low = query.lower()

    # 1) Intent classifier flagged crisis
    if intent_label == "serious_issue":
        logger.warning(f"[CRISIS DETECTED: LLM Intent] {query}")
        return True

    # 2) Keyword safety net
    if _keyword_crisis(query):
        logger.warning(f"[CRISIS DETECTED: Keyword] {query}")
        return True

    # 3) HF Emotion Model
    try:
        emo = hf_detect_emotion(query)
        if emo:
            emo_label = emo["label"]
            emo_score = emo["score"]
            if emo_label in EMOTION_CRISIS_LABELS and emo_score >= 0.70:
                logger.warning(f"[CRISIS DETECTED: Emotion] {query} → {emo_label} {emo_score}")
                return True
    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")

    return False


# =============================================================
# TF-IDF RETRIEVAL
# =============================================================
def retrieve_top_k(query: str, k=5):
    try:
        q_vec = tfidf_vectorizer.transform([query])
        sims = cosine_similarity(q_vec, kb_tfidf_matrix)[0]

        idx = np.argsort(sims)[::-1][:k]
        results = kb_df.iloc[idx].copy()
        results["similarity"] = sims[idx]
        return results

    except Exception as e:
        logger.error(f"TF-IDF retrieval failed: {e}")
        return pd.DataFrame([])


# =============================================================
# MAIN ANSWER FLOW
# =============================================================
def answer_query(query: str, k=3, min_similarity=MIN_SIMILARITY_DIRECT):

    query = query.strip()
    if not query:
        return _invalid("Please enter a question so I can help you.")

    # Quick small talk
    if _is_small_talk(query):
        return _invalid(
            "Hi! I’m your Student Success Assistant. "
            "Ask me about orientation, student support, career services, or campus resources."
        )

    # Garbage
    if _is_too_short(query):
        return _invalid(
            "I couldn't understand that. Try asking a more complete question related to Conestoga services."
        )

    # Intent classification
    intent_info = classify_intent(query)
    intent_label = intent_info["label"]
    logger.info(f"[Intent] {query} → {intent_label}")

    # CRISIS DETECTION
    if is_crisis_query(query, intent_label):
        return {
            "mode": "escalate",
            "query": query,
            "answer": (
                "Your message seems sensitive or urgent. "
                "Please speak directly with a Student Success Advisor or support services:\n"
                f"{ADVISOR_BOOKING_URL}"
            ),
            "matched_question": "",
            "similarity": 0.0,
            "source_type": "escalation",
            "source_url": ADVISOR_BOOKING_URL,
        }

    # Out-of-scope
    if intent_label == "out_of_scope":
        return _invalid(
            "I can help with Conestoga’s student services, orientation, academic support, "
            "career center, and student rights. This question seems outside that scope."
        )

    # Small talk through LLM
    if intent_label == "small_talk":
        return _invalid(
            "Hello! Ask me anything about Student Success, Orientation, Career Services, or Campus Support."
        )

    # Normal TF-IDF retrieval
    results = retrieve_top_k(query, k=k)
    if results.empty:
        return _invalid(
            "I couldn't match this question with any known FAQs. "
            f"Try rephrasing or contact a Student Success Advisor:\n{ADVISOR_BOOKING_URL}"
        )

    best = results.iloc[0]
    similarity = float(best["similarity"])
    logger.info(f"[Retrieval] Best Match Sim={similarity:.3f} | Q={best['question']}")

    # Low confidence → Let LLM synthesize based on context
    if similarity < min_similarity:
        answer = generate_llm_answer(query, results)
        return {
            "mode": "low_confidence",
            "query": query,
            "answer": answer,
            "matched_question": best["question"],
            "similarity": similarity,
            "source_type": best.get("source_type", ""),
            "source_url": best.get("source_url", ""),
        }

    # High confidence direct FAQ answer
    return {
        "mode": "direct",
        "query": query,
        "answer": best["answer"],
        "matched_question": best["question"],
        "similarity": similarity,
        "source_type": best.get("source_type", ""),
        "source_url": best.get("source_url", ""),
    }


# =============================================================
# Helper: Invalid Mode
# =============================================================
def _invalid(msg: str):
    return {
        "mode": "invalid",
        "query": "",
        "answer": msg,
        "matched_question": "",
        "similarity": 0.0,
        "source_type": "",
        "source_url": "",
    }