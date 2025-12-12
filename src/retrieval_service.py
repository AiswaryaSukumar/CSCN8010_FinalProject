from pathlib import Path
import sys
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import joblib
import pandas as pd
from src.dataCleaningProcessing.tokenizer_utils import simple_tokenizer  # noqa: F401
from src.intent_classifier import classify_intent
from src.hf_models import hf_detect_emotion
from src.llm_service import (
    generate_llm_answer,
    generate_supportive_response,
)

# ======================================================
# GLOBAL PATHS & LOADING
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer_v2.pkl"
MATRIX_PATH = MODELS_DIR / "kb_tfidf_matrix_v2.npz"
KB_PATH = DATA_DIR / "student_affairs_knowledge_base.csv"

print("PROJECT_ROOT:", PROJECT_ROOT)
print("VECTORIZER_PATH:", VECTORIZER_PATH)
print("MATRIX_PATH:", MATRIX_PATH)
print("KB_PATH:", KB_PATH)

if not VECTORIZER_PATH.exists():
    raise FileNotFoundError(f"TF-IDF vectorizer missing: {VECTORIZER_PATH}")

if not MATRIX_PATH.exists():
    raise FileNotFoundError(f"TF-IDF matrix missing: {MATRIX_PATH}")

if not KB_PATH.exists():
    raise FileNotFoundError(f"Knowledge base CSV missing: {KB_PATH}")

# GLOBALS (used in pipeline & answer_query)
vectorizer = None
tfidf_matrix = None
kb = None

# ======================================================
# 1) REQUIRED BY STREAMLIT — load_resources() FIX
# ======================================================
def load_resources():
    """
    Ensures vectorizer, TF-IDF matrix, and KB dataframe are loaded only once.
    Streamlit imports this; MUST EXIST.
    """
    global vectorizer, tfidf_matrix, kb

    if vectorizer is None:
        vectorizer = joblib.load(VECTORIZER_PATH)

    if tfidf_matrix is None:
        tfidf_matrix = sparse.load_npz(MATRIX_PATH)

    if kb is None:
        kb = pd.read_csv(KB_PATH)

    print("✔ load_resources(): TF-IDF + KB loaded")
    return vectorizer, tfidf_matrix, kb


# Load immediately for use inside this module
load_resources()


# ======================================================
# SAFETY LOGIC
# ======================================================

EXPLICIT_SELF_HARM = [
    "kill",
    "end my life",
    "i want to die",
    "suicide",
    "hurt myself",
    "self harm",
    "self-harm",
]

def contains_explicit_self_harm(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in EXPLICIT_SELF_HARM)


CRISIS_EMOTIONS = {"sadness", "fear", "anxiety", "despair"}
HIGH_SEVERITY_THRESHOLD = 0.80


def is_high_severity_serious(query: str) -> bool:
    emo = hf_detect_emotion(query)
    if emo is None:
        return False

    label = emo["label"].lower()
    score = emo["score"]

    logging.info(f"[Emotion] label={label}, score={score:.2f}")

    return (label in CRISIS_EMOTIONS) and (score >= HIGH_SEVERITY_THRESHOLD)


# ======================================================
# RETRIEVAL
# ======================================================

def retrieve_best_match(query: str):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()

    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    row = kb.iloc[best_idx]

    return {
        "matched_question": row["question"],
        "answer": row["answer"],
        "source_type": row.get("source_type", "faq"),
        "similarity": best_sim,
    }



# ======================================================
# ESCALATION HELPERS
# ======================================================

ADVISOR_URL = "https://www.conestogac.on.ca/student-success/advising"

def escalate_response(reason: str):
    return {
        "mode": "escalate",
        "answer": (
            "It sounds like you're dealing with something very difficult. "
            "I want to make sure you get the right support.\n\n"
            f"Please reach out to a Student Success Advisor: {ADVISOR_URL}\n\n"
            "If you ever feel in immediate danger, contact emergency services."
        ),
        "reason": reason,
    }


# ======================================================
# MAIN PIPELINE
# ======================================================

def answer_query(query: str):
    """
    1) Intent classifier
    2) Safety checks
    3) TF-IDF retrieval
    4) LLM fallback
    """
    logging.info(f"[Query] {query}")

    intent = classify_intent(query)
    intent_label = intent["label"]
    intent_score = intent["score"]
    print(">>> INTENT PREDICTED:", intent_label,intent_score)


    logging.info(f"[Intent] {intent_label}")

    # OUT OF SCOPE
    if intent_label == "out_of_scope":
        return {
            "mode": "out_of_scope",
            "answer": (
                "I'm here to help with student and college-related questions. "
                "Could you try rephrasing or ask something about Conestoga services?"
            ),
        }

    # SMALL TALK
    if intent_label == "small_talk" and intent.get("score", 1.0) > 0.95:
        return {
            "mode": "small_talk",
            "answer": "Hi there! I'm your Student Success Assistant. How can I help today?",
        }

    # SERIOUS ISSUE
    if intent_label == "serious_issue":
        if contains_explicit_self_harm(query):
            return escalate_response("explicit_self_harm_detected")

        if is_high_severity_serious(query):
            return escalate_response("high_severity_emotion")

        supportive = generate_supportive_response(query)
        return {
            "mode": "serious_support",
            "answer": supportive,
            "similarity": 0.0,
            "source_type": "serious_issue_support",
            "source_url": ADVISOR_URL,
        }

    # NORMAL → TF-IDF RETRIEVAL
    match = retrieve_best_match(query)
    logging.info(f"[TF-IDF] similarity={match['similarity']:.3f}")

    # LLM fallback if similarity too low
    if match["similarity"] < 0.50:
        logging.info("[Fallback] Using LLM because similarity < 0.50")
        faq_rows = retrieve_top_k(query, k=5)
        llm_answer = generate_llm_answer(query, faq_rows)

        return {
            "mode": "llm_fallback",
            "answer": llm_answer,
            "similarity": match["similarity"],
            "matched_question": match["matched_question"],
            "source_type": "llm_fallback",
        }

    # -------------------------
    # MISSING BLOCK (now added)
    # -------------------------
    # Normal KB match → return the retrieved answer
    return {
        "mode": "kb_match",
        "answer": match["answer"],
        "similarity": match["similarity"],
        "matched_question": match["matched_question"],
        "source_type": "knowledge_base",
        "source_url": match.get("source_url"),
    }

def retrieve_top_k(query: str, k: int = 5) -> pd.DataFrame:
    """Return the top-K most similar FAQ rows using TF-IDF cosine similarity."""
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:k]
    return kb.iloc[top_indices].copy()
