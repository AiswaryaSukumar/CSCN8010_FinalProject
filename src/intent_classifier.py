# intent_classifier.py
# HYBRID INTENT CLASSIFIER (HF + OpenAI)

import logging
from functools import lru_cache
from openai import OpenAI
from src.hf_models import hf_predict_intent
import os
from dotenv import load_dotenv

load_dotenv()  # looks for .env in current or parent dirs
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
logger = logging.getLogger(__name__)


INTENT_LABELS = [
    "student_affairs",
    "small_talk",
    "out_of_scope",
    "serious_issue",
]

# HF → Custom Intent Mapping
HF_TO_LOCAL = {
    "greeting": "small_talk",
    "goodbye": "small_talk",
    "affirm": "small_talk",
    "deny": "small_talk",
    "question": "student_affairs",
    "help": "student_affairs",
    "other": "out_of_scope"
}


# ============================================================
# Hybrid Intent Classifier
# ============================================================

@lru_cache(maxsize=100)
def classify_intent(text: str):
    # ----------------------------------------------------
    # 1) TRY HUGGINGFACE FIRST (OFFLINE + FAST)
    # ----------------------------------------------------
    try:
        hf_out = hf_predict_intent(text)
        if hf_out is not None and hf_out["score"] >= 0.85:
            mapped = HF_TO_LOCAL.get(hf_out["label"], "out_of_scope")
            logger.info(f"[HF Intent] {text} → {mapped} ({hf_out['score']:.2f})")
            return {"label": mapped, "score": hf_out["score"]}
    except Exception as e:
        logger.error(f"HF intent error: {e}")

    # ----------------------------------------------------
    # 2) FALLBACK TO OPENAI LLM (Gold Standard)
    # ----------------------------------------------------
    logger.info("[LLM Intent Fallback Triggered]")

    system_msg = """
You are an intent classifier for a student affairs assistant.
Decide ONE label only:
student_affairs, small_talk, out_of_scope, serious_issue.
Respond ONLY with the label.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        label = resp.choices[0].message.content.strip().lower()
        if label not in INTENT_LABELS:
            label = "out_of_scope"
    except Exception as e:
        logger.error(f"LLM intent classification failed: {e}")
        label = "out_of_scope"

    return {"label": label, "score": 1.0}