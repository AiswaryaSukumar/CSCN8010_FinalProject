# -------------------------------------------------------------
# HuggingFace Models for Intent Classification & Crisis Detection
# -------------------------------------------------------------

import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logger = logging.getLogger(__name__)

# ---------------------------
# GLOBAL STATE (Lazy Loading)
# ---------------------------

HF_INTENT_PIPE = None
HF_EMOTION_PIPE = None


# =============================================================
# LOAD INTENT CLASSIFIER (CPU Friendly)
# =============================================================
def load_intent_model():
    global HF_INTENT_PIPE
    if HF_INTENT_PIPE is not None:
        return HF_INTENT_PIPE
    
    try:
        model_name = "transformersbook/distilbert-base-uncased-finetuned-intent"
        HF_INTENT_PIPE = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=-1
        )
        logger.info("HF Intent Model Loaded Successfully")
    except Exception as e:
        logger.error(f"Failed to load HF intent model: {e}")
        HF_INTENT_PIPE = None
    
    return HF_INTENT_PIPE


# =============================================================
# LOAD EMOTION MODEL (GoEmotions)
# =============================================================
def load_emotion_model():
    global HF_EMOTION_PIPE
    if HF_EMOTION_PIPE is not None:
        return HF_EMOTION_PIPE
    
    try:
        model_name = "SamLowe/roberta-base-go-emotions"
        HF_EMOTION_PIPE = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            top_k=None,
            device=-1
        )
        logger.info("HF Emotion Model Loaded Successfully")
    except Exception as e:
        logger.error(f"Failed to load HF emotion model: {e}")
        HF_EMOTION_PIPE = None
        
    return HF_EMOTION_PIPE


# =============================================================
# HF INTENT PREDICTION
# =============================================================
def hf_predict_intent(text: str):
    pipe = load_intent_model()
    if pipe is None:
        return None
    
    try:
        result = pipe(text)[0]  # {"label": "...", "score": float}
        return {"label": result["label"].lower(), "score": float(result["score"])}
    except Exception as e:
        logger.error(f"HF intent prediction failed: {e}")
        return None


# =============================================================
# HF EMOTION DETECTION
# =============================================================
def hf_detect_emotion(text: str):
    pipe = load_emotion_model()
    if pipe is None:
        return None
    
    try:
        results = pipe(text)
        # results = [{"label":"sadness","score":0.92}, ...]
        top = sorted(results, key=lambda x: x["score"], reverse=True)[0]
        return {"label": top["label"].lower(), "score": float(top["score"])}
    except Exception as e:
        logger.error(f"HF emotion detection failed: {e}")
        return None