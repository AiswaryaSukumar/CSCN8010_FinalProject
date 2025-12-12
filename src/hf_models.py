# -------------------------------------------------------------
# HuggingFace Models for Intent Classification & Crisis Detection
# -------------------------------------------------------------

import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logger = logging.getLogger(__name__)


# GLOBAL STATE (Lazy Loading


HF_EMOTION_PIPE = None


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

# HF EMOTION DETECTION

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