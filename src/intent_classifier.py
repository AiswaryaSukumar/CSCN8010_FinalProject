"""
Intent classifier using our own deep learning model + HF embeddings.

- Encodes queries with SentenceTransformer (encode_texts)
- Uses a small MLP trained on 4 labels:
    student_affairs, small_talk, out_of_scope, serious_issue
- Returns: {"label": <str>, "score": <float dummy=1.0>}
"""

import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# IMPORTANT: this is the HF encoder we used in training
# If your module name is different, change this import.
from src.embedding_hf import encode_texts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths (project-root relative)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "intent_classifier_best.pt"
ENCODER_PATH = MODELS_DIR / "intent_label_encoder.pkl"

INTENT_LABELS = [
    "student_affairs",
    "small_talk",
    "out_of_scope",
    "serious_issue",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_label_encoder = None


# ---------------------------------------------------------------------
# MLP model definition – must match what you used during training
# ---------------------------------------------------------------------
class IntentClassifierMLP(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, output_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------
# Lazy loading of model + label encoder
# ---------------------------------------------------------------------
def _load_model_and_encoder():
    global _model, _label_encoder

    if _model is not None and _label_encoder is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Intent model not found at: {MODEL_PATH}")

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found at: {ENCODER_PATH}")

    # Load label encoder
    with open(ENCODER_PATH, "rb") as f:
        _label_encoder = pickle.load(f)

    # Build model and load weights
    _model = IntentClassifierMLP(output_dim=len(INTENT_LABELS))
    state = torch.load(MODEL_PATH, map_location=device)
    _model.load_state_dict(state)
    _model.to(device)
    _model.eval()

    logger.info("Intent classifier model + label encoder loaded successfully.")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
@lru_cache(maxsize=256)
def classify_intent(text: str):
    """
    Classify a single user query into one of:
    student_affairs, small_talk, out_of_scope, serious_issue
    """
    _load_model_and_encoder()

    # 1) Encode text with HF encoder -> numpy array shape (1, 384)
    emb = encode_texts([text])
    tensor = torch.tensor(emb, dtype=torch.float32, device=device)

    # 2) Forward pass
    with torch.no_grad():
        logits = _model(tensor)
        pred_idx = torch.argmax(logits, dim=1).item()

    # 3) Map back to string label via label encoder
    label = _label_encoder.inverse_transform([pred_idx])[0]

    # Safety: if label is something unexpected, fall back to out_of_scope
    if label not in INTENT_LABELS:
        logger.warning(f"Unknown intent label from encoder: {label}. Falling back to 'out_of_scope'.")
        label = "out_of_scope"

    return {"label": label, "score": 1.0}


# ---------------------------------------------------------------------
# Simple manual test:  python -m src.intent_classifier
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("MODEL_PATH  :", MODEL_PATH)
    print("ENCODER_PATH:", ENCODER_PATH)

    _load_model_and_encoder()
    print("Type a query (or 'quit'):")

    while True:
        q = input("> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        result = classify_intent(q)
        print(result)
