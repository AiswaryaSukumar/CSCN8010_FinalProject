# embedding_hf.py
from sentence_transformers import SentenceTransformer

# Load once, reuse everywhere
_hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def encode_texts(texts):
    """
    Convert list of strings to embeddings (numpy array)
    """
    return _hf_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
