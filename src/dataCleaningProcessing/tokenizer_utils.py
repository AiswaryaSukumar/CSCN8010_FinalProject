# src/dataCleaningProcessing/tokenizer_utils.py
import re
from typing import List

def simple_tokenizer(text: str) -> List[str]:
    """
    Very simple tokenizer:
    - lowercases
    - keeps only alphanumeric word tokens
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    # split on word boundaries (letters/numbers)
    tokens = re.findall(r"\b\w+\b", text)
    return tokens
