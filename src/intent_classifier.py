"""
intent_classifier.py

LLM-based intent classification for the Student Affairs chatbot.

We use the OpenAI Chat Completions API to classify each user query into one of:
    - student_affairs  -> in-domain, normal handling (use TF-IDF + LLM)
    - small_talk       -> greet the user / simple reply
    - out_of_scope     -> explain bot scope, no RAG/LLM
    - serious_issue    -> escalation: suggest contacting advisor/support

NOTE:
- This is heuristic and not a replacement for real crisis detection.
- We keep it simple and avoid heavy libraries like transformers/torch.
"""

from typing import Dict, List
from openai import OpenAI

client = OpenAI()

INTENT_LABELS: List[str] = [
    "student_affairs",
    "small_talk",
    "out_of_scope",
    "serious_issue",
]


def classify_intent(text: str) -> Dict:
    """
    Use an OpenAI model to classify the intent of the query.

    Returns a dict:
        {
          "label": top_label,
          "score": 1.0,
          "labels": [top_label],
          "scores": [1.0]
        }

    We keep the structure similar to the original transformers-based
    implementation so the rest of the code can use it easily.
    """

    system_msg = """
You are an intent classifier for a Student Affairs virtual assistant at Conestoga College.

Your job is to classify a student's question into exactly ONE of these labels:

- student_affairs  -> questions about orientation, student support, student success,
                      academic support, career services, campus life, student rights,
                      student responsibilities, fees, scholarships, registration, etc.
- small_talk       -> greetings or chit-chat like "hi", "hello", "how are you",
                      "who are you", "what can you do", etc.
- out_of_scope     -> questions unrelated to Conestoga student affairs or services,
                      for example: pure math, random trivia, cooking recipes,
                      general programming, world news, etc.
- serious_issue    -> anything that sounds like crisis, self-harm, suicide, harassment,
                      violence, abuse, or very serious personal distress.

Always respond with ONLY the label name, nothing else.
"""

    user_msg = f"""
Classify the following student message into exactly one label from:
{", ".join(INTENT_LABELS)}

Student message:
{text}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",  # or "gpt-4o-mini" if you prefer
        messages=[
            {"role": "system", "content": system_msg.strip()},
            {"role": "user", "content": user_msg.strip()},
        ],
        temperature=0.0,
        max_tokens=10,
    )

    raw_label = resp.choices[0].message.content.strip().lower()

    # Normalise and guard against weird outputs
    if raw_label not in INTENT_LABELS:
        label = "out_of_scope"
    else:
        label = raw_label

    return {
        "label": label,
        "score": 1.0,          # we don't get probabilities from here, so we just set 1.0
        "labels": [label],
        "scores": [1.0],
    }


def is_student_affairs(text: str, threshold: float = 0.5) -> bool:
    """
    Convenience helper: True if model thinks this is a student_affairs query.
    """
    out = classify_intent(text)
    return out["label"] == "student_affairs"


def is_serious_issue(text: str, threshold: float = 0.4) -> bool:
    """
    Convenience helper: True if model thinks this may be a serious_issue.
    """
    out = classify_intent(text)
    return out["label"] == "serious_issue"


if __name__ == "__main__":
    # quick manual tests
    tests = [
        "hi",
        "What is the difference between orientation and CSI welcome events?",
        "How can I book an appointment with a Student Success Advisor?",
        "How to cook biryani?",
        "I feel very depressed and I don't know what to do",
    ]
    from pprint import pprint

    for t in tests:
        print("TEXT:", t)
        pprint(classify_intent(t))
        print("-" * 60)
