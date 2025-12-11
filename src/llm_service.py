# -------------------------------------------------------------------
# llm_service.py
# Patched Version – With Error Handling, Logging, Safe RAG Execution
# -------------------------------------------------------------------

import logging
from typing import List
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()  # looks for .env in current or parent dirs
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)

# Student Advisor Escalation Link
ADVISOR_BOOKING_LINK = "https://successportal.conestogac.on.ca"


# ================================================================
# Build Context from Top-K FAQs
# ================================================================
def build_faq_context(faq_rows: pd.DataFrame, max_chunks: int = 3) -> str:
    context_parts: List[str] = []

    try:
        for i, (_, row) in enumerate(faq_rows.head(max_chunks).iterrows(), start=1):
            q = str(row["question"]).strip()
            a = str(row["answer"]).strip()
            source = str(row.get("source_type", ""))
            context_parts.append(
                f"FAQ #{i}\n"
                f"source: {source}\n"
                f"question: {q}\n"
                f"answer: {a}\n"
            )
    except Exception as e:
        logger.error(f"FAQ context building failed: {e}")
        return "No FAQ context available."

    if not context_parts:
        return "No FAQ context available."

    return "\n\n".join(context_parts)


# ================================================================
# LLM Answer Generation with Strict RAG
# ================================================================
def generate_llm_answer(query: str, faq_rows: pd.DataFrame) -> str:

    context = build_faq_context(faq_rows, max_chunks=3)

    system_prompt = f"""
You are a virtual Student Success Assistant for Conestoga College.

Your responsibilities:
- Assist with orientation, student support, academic support, career services,
  student rights & responsibilities, and related campus service information.
- You MUST rely ONLY on the FAQ context provided.

STRICT RULES:
- Do NOT invent any policies, procedures, dates, or rules.
- If the FAQ context lacks enough information,
  say “I’m not fully sure based on the available information.”
  Then recommend booking a Student Success Advisor:
  {ADVISOR_BOOKING_LINK}
- Never give legal, immigration, visa, medical, or personal mental-health advice.
- If the query is outside Student Success scope,
  say that politely and direct them to book an advisor.
- If the student expresses mental health crisis or self-harm,
  tell them to seek immediate help from emergency services or campus support.

Be concise, supportive, and factual.
"""

    user_prompt = f"""
Student Question:
{query}

FAQ Context:
{context}

Using ONLY the FAQ context above, answer the question.
If insufficient information exists, say so explicitly
and recommend booking a Student Success Advisor:
{ADVISOR_BOOKING_LINK}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user",    "content": user_prompt.strip()},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"[LLM Answer Generated] Query: {query}")
        return answer

    except Exception as e:
        logger.error(f"LLM answer generation failed: {e}")
        return (
            "I'm having trouble generating a complete response right now. "
            "Please try again shortly or contact a Student Success Advisor:\n"
            f"{ADVISOR_BOOKING_LINK}"
        )