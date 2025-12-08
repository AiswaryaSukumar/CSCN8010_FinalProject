"""
llm_service.py

LLM-based answer generation for the Student Affairs chatbot.

Workflow:
- TF-IDF retrieval gives top-k FAQs (DataFrame).
- We turn those rows into a context string.
- We call the OpenAI chat model with:
    * strict system prompt (scope + safety rules)
    * student query
    * FAQ context

This module does NOT decide:
- if a query is in-scope / out-of-scope
- if something is a serious crisis

Those decisions are done BEFORE calling this module,
e.g., by an intent classifier or rule-based logic.
Here we only generate an answer when it is appropriate
to do RAG-style answering.
"""

from typing import List
import pandas as pd
from openai import OpenAI

# Create a single client for reuse
client = OpenAI()

# Where to send serious / sensitive queries
ADVISOR_BOOKING_LINK = "https://successportal.conestogac.on.ca"  # adjust if you want


def build_faq_context(faq_rows: pd.DataFrame, max_chunks: int = 3) -> str:
    """
    Turn top-k FAQ rows into a single context string for the LLM.

    We assume faq_rows has columns:
        - question
        - answer
        - source_type (optional)
    """
    context_parts: List[str] = []

    for i, (_, row) in enumerate(faq_rows.head(max_chunks).iterrows(), start=1):
        q = str(row["question"]).strip().lower()
        a = str(row["answer"]).strip()
        source = str(row.get("source_type", ""))
        context_parts.append(
            f"FAQ #{i}\n"
            f"source: {source}\n"
            f"q: {q}\n"
            f"a: {a}\n"
        )

    if not context_parts:
        return "No FAQ context available."

    return "\n\n".join(context_parts)


def generate_llm_answer(query: str, faq_rows: pd.DataFrame) -> str:
    """
    Call the OpenAI model with:
      - a strict system prompt (role + safety rules)
      - the student query
      - the FAQ context

    This function assumes the query has already been checked as:
        - in student-affairs scope
        - not a severe crisis that must be escalated directly

    Returns:
        A single string answer suitable to show to the user.
    """
    context = build_faq_context(faq_rows, max_chunks=3)

    system_prompt = f"""
You are a virtual Student Success Assistant for Conestoga College.

Your job:
- Help students with questions about:
  * orientation and transition to Conestoga
  * student success services and academic support
  * career services and career planning
  * student rights and responsibilities
  * related campus services and procedures

Rules:
- Base your answers ONLY on the FAQ context I provide.
- If the context does not contain enough information to answer confidently,
  say you are not sure and recommend the student contact a Student Success Advisor
  using this link: {ADVISOR_BOOKING_LINK}.
- If the question is clearly outside Student Success scope
  (for example: immigration advice, legal advice, medical diagnosis,
  mental health crisis, or unrelated general topics), do NOT answer directly.
  Instead, respond briefly that you can't help with that and suggest contacting
  the appropriate professional or a Student Success Advisor.
- If the student mentions self-harm, harassment, violence, or serious personal crisis,
  do NOT try to solve it yourself. Encourage them to seek immediate help from
  campus support services, emergency contacts, or local emergency services.
- Be concise, clear, and supportive. Use a friendly but professional tone.
"""

    user_prompt = f"""
Student question:
{query}

FAQ context:
{context}

Using ONLY the information in the FAQ context and obeying the rules above,
provide the best possible answer for the student.
If the answer is not clearly in the context, clearly say that and recommend
they book an appointment with a Student Success Advisor here:
{ADVISOR_BOOKING_LINK}.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # or "gpt-4o-mini" if you prefer
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()
