import os
import sys
import streamlit as st

# ------------------------------------------------------------------
# Import retrieval service from src/
# ------------------------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from retrieval_service import load_resources, answer_query

# Load TF-IDF index and KB once at startup
load_resources()

# ------------------------------------------------------------------
# Streamlit page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Student Affairs Self-Service Assistant",
    layout="wide"
)

st.title("Student Affairs Self-Service Assistant (Dry Run -2)")

st.markdown(
    """
This prototype uses a TF-IDF based retrieval model over Conestoga Student Affairs
resources (orientation, career centre, student rights, etc.).

Type a question below and the system will retrieve the most relevant FAQ and show its answer.
"""
)

# ------------------------------------------------------------------
# Main input area
# ------------------------------------------------------------------

query = st.text_input("Ask your question:", placeholder="Example: How do I access Wi-Fi on campus?")

if st.button("Get Answer") or query:
    query = query.strip()
    if not query:
        st.warning("Please enter a question.")
    else:
        result = answer_query(query, k=3)

        st.subheader("Answer")
        st.write(result["answer"])

        st.markdown("---")
        st.subheader("Matched Knowledge Base Entry")
        st.write(f"Question: {result['matched_question']}")
        st.write(f"Similarity score: {result['similarity']:.3f}")

        if result.get("source_url"):
            st.write(f"Source: {result['source_type']}")
            st.markdown(f"[Open source page]({result['source_url']})")
        else:
            st.write(f"Source: {result['source_type']}")
