# app.py

import os
import sys
import streamlit as st

# ------------------------------------------------------------------
# Locate and import from src/
# ------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from retrieval_service import load_resources, answer_query

# Load TF-IDF index + KB once
load_resources()

# ------------------------------------------------------------------
# Streamlit page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Condors Ask!",
    layout="wide"
)

# Optional small style tweak so content doesn’t stretch edge-to-edge
st.markdown(
    """
    <style>
        .main {
            padding-left: 5rem;
            padding-right: 5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Centre layout: put everything in the middle column
# ------------------------------------------------------------------
left, center, right = st.columns([1, 2, 1])

with center:
    # Header
    st.markdown("### 🦅 Condors Ask!")
    st.caption(
        "A TF-IDF powered Student Affairs assistant for Conestoga students. "
        "Ask about orientation, student success services, career centre, or student rights."
    )

    # Toggle for debugging view of matched FAQ
    show_debug = st.toggle(
        "Show matched FAQ details (for assignment / debugging)",
        value=False,
        help="When ON, you’ll see the matched FAQ question, similarity score, and source."
    )

    # Main input
    user_input = st.text_input(
        "Ask your question:",
        placeholder="Example: what is the purpose of my ONE Card? Do I need to pick it up during orientation?",
        key="user_query",
    )

    get_answer = st.button("Get Answer", type="primary")

    # ------------------------------------------------------------------
    # Handle query
    # ------------------------------------------------------------------
    if get_answer or user_input:
        query = user_input.strip()

        if not query:
            st.warning("Please enter a question.")
        else:
            result = answer_query(query, k=3)

            st.subheader("Answer")
            st.write(result["answer"])

            mode = result.get("mode", "direct")

            # If escalation, show extra notice bar
            if mode == "escalate":
                st.warning(
                    "This looks like something that should be handled by a real person. "
                    "Please use the portal link in the answer above to reach appropriate support."
                )

            # If invalid, we just show the answer text – nothing else.

            # ------------------------------------------------------------------
            # Debug / assignment section – only when toggle is ON
            # ------------------------------------------------------------------
            if show_debug and mode in {"direct", "low_confidence"}:
                st.markdown("---")
                if mode == "direct":
                    st.subheader("Matched Knowledge Base Entry")
                else:  # low_confidence
                    st.subheader("Closest FAQ Match (for reference)")

                st.write(f"**Question:** {result['matched_question']}")
                st.write(f"**Similarity score:** {result['similarity']:.3f}")

                source_type = result.get("source_type", "")
                source_url = result.get("source_url", "")

                if source_url:
                    st.write(f"**Source:** {source_type}")
                    st.markdown(f"[Open source page]({source_url})")
                elif source_type:
                    st.write(f"**Source:** {source_type}")
