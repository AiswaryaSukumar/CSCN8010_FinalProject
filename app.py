import streamlit as st
from src.retrieval_service import load_resources, answer_query
from src.translation import translate_text
import time

# Load models
load_resources()

# Page setup
st.set_page_config(page_title="Condors Ask!", layout="wide")

# -------------------------------------------------------------------
# CSS Styles
# -------------------------------------------------------------------
st.markdown("""
<style>
.chat-window {
    height: 70vh;
    overflow-y: auto;
    padding: 1rem;
    border-radius: 12px;
    background: #111;
    border: 1px solid #333;
}
.chat-bubble-user {
    background: #0056ff;
    color: white;
    padding: 12px 16px;
    border-radius: 16px 16px 0 16px;
    margin: 8px 0;
    max-width: 70%;
    float: right;
    clear: both;
}
.chat-bubble-bot {
    background: #1e1f24;
    color: #e8e8e8;
    padding: 12px 16px;
    border-radius: 16px 16px 16px 0;
    margin: 8px 0;
    max-width: 70%;
    float: left;
    clear: both;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------------------------------------------
# Sidebar Language Selector
# -------------------------------------------------------------------
st.sidebar.title("🌐 Language")

language = st.sidebar.selectbox("Select language:", ["English", "French", "Hindi"])
lang_code = {"English": "en", "French": "fr", "Hindi": "hi"}[language]

# Layout
chat_col, news_col = st.columns([3, 1])

# -------------------------------------------------------------------
# CHAT COLUMN
# -------------------------------------------------------------------
with chat_col:

    st.markdown("<h2>🦅 Condors Ask!</h2>", unsafe_allow_html=True)

    # Chat window container
   # st.markdown('<div class="chat-window">', unsafe_allow_html=True)

    for sender, msg in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f'<div class="chat-bubble-user">{msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-bot">{msg}</div>', unsafe_allow_html=True)

    #st.markdown('</div>', unsafe_allow_html=True)

    # User input
    user_message = st.text_input("Type your message:", value="", placeholder="Ask me anything...")

    send_btn = st.button("Send")

    if send_btn and user_message.strip() != "":
        st.session_state.chat_history.append(("user", user_message))
        st.session_state.chat_history.append(("bot", "typing..."))
        st.rerun()   # <-- FIXED HERE

    # Process typing placeholder AFTER rerun
    if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "typing...":
        last_user_message = st.session_state.chat_history[-2][1]

        time.sleep(0.2)
        result = answer_query(last_user_message)
        reply = translate_text(result["answer"], lang_code)

        st.session_state.chat_history[-1] = ("bot", reply)
        st.rerun()   # <-- FIXED HERE


# -------------------------------------------------------------------
# NEWS COLUMN
# -------------------------------------------------------------------
with news_col:
    st.markdown("<h4>📢 News & Updates</h4>", unsafe_allow_html=True)

    news_items = [
        {"title": "Campus Winter Hours Updated", "link": "#"},
        {"title": "New Mental Health Services Now Open", "link": "#"},
        {"title": "Career Center Job Fair – Feb 12", "link": "#"}
    ]

    for item in news_items:
        st.markdown(
            f"<p><a href='{item['link']}' style='color:#4A53E8;text-decoration:none;'>{item['title']}</a></p>",
            unsafe_allow_html=True
        )