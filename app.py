import streamlit as st
from src.retrieval_service import load_resources, answer_query
from src.translation import translate_text
import time
import base64

# ------------------------------
# Load resources
# ------------------------------
load_resources()

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Condors Ask!", layout="wide")


# ------------------------------
# Helper: Load image as Base64
# ------------------------------
def load_base64_image(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None


# Load avatar images
bot_avatar = load_base64_image("assets/bot.png")
user_avatar = load_base64_image("assets/user.png")


# ------------------------------
# CSS Styles
# ------------------------------
st.markdown(
    """
<style>
/* Chat row container */
.chat-row {
    display: flex;
    align-items: center;
    margin: 10px 0;
}

/* Avatar style */
.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
}

/* Bot bubble */
.chat-bubble-bot {
    background: #1e1f24;
    color: #e8e8e8;
    padding: 12px 16px;
    border-radius: 16px 16px 16px 0;
    margin-left: 10px;
    max-width: 70%;
    line-height: 1.4;
}

/* User bubble */
.chat-bubble-user {
    background: #0056ff;
    color: white;
    padding: 12px 16px;
    border-radius: 16px 16px 0 16px;
    margin-right: 10px;
    max-width: 70%;
    line-height: 1.4;
}

/* Typing animation */
.typing-dots {
    display: inline-block;
    width: 8px;
    height: 8px;
    margin-left: 2px;
    background-color: #e8e8e8;
    border-radius: 50%;
    animation: blink 1.4s infinite both;
}
.typing-dots:nth-child(2) { animation-delay: 0.2s; }
.typing-dots:nth-child(3) { animation-delay: 0.4s; }

@keyframes blink {
    0% { opacity: .2; }
    20% { opacity: 1; }
    100% { opacity: .2; }
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 13px;
    color: #888;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------
# SESSION STATE
# ------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_box" not in st.session_state:
    st.session_state.input_box = ""


# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.title("🌐 Language")
language = st.sidebar.selectbox("Select language:", ["English", "French", "Hindi"])
lang_code = {"English": "en", "French": "fr", "Hindi": "hi"}[language]

st.sidebar.markdown("---")

# College Logo
try:
    st.sidebar.image("assets/college_logo.png", width=220)
except:
    st.sidebar.error("College logo missing in assets folder.")

# About Section
st.sidebar.markdown(
    """
### 🦅 About Condors Ask!
Your AI-powered student support assistant.

This tool helps with:
- Academic advising  
- Course planning  
- Fees & payments  
- Campus services  
- Mental health resources  

Type your question → Get instant support!
"""
)

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.input_box = ""
    st.rerun()


chat_col, news_col = st.columns([3, 1])


# ==========================================================
# CHAT COLUMN
# ==========================================================
with chat_col:

    st.markdown("<h2>🦅 Condors Ask!</h2>", unsafe_allow_html=True)

    # Render chat history
    for sender, msg in st.session_state.chat_history:

        if sender == "user":
            avatar = user_avatar
            st.markdown(
                f"""
                <div class="chat-row" style="justify-content: flex-end;">
                    <div class="chat-bubble-user">{msg}</div>
                    <img class="avatar" src="data:image/png;base64,{avatar}">
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            avatar = bot_avatar
            st.markdown(
                f"""
                <div class="chat-row" style="justify-content: flex-start;">
                    <img class="avatar" src="data:image/png;base64,{avatar}">
                    <div class="chat-bubble-bot">{msg}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --------------------------
    # SEND MESSAGE CALLBACK
    # --------------------------
    def send_message():
        user_message = st.session_state.input_box.strip()
        if not user_message:
            return

        # Add user message
        st.session_state.chat_history.append(("user", user_message))

        # Add typing animation
        st.session_state.chat_history.append(
            ("bot", "<span class='typing-dots'></span><span class='typing-dots'></span><span class='typing-dots'></span>")
        )

        st.session_state.input_box = ""  # Reset input

    st.text_input("Type your message:", key="input_box", placeholder="Ask me anything...")
    st.button("Send", on_click=send_message)

    # --------------------------
    # PROCESS BOT RESPONSE
    # --------------------------
    if (
        st.session_state.chat_history
        and "typing-dots" in st.session_state.chat_history[-1][1]
    ):
        last_user_message = st.session_state.chat_history[-2][1]

        time.sleep(1.0)
        result = answer_query(last_user_message)
        reply = translate_text(result["answer"], lang_code)

        # Replace typing animation
        st.session_state.chat_history[-1] = ("bot", reply)

        st.rerun()


# ==========================================================
# NEWS COLUMN
# ==========================================================
with news_col:
    st.markdown(
        """
        <div style="
            background-color:#1E1E2F;
            padding:15px;
            border-radius:10px;
            border:1px solid #3A3A53;
        ">
            <h4 style='color:#9FA8FF;margin-bottom:10px;'>📢 News & Updates</h4>
        """,
        unsafe_allow_html=True,
    )

    news_items = [
        {"title": "Campus Winter Hours Updated", "link": "#"},
        {"title": "New Mental Health Services Now Open", "link": "#"},
        {"title": "Career Center Job Fair – Feb 12", "link": "#"},
    ]

    for item in news_items:
        st.markdown(
            f"""
            <p style='margin:4px 0;'>
                <a href='{item['link']}'
                style='color:#4A53E8;text-decoration:none;font-weight:500;'>
                {item['title']}
                </a>
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ==========================================================
# FOOTER
# ==========================================================
st.markdown(
    """
<div class="footer">
    Condors Ask! v1.0 — Built by Group 4 | Powered by Conestoga College
</div>
""",
    unsafe_allow_html=True
)
