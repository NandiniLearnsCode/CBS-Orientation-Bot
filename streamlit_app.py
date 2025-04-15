import streamlit as st
from copilot import Copilot

# --- Page Configuration ---
st.set_page_config(page_title="CBS Peer Advisor", layout="centered")

# --- Title ---
st.title("Chat with CBS Peer Advisor")

# --- OpenAI API Key Input ---
openai_api_key = st.text_input("Please enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.info("🔑 Please add your OpenAI API key to continue.")
    st.stop()

# Store the API key for use in copilot.py
st.session_state.openai_api_key = openai_api_key

# --- Initialize chat state ---
@st.cache_resource
def load_copilot():
    return Copilot()

if "chat_copilot" not in st.session_state:
    st.session_state.chat_copilot = load_copilot()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Assistant welcome message ---
if len(st.session_state.messages) == 0:
    st.chat_message("assistant").markdown(
        "Hi! I’m your **Peer Advisor** 🤝\n\nAsk me anything about navigating CBS life — from academics to fun things around campus!"
    )

# --- Chat history loop ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Chat input ---
user_input = st.chat_input("What would you like to ask?")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response = st.session_state.chat_copilot.get_response(
            message_history=st.session_state.messages[:-1],
            user_input=user_input
        )
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
