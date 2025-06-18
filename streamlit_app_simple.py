import streamlit as st
import os

# --- Page Configuration ---
st.set_page_config(page_title="CBS Peer Advisor", layout="centered")

# --- Title ---
st.title("Chat with CBS Peer Advisor")

# --- OpenAI API Key Input ---
openai_api_key = st.text_input("Please enter your OpenAI API Key", type="password")

if not openai_api_key:
    st.info("🔑 Please add your OpenAI API key to continue.")
    st.stop()

# Store the API key
st.session_state.openai_api_key = openai_api_key

# --- Initialize chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Assistant welcome message ---
if len(st.session_state.messages) == 0:
    st.chat_message("assistant").markdown(
        "Hi! I'm your **Peer Advisor** 🤝\n\nAsk me anything about navigating CBS life — from academics to fun things around campus!"
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
        # For now, just echo back to test the interface
        response = f"I received your question: {user_input}. The full chatbot functionality will be loaded once the API key is working."
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response}) 