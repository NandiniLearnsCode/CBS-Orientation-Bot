import os
import nltk
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.insert(0, nltk_data_dir)
import streamlit as st
from copilot import Copilot

# --- Page Configuration ---
st.set_page_config(page_title="CBS Peer Advisor", layout="centered")

# --- Title ---
st.title("Chat with CBS Peer Advisor")

# --- OpenAI API Key Input ---
# Check if API key is set in environment variables or Streamlit secrets
default_api_key = os.getenv('OPENAI_API_KEY', '') or st.secrets.get('OPENAI_API_KEY', '')

if default_api_key:
    # If API key is in environment or secrets, use it automatically
    openai_api_key = default_api_key
    st.success("✅ API key loaded automatically")
else:
    # Otherwise, ask user to input it
    openai_api_key = st.text_input("Please enter your OpenAI API Key", type="password")
    
    if not openai_api_key:
        st.info("🔑 Please add your OpenAI API key to continue.")
        st.info("💡 **For friends:** Ask the developer for the shared API key!")
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

# --- Show data sources info ---
with st.sidebar:
    st.header("📚 Data Sources")
    try:
        sources_info = st.session_state.chat_copilot.get_data_sources_info()
        st.write(f"**Total Sources:** {sources_info['total_sources']}")
        
        st.write("**PDF Files:**")
        for pdf in sources_info['pdf_files']:
            st.write(f"• {pdf}")
        
        st.write("**Website:**")
        st.write(f"• {sources_info['website']}")
        
        st.write("---")
        st.write("💡 **Tip:** The bot searches through all these sources to find relevant information for your questions!")
        
    except Exception as e:
        st.error(f"Error loading data sources info: {e}")

# --- Assistant welcome message ---
if len(st.session_state.messages) == 0:
    st.chat_message("assistant").markdown(
        "Hi! I'm your **Peer Advisor** 🤝\n\n"
        "I can help you with questions about Columbia Business School using information from:\n"
        "• Multiple PDF documents in the data folder\n"
        "• The CBS Office of Student Affairs website\n\n"
        "Ask me anything about navigating CBS life — from academics to campus resources!"
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
        with st.spinner("Searching through CBS resources..."):
            response = st.session_state.chat_copilot.get_response(
                message_history=st.session_state.messages[:-1],
                user_input=user_input
            )
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
