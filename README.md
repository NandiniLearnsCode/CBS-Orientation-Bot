# 🦙📚 Columbia Business School Orientation Chatbot

Columbia Business School Orientation chatbot Copilot is an AI-powered chatbot designed to answer questions from students during orientation. It leverages the capabilities of OpenAI's GPT-4o and LlamaIndex to provide accurate and contextually relevant responses based on the data it has been trained on.

## Overview of the App

- Takes user queries via Streamlit's `st.chat_input` and displays both user queries and model responses with `st.chat_message`.
- Use `Copilot` class to interact with the chatbot by passing in a question and a list of messages and get a response.
- Uses LlamaIndex to load and index data and create a Retriever to retrieve context from that data to respond to each user query.
- **NEW**: Supports multiple PDFs and website content for comprehensive CBS information.

## Data Sources

The bot uses information from:
- Multiple PDF documents in the `data/` folder
- CBS Office of Student Affairs website: https://students.business.columbia.edu/office-of-student-affairs

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://llamaindex-chat-with-student-handbook-8tp48eikcchw2w9g9fvmsj.streamlit.app/)

## How to Share with Friends

### Option 1: Deploy to Streamlit Cloud (Recommended)

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Enhanced CBS Orientation Bot with multiple data sources"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository and set the path to `streamlit_app.py`
   - Deploy!

3. **Share the URL** with your friends - they'll be able to use the bot directly in their browser!

### Option 2: Local Network Sharing

1. **Find your local IP address:**
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```

2. **Run the app with network access:**
   ```bash
   streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
   ```

3. **Share the URL** with friends on the same network: `http://YOUR_IP:8501`

### Option 3: Package as Executable

1. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

2. **Create executable:**
   ```bash
   pyinstaller --onefile --add-data "data:data" streamlit_app.py
   ```

3. **Share the executable** file with friends (they'll need to install Python dependencies)

## Run the app locally

1. Clone the repository
2. Install the dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run streamlit_app.py
```

## Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:
1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button, then copy the API key.

## Features

- **Multi-source Information**: Searches through multiple PDFs and website content
- **Smart Context Retrieval**: Finds the most relevant information for each question
- **OpenAI Integration**: Uses GPT-4o for natural language responses
- **User-friendly Interface**: Clean Streamlit interface with chat history
- **Real-time Debugging**: Shows retrieved context in terminal for transparency

## Testing Questions

Try asking:
- "What services does the Office of Student Affairs provide?"
- "Tell me about J-Term at CBS"
- "What are the core courses?"
- "Where can I find coffee near campus?"
- "What are the campus resources available?"
