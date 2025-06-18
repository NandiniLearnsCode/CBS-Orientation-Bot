# 🚀 CBS Orientation Bot - Setup for Friends

## How to Use the Bot with Shared API Key

### Option 1: Use the Deployed Version (Easiest)

1. **Get the Streamlit Cloud URL** from the developer
2. **Open the URL** in your browser
3. **The API key is already configured** - just start chatting!

### Option 2: Run Locally with Shared API Key

#### Step 1: Clone the Repository
```bash
git clone https://github.com/NandiniLearnsCode/CBS-Orientation-Bot.git
cd CBS-Orientation-Bot
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Set Up API Key

**Method A: Environment Variable (Recommended)**
```bash
# On Mac/Linux:
export OPENAI_API_KEY="your_shared_api_key_here"

# On Windows:
set OPENAI_API_KEY=your_shared_api_key_here
```

**Method B: Create .env file**
1. Copy `env_example.txt` to `.env`
2. Replace `your_openai_api_key_here` with the actual API key
3. Install python-dotenv: `pip install python-dotenv`

#### Step 4: Run the App
```bash
streamlit run streamlit_app.py
```

#### Step 5: Open in Browser
Go to `http://localhost:8501` and start chatting!

### Option 3: Deploy Your Own Version

1. **Fork the repository** on GitHub
2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your forked repository
   - Set main file path to `streamlit_app.py`
   - Add your API key in the secrets section
3. **Share your deployment URL** with friends

## 🔑 Getting the Shared API Key

Contact the developer (Nandini) to get the shared OpenAI API key.

## 💡 Tips for Testing

Try asking questions like:
- "What services does the Office of Student Affairs provide?"
- "Tell me about J-Term at CBS"
- "What are the core courses?"
- "Where can I find coffee near campus?"
- "What are the campus resources available?"

## 🐛 Troubleshooting

**If you get an error about missing API key:**
- Make sure you've set the environment variable correctly
- Check that the .env file exists and has the correct API key
- Restart your terminal/command prompt after setting environment variables

**If the app doesn't load:**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the correct directory
- Try running with: `streamlit run streamlit_app.py --server.fileWatcherType none`

## 📞 Need Help?

Contact the developer if you encounter any issues! 