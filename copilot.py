import os
import streamlit as st
import nltk
from openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from tenacity import retry, wait_random_exponential, stop_after_attempt

# --- NLTK setup for Streamlit Cloud ---
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.download("punkt", download_dir=nltk_data_dir)
os.environ["NLTK_DATA"] = nltk_data_dir

# --- Retry wrapper for OpenAI calls ---
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
def chat_completion_request(client, messages, model="gpt-4o", **kwargs):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

# --- Main Copilot class ---
class Copilot:
    def __init__(self, pdf_path="data/J-Term CBS Survival Guide.pdf"):
        """
        Load the PDF, parse and embed its content, and set up a retriever for Q&A.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist")

        # Setup LLM and embedding model
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = LlamaOpenAI(model="gpt-4o")

        # Load and parse documents
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        nodes = SentenceSplitter().get_nodes_from_documents(documents)

        # Create vector index
        storage_context = StorageContext.from_defaults(vector_store=SimpleVectorStore())
        self.index = VectorStoreIndex(nodes, storage_context=storage_context)
        self.retriever = self.index.as_retriever(similarity_top_k=3)

    def get_response(self, message_history, user_input):
        """
        Generate a response using retrieved context and OpenAI completion.
        """
        retrieved_nodes = self.retriever.retrieve(user_input)
        context_str = "\n".join([node.text for node in retrieved_nodes])

        messages = message_history + [
            {"role": "system", "content": f"Use the following context to answer the question:\n\n{context_str}"},
            {"role": "user", "content": user_input}
        ]

        # Use the user's provided API key
        client = OpenAI(api_key=st.session_state.openai_api_key)
        response = chat_completion_request(client, messages)
        return response.choices[0].message.content if hasattr(response, "choices") else str(response)
