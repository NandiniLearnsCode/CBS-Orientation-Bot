import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Retry logic for OpenAI calls
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

def scrape_website(url):
    """
    Scrape content from a website and return it as a Document.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Create a Document object
        document = Document(text=text, metadata={"source": url, "type": "website"})
        return document
        
    except Exception as e:
        print(f"Error scraping website {url}: {e}")
        return None

class Copilot:
    def __init__(self, data_folder="data", website_url="https://students.business.columbia.edu/office-of-student-affairs"):
        """
        Load and embed multiple PDFs and website content for Q&A using OpenAI and LlamaIndex.
        """
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"The folder {data_folder} does not exist")

        # Set up embedding and LLM
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = LlamaOpenAI(model="gpt-4o")

        # Load all PDFs from the data folder
        pdf_files = []
        for file in os.listdir(data_folder):
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(data_folder, file)
                pdf_files.append(pdf_path)
                print(f"Found PDF: {file}")

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {data_folder}")

        # Load PDF documents
        documents = SimpleDirectoryReader(input_files=pdf_files).load_data()
        print(f"Loaded {len(documents)} PDF documents")

        # Scrape website content
        website_doc = scrape_website(website_url)
        if website_doc:
            documents.append(website_doc)
            print(f"Added website content from: {website_url}")
        else:
            print(f"Failed to scrape website: {website_url}")

        # Split documents into nodes
        splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"Created {len(nodes)} nodes from documents")

        # Build vector index
        storage_context = StorageContext.from_defaults(vector_store=SimpleVectorStore())
        self.index = VectorStoreIndex(nodes, storage_context=storage_context)
        self.retriever = self.index.as_retriever(similarity_top_k=5)  # Increased to get more context

    def get_response(self, message_history, user_input):
        """
        Retrieve context from the documents and generate a response using OpenAI.
        """
        # Retrieve relevant context
        retrieved_nodes = self.retriever.retrieve(user_input)
        context_str = "\n".join([node.text for node in retrieved_nodes])
        
        # Debug print
        print(f"Retrieved context for query '{user_input}':")
        print(f"Number of context chunks: {len(retrieved_nodes)}")
        print(context_str[:500] + "..." if len(context_str) > 500 else context_str)
        print("\n" + "="*50 + "\n")

        # Check if we have sufficient context
        if not context_str.strip():
            # If no context found, use OpenAI directly for general questions
            print("No context found, using OpenAI directly")
            messages = message_history + [
                {"role": "system", "content": "You are a helpful assistant for Columbia Business School students. Answer questions based on your general knowledge about business schools and student life."},
                {"role": "user", "content": user_input}
            ]
        else:
            # Use context from documents
            if len(message_history) == 0:
                messages = [
                    {"role": "system", "content": f"You are a helpful assistant for Columbia Business School students. Use the following context to answer questions about CBS:\n\n{context_str}"},
                    {"role": "user", "content": user_input}
                ]
            else:
                messages = message_history + [{"role": "user", "content": f"Context: {context_str}\n\nQuestion: {user_input}"}]

        client = OpenAI(api_key=st.session_state.openai_api_key)
        response = chat_completion_request(client, messages)
        return response.choices[0].message.content if hasattr(response, "choices") else str(response)

    def get_data_sources_info(self):
        """
        Return information about the data sources being used.
        """
        data_folder = "data"
        pdf_files = []
        for file in os.listdir(data_folder):
            if file.lower().endswith('.pdf'):
                pdf_files.append(file)
        
        return {
            "pdf_files": pdf_files,
            "website": "https://students.business.columbia.edu/office-of-student-affairs",
            "total_sources": len(pdf_files) + 1  # +1 for website
        }
