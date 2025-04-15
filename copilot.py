from openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI  # UPDATED
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from tenacity import retry, wait_random_exponential, stop_after_attempt
import os

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

class Copilot:
    def __init__(self, pdf_path="J-Term CBS Survival Guide.pdf"):
        """
        Initialize the chatbot by loading a PDF, parsing its content, embedding it,
        and preparing a vector index for retrieval.
        """
        # Check if the file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist")

        # Set up the embedding model and LLM
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        Settings.llm = LlamaOpenAI(model="gpt-4o")  # UPDATED

        # Read and parse the document
        documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        nodes = SentenceSplitter().get_nodes_from_documents(documents)

        # Create and persist the vector index
        storage_context = StorageContext.from_defaults(vector_store=SimpleVectorStore())
        self.index = VectorStoreIndex(nodes, storage_context=storage_context)

        # Set up the retriever
        self.retriever = self.index.as_retriever(similarity_top_k=3)

    def get_response(self, message_history, user_input):
        """
        Get a model-generated response using context from the indexed document.
        """
        retrieved_nodes = self.retriever.retrieve(user_input)
        context_str = "\n".join([node.text for node in retrieved_nodes])

        messages = message_history + [
            {"role": "system", "content": f"Use the following context to answer the question:\n\n{context_str}"},
            {"role": "user", "content": user_input}
        ]

        response = chat_completion_request(OpenAI(), messages)
        return response.choices[0].message.content if hasattr(response, "choices") else str(response)

