# copilot.py

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine

import os

class Copilot:
    def __init__(self, data_dir="data", model="gpt-4o"):
        self.data_dir = data_dir
        self.model = model
        self.chat_engine = self._load_engine()

    def _load_engine(self):
        # Load documents from the data directory
        documents = SimpleDirectoryReader(self.data_dir).load_data()

        # Create the embedding model
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Set up the service context
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model=self.model),
            embed_model=embed_model
        )

        # Create index and retriever
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        retriever = index.as_retriever(similarity_top_k=5)

        # Optional memory for conversation context
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        # Set up the chat engine
        chat_engine = ContextChatEngine.from_defaults(
            retriever=retriever,
            memory=memory,
            system_prompt="You are an AI-powered Columbia Orientation Bot. Be helpful, friendly, and precise."
        )

        return chat_engine

    def chat(self, user_input, chat_history):
        response = self.chat_engine.chat(user_input)
        chat_history.append(("user", user_input))
        chat_history.append(("assistant", response.response))
        return response.response, chat_history

