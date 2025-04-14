import os
import openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tenacity import retry, wait_random_exponential, stop_after_attempt

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(5))
def chat_completion_request(client, messages, model="gpt-4", **kwargs):
    """
    A retry-enabled function to create a chat completion request.
    """
    try:
        response = client.ChatCompletion.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

class CBSSurvivalGuideBot:
    def __init__(self, pdf_path="J-Term CBS Survival Guide.pdf"):
        """
        Initialize the bot with a specific PDF file.
        """
        # Check if the file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file {pdf_path} does not exist")
            
        # Load the PDF file using the directory reader
        reader = SimpleDirectoryReader(input_files=[pdf_path])
        docs = reader.load_data()
        print(f"Loaded {len(docs)} document(s) from {pdf_path}")
        
        # Initialize the HuggingFace embedding model
        embedding_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en"
        )
        
        # Create an index from the documents
        self.index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model,
                                                     show_progress=True)
        
        # Setup retriever with a similarity threshold
        self.retriever = self.index.as_retriever(similarity_top_k=3)
        
        # Define a system prompt specific to the domain
        self.system_prompt = (
            "You are a Columbia Business School J-Term expert. Your job is to answer questions "
            "about the J-Term CBS Survival Guide. "
            "If the question is not related to the Columbia Business School J-Term program or the survival guide, "
            "politely decline to answer and suggest asking questions related to the guide."
        )

    def ask(self, question, messages, openai_key=None):
        """
        Ask a question to the bot.
        """
        # Set the OpenAI API key and initialize the client.
        if openai_key:
            openai.api_key = openai_key
        self.llm_client = openai

        # Retrieve relevant information from the indexed documents
        nodes = self.retriever.retrieve(question)
        retrieved_info = "\n".join([f"{i+1}. {node.text}" for i, node in enumerate(nodes)])
        
        # Create a prompt that includes the user question and the retrieved information
        processed_query_prompt = (
            "The user is asking a question about the Columbia Business School J-Term Survival Guide: {question}\n\n"
            "The retrieved information from the guide is: {retrieved_info}\n\n"
            "Please answer the question based on the retrieved information if they are relevant. \n"
            "Please use markdown format and $$ for inline latex instead of \\( \\).\n\n"
            "Please highlight key information with **bold text** and bullet points.\n\n"
            "If the question cannot be answered using the guide, politely mention that "
            "the information isn't covered in the J-Term CBS Survival Guide."
        )
        
        processed_query = processed_query_prompt.format(question=question, 
                                                          retrieved_info=retrieved_info)
        
        # Build conversation messages with the system prompt and user query
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + messages
            + [{"role": "user", "content": processed_query}]
        )
        
        # Request a response from the OpenAI ChatCompletion endpoint using streaming mode
        response = chat_completion_request(self.llm_client, messages=messages, stream=True)
        
        return retrieved_info, response

if __name__ == "__main__":
    # Retrieve OpenAI API key from the environment, or prompt the user
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = input(
            "Please enter your OpenAI API Key (or set it as an environment variable OPENAI_API_KEY): "
        )
    
    # Initialize the bot with the specified PDF file
    bot = CBSSurvivalGuideBot("J-Term CBS Survival Guide.pdf")
    
    # Initialize conversation history as an empty list
    messages = []
    
    # Interactive loop for user questions
    print("\nCBS J-Term Survival Guide Assistant Ready! (Type 'exit' to quit)")
    print("=" * 60)
    
    while True:
        question = input("\nPlease ask a question: ")
        
        if question.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using the CBS J-Term Survival Guide Assistant. Goodbye!")
            break
            
        retrieved_info, answer = bot.ask(question, messages=messages, openai_key=openai_api_key)
        
        # Process and print the response
        if isinstance(answer, str):
            print(answer)
        else:
            answer_str = ""
            print("\nAnswer: ", end="")
            for chunk in answer:
                # Check safely for delta content (streaming responses)
                delta = getattr(chunk.choices[0], "delta", {})
                content = delta.get("content", "")
                if content:
                    answer_str += content
                    print(content, end="", flush=True)
            print("\n")
            answer = answer_str

        # Update conversation history with the new entries
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

