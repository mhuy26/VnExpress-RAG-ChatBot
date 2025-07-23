
# agent.py
# Retrieval-Augmented Generation (RAG) pipeline for VnExpress news using Gemini LLM and Qdrant vector search.
#
# This module provides a manual RAG pipeline: retrieves relevant news articles from Qdrant,
# then generates a concise answer using a Gemini LLM. Designed for clarity, extensibility, and reproducibility.

import getpass
import os
from dotenv import load_dotenv
load_dotenv()


# LangChain and Qdrant imports
from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant


# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")  # Qdrant server URL
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vnexpress_articles")  # Qdrant collection name
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING", "nomic-embed-text")  # Embedding model for text

# Ensure Google API key is set for Gemini LLM
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Missing GOOGLE_API_KEY in .env")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# --- Model and Vector Store Initialization ---
# Initialize Gemini LLM and embedding model
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
# embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Initialize Qdrant vector store for document retrieval
client = QdrantClient(url=QDRANT_URL)
vectorstore = Qdrant(
    collection_name=QDRANT_COLLECTION,
    embeddings=embedding,  # Note: For langchain_qdrant, use 'embeddings' for backward compatibility
    client=client
)

# --- Dynamic K Retrieval ---
def get_dynamic_k(user_query):
    """
    Dynamically determine the number of documents to retrieve based on the user query.
    Args:
        user_query (str): The user's input question.
    Returns:
        int: Number of documents to retrieve (higher for summary-type queries).
    """
    if "summary" in user_query.lower():
        return 100
    else:
        return 5


# --- Step 1: Manual document retrieval ---
def retrieve_documents(query: str, k=None) -> str:
    """
    Retrieve relevant documents from Qdrant using Max Marginal Relevance (MMR).
    Args:
        query (str): The user's search query.
        k (int, optional): Number of documents to retrieve. If None, uses get_dynamic_k().
    Returns:
        str: Concatenated string of retrieved document titles and content.
    """
    k = k or get_dynamic_k(query)
    docs = vectorstore.max_marginal_relevance_search(
    query,
    k=k,
    fetch_k=min(5 * k, 200)
)

    return "\n\n".join(f"📌 {doc.metadata.get('title', '')}\n{doc.page_content}" for doc in docs)


# --- Step 2: Compose prompt and generate response ---
def generate_response(user_question: str, retrieved_content: str) -> str:
    """
    Generate a concise answer using the Gemini LLM, given the user question and retrieved content.
    Args:
        user_question (str): The user's question.
        retrieved_content (str): The context retrieved from Qdrant.
    Returns:
        str: The LLM's answer.
    """
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved content to answer concisely. "
        "If unsure, say you don’t know. Use three sentences max.\n\n"
        f"{retrieved_content}"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]
    response = llm.invoke(messages)
    return response.content


from langsmith.run_helpers import traceable

@traceable(name="vnexpress_rag_manual_rag")
def run_pipeline():
    """
    End-to-end pipeline: retrieves relevant news articles and generates an answer.
    Example query is hardcoded for demonstration.
    """
    query = "Ủy ban Thường vụ Quốc hội đã làm gì ngày 10/7"
    retrieved_text = retrieve_documents(query)
    final_answer = generate_response(query, retrieved_text)
    print("\n🧠 Final Answer:")
    print(final_answer)


if __name__ == "__main__":
    run_pipeline()

