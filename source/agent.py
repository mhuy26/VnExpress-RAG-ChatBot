"""
A script to retrieve relevant news articles from Qdrant and generate concise answers using Gemini.
Designed for clarity, extensibility, and reproducibility.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# LangChain and Qdrant imports
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # CLOUD: Gemini embedding
# from langchain_ollama import OllamaEmbeddings  # LOCAL: Ollama embedding (commented below)

# --- Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vnexpress_articles")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING", "nomic-embed-text")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Missing GOOGLE_API_KEY in .env")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# --- Model and Vector Store Initialization ---

# CLOUD: Gemini LLM + Google embeddings for Streamlit Cloud
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# LOCAL: Ollama LLM + Ollama embeddings (uncomment to use locally)
# llm = init_chat_model("llama3", model_provider="ollama")
# embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

# CLOUD: Qdrant Cloud or public Qdrant
client = QdrantClient(url=QDRANT_URL)

# LOCAL: if using local Qdrant, same URL but without auth
# client = QdrantClient(url="http://localhost:6333")

vectorstore = Qdrant(
    collection_name=QDRANT_COLLECTION,
    embeddings=embedding,
    client=client
)


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


def retrieve_documents(query: str, k=None) -> str:
    """
    Retrieve relevant documents from Qdrant.
    Uses similarity search instead of MMR for better reliability with small datasets.

    Args:
        query (str): The user's search query.
        k (int, optional): Number of documents to retrieve. If None, uses get_dynamic_k().
    Returns:
        str: Concatenated string of retrieved document titles and content.
    """
    k = k or get_dynamic_k(query)

    # simplified retrieval for small collections
    docs = vectorstore.similarity_search(query, k=k)

    # use MMR retrieval when you have many docs (100+)
    # docs = vectorstore.max_marginal_relevance_search(
    #     query,
    #     k=k,
    #     fetch_k=min(5 * k, 200)
    # )

    return "\n\n".join(f"📌 {doc.metadata.get('title', '')}\n{doc.page_content}" for doc in docs)


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
