# """
# A script to retrieve relevant news articles from Qdrant and generate concise answers using Gemini.
# Designed for clarity, extensibility, and reproducibility.
# """

# import os
# from dotenv import load_dotenv
# load_dotenv()

# # LangChain and Qdrant imports
# from langchain.chat_models import init_chat_model
# from langchain_core.messages import SystemMessage, HumanMessage
# from qdrant_client import QdrantClient
# from langchain_qdrant import Qdrant
# from langchain_google_genai import GoogleGenerativeAIEmbeddings  # CLOUD: Gemini embedding
# # from langchain_ollama import OllamaEmbeddings  # LOCAL: Ollama embedding (commented below)

# # --- Configuration ---
# QDRANT_URL = os.getenv("QDRANT_URL")
# QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vnexpress_articles")
# # EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING", "nomic-embed-text")
# EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL_NAME", "models/embedding-001")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise EnvironmentError("Missing GOOGLE_API_KEY in .env")
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# # --- Model and Vector Store Initialization ---

# # CLOUD: Gemini LLM + Google embeddings for Streamlit Cloud
# llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
# embedding = GoogleGenerativeAIEmbeddings(
#     model=EMBEDDING_MODEL,
#     google_api_key=GOOGLE_API_KEY
# )

# # LOCAL: Ollama LLM + Ollama embeddings (uncomment to use locally)
# # llm = init_chat_model("llama3", model_provider="ollama")
# # embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

# # CLOUD: Qdrant Cloud or public Qdrant
# # CLOUD: Qdrant Cloud or public Qdrant
# try:
#     client = QdrantClient(url=QDRANT_URL)
#     print(f"✅ Qdrant client initialized with URL: {QDRANT_URL}")
# except Exception as e:
#     print(f"❌ Failed to initialize Qdrant client: {e}")
#     raise


# # LOCAL: if using local Qdrant, same URL but without auth
# # client = QdrantClient(url="http://localhost:6333")

# # Initialize vectorstore safely
# vectorstore, connection_success = initialize_vectorstore_safely()

# if not connection_success:
#     print("❌ Failed to establish vectorstore connection")
#     # You might want to exit here or provide a fallback
    
# # For debugging: print connection status
# connection_status = test_qdrant_connection()
# print(f"🔍 Connection status: {connection_status}")


# def get_dynamic_k(user_query):
#     """
#     Dynamically determine the number of documents to retrieve based on the user query.
#     Args:
#         user_query (str): The user's input question.
#     Returns:
#         int: Number of documents to retrieve (higher for summary-type queries).
#     """
#     if "summary" in user_query.lower():
#         return 100
#     else:
#         return 5


# def retrieve_documents(query: str, k=None) -> str:
#     """
#     Retrieve relevant documents from Qdrant with robust error handling.
    
#     Args:
#         query (str): The user's search query.
#         k (int, optional): Number of documents to retrieve. If None, uses get_dynamic_k().
#     Returns:
#         str: Concatenated string of retrieved document titles and content.
#     """
#     k = k or get_dynamic_k(query)
    
#     try:
#         # Test connection first
#         collections = client.get_collections()
#         collection_exists = any(col.name == QDRANT_COLLECTION for col in collections.collections)
        
#         if not collection_exists:
#             print(f"❌ Collection '{QDRANT_COLLECTION}' does not exist")
#             return "⚠️ Vector database collection not found. Please check your setup."
        
#         # Perform similarity search
#         docs = vectorstore.similarity_search(query, k=k)
        
#         if not docs:
#             return "🔍 No relevant documents found for your query."
            
#         return "\n\n".join(f"📌 {doc.metadata.get('title', '')}\n{doc.page_content}" for doc in docs)
        
#     except Exception as e:
#         error_msg = str(e)
#         print(f"❌ Qdrant error in retrieve_documents: {type(e).__name__}: {error_msg}")
        
#         # Handle specific error types
#         if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
#             return "🔌 Unable to connect to the vector database. Please check your connection."
#         elif "unauthorized" in error_msg.lower() or "forbidden" in error_msg.lower():
#             return "🔐 Authentication failed. Please check your API credentials."
#         elif "not found" in error_msg.lower():
#             return "📂 Collection not found in the vector database."
#         else:
#             return f"⚠️ Search failed: {error_msg}"


# def test_qdrant_connection():
#     """
#     Test Qdrant connection and return status information.
#     Returns:
#         dict: Connection status and collection info
#     """
#     try:
#         # Test basic connection
#         collections = client.get_collections()
#         collection_names = [col.name for col in collections.collections]
        
#         # Check if our specific collection exists
#         collection_exists = QDRANT_COLLECTION in collection_names
        
#         # Get collection info if it exists
#         collection_info = None
#         if collection_exists:
#             collection_info = client.get_collection(QDRANT_COLLECTION)
        
#         return {
#             "status": "connected",
#             "collections": collection_names,
#             "target_collection": QDRANT_COLLECTION,
#             "target_exists": collection_exists,
#             "collection_info": collection_info
#         }
        
#     except Exception as e:
#         return {
#             "status": "failed",
#             "error": str(e),
#             "error_type": type(e).__name__
#         }


# def initialize_vectorstore_safely():
#     """
#     Initialize vectorstore with connection testing.
#     Returns:
#         tuple: (vectorstore, success_status)
#     """
#     try:
#         # Test connection first
#         connection_status = test_qdrant_connection()
        
#         if connection_status["status"] != "connected":
#             print(f"❌ Connection test failed: {connection_status.get('error', 'Unknown error')}")
#             return None, False
            
#         if not connection_status["target_exists"]:
#             print(f"❌ Collection '{QDRANT_COLLECTION}' does not exist")
#             print(f"Available collections: {connection_status['collections']}")
#             return None, False
            
#         # Initialize vectorstore
#         vectorstore = Qdrant(
#             collection_name=QDRANT_COLLECTION,
#             embeddings=embedding,
#             client=client
#         )
        
#         print(f"✅ Successfully connected to Qdrant collection '{QDRANT_COLLECTION}'")
#         return vectorstore, True
        
#     except Exception as e:
#         print(f"❌ Failed to initialize vectorstore: {e}")
#         return None, False


# def generate_response(user_question: str, retrieved_content: str) -> str:
#     """
#     Generate a concise answer using the Gemini LLM, given the user question and retrieved content.
#     Args:
#         user_question (str): The user's question.
#         retrieved_content (str): The context retrieved from Qdrant.
#     Returns:
#         str: The LLM's answer.
#     """
#     system_prompt = (
#         "You are an assistant for question-answering tasks. "
#         "Use the following retrieved content to answer concisely. "
#         "If unsure, say you don’t know. Use three sentences max.\n\n"
#         f"{retrieved_content}"
#     )
#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content=user_question)
#     ]
#     response = llm.invoke(messages)
#     return response.content


# from langsmith.run_helpers import traceable

# @traceable(name="vnexpress_rag_manual_rag")
# def run_pipeline():
#     """
#     End-to-end pipeline: retrieves relevant news articles and generates an answer.
#     Example query is hardcoded for demonstration.
#     """
#     query = "Ủy ban Thường vụ Quốc hội đã làm gì ngày 10/7"
#     retrieved_text = retrieve_documents(query)
#     final_answer = generate_response(query, retrieved_text)
#     print("\n🧠 Final Answer:")
#     print(final_answer)


# if __name__ == "__main__":
#     run_pipeline()
"""
A script to retrieve relevant news articles from Qdrant and generate concise answers using Gemini.
Designed for clarity, extensibility, and reproducibility with robust error handling.
"""

import os
import time
from functools import wraps
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
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Added API key support
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vnexpress_articles")
# EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING", "nomic-embed-text")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL_NAME", "models/embedding-001")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def validate_environment():
    """Validate all required environment variables are present."""
    required_vars = {
        "QDRANT_URL": QDRANT_URL,
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "QDRANT_COLLECTION": QDRANT_COLLECTION
    }
    
    missing_vars = []
    for var_name, var_value in required_vars.items():
        if not var_value:
            missing_vars.append(var_name)
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print("✅ All required environment variables are present")
    print(f"📊 Using collection: {QDRANT_COLLECTION}")
    print(f"🌐 Qdrant URL: {QDRANT_URL}")

# Validate environment on import
validate_environment()

if not GOOGLE_API_KEY:
    raise EnvironmentError("Missing GOOGLE_API_KEY in .env")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- Retry decorator ---
def retry_on_failure(max_retries=3, delay=1):
    """Decorator to retry functions on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# --- Model and Vector Store Initialization ---

# CLOUD: Gemini LLM + Google embeddings for Streamlit Cloud
try:
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    print("✅ Gemini LLM initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Gemini LLM: {e}")
    raise

try:
    embedding = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
    print("✅ Google embeddings initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Google embeddings: {e}")
    raise

# LOCAL: Ollama LLM + Ollama embeddings (uncomment to use locally)
# llm = init_chat_model("llama3", model_provider="ollama")
# embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

# CLOUD: Qdrant Cloud or public Qdrant
try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print(f"✅ Qdrant client initialized with URL: {QDRANT_URL}")
except Exception as e:
    print(f"❌ Failed to initialize Qdrant client: {e}")
    raise

# LOCAL: if using local Qdrant, same URL but without auth
# client = QdrantClient(url="http://localhost:6333")

def test_qdrant_connection():
    """
    Test Qdrant connection and return status information.
    Returns:
        dict: Connection status and collection info
    """
    try:
        # Test basic connection
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        # Check if our specific collection exists
        collection_exists = QDRANT_COLLECTION in collection_names
        
        # Get collection info if it exists
        collection_info = None
        if collection_exists:
            collection_info = client.get_collection(QDRANT_COLLECTION)
        
        return {
            "status": "connected",
            "collections": collection_names,
            "target_collection": QDRANT_COLLECTION,
            "target_exists": collection_exists,
            "collection_info": collection_info
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        }

def initialize_vectorstore_safely():
    """
    Initialize vectorstore with connection testing.
    Returns:
        tuple: (vectorstore, success_status)
    """
    try:
        # Test connection first
        connection_status = test_qdrant_connection()
        
        if connection_status["status"] != "connected":
            print(f"❌ Connection test failed: {connection_status.get('error', 'Unknown error')}")
            return None, False
            
        if not connection_status["target_exists"]:
            print(f"❌ Collection '{QDRANT_COLLECTION}' does not exist")
            print(f"Available collections: {connection_status['collections']}")
            return None, False
            
        # Initialize vectorstore
        vectorstore = Qdrant(
            collection_name=QDRANT_COLLECTION,
            embeddings=embedding,
            client=client
        )
        
        print(f"✅ Successfully connected to Qdrant collection '{QDRANT_COLLECTION}'")
        return vectorstore, True
        
    except Exception as e:
        print(f"❌ Failed to initialize vectorstore: {e}")
        return None, False

# Initialize vectorstore safely
vectorstore, connection_success = initialize_vectorstore_safely()

if not connection_success:
    print("❌ Failed to establish vectorstore connection")
    vectorstore = None  # Set to None for error handling in functions

# For debugging: print connection status
connection_status = test_qdrant_connection()
print(f"🔍 Connection status: {connection_status['status']}")

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

@retry_on_failure(max_retries=3, delay=2)
def retrieve_documents(query: str, k=None) -> str:
    """
    Retrieve relevant documents from Qdrant with robust error handling.
    Uses similarity search instead of MMR for better reliability with small datasets.

    Args:
        query (str): The user's search query.
        k (int, optional): Number of documents to retrieve. If None, uses get_dynamic_k().
    Returns:
        str: Concatenated string of retrieved document titles and content.
    """
    if not vectorstore:
        return "⚠️ Vector database is not available. Please check your connection and try again."
    
    k = k or get_dynamic_k(query)
    
    try:
        # Test connection first
        collections = client.get_collections()
        collection_exists = any(col.name == QDRANT_COLLECTION for col in collections.collections)
        
        if not collection_exists:
            print(f"❌ Collection '{QDRANT_COLLECTION}' does not exist")
            return "⚠️ Vector database collection not found. Please check your setup."
        
        # Perform similarity search
        docs = vectorstore.similarity_search(query, k=k)
        
        if not docs:
            return "🔍 No relevant documents found for your query. Try rephrasing or using different keywords."
            
        # Format results
        formatted_docs = []
        for doc in docs:
            title = doc.metadata.get('title', 'Untitled')
            content = doc.page_content.strip()
            if content:  # Only include non-empty content
                formatted_docs.append(f"📌 {title}\n{content}")
        
        if not formatted_docs:
            return "🔍 No relevant content found for your query."
            
        return "\n\n".join(formatted_docs)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Qdrant error in retrieve_documents: {type(e).__name__}: {error_msg}")
        
        # Handle specific error types
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return "🔌 Unable to connect to the vector database. Please check your connection and try again."
        elif "unauthorized" in error_msg.lower() or "forbidden" in error_msg.lower():
            return "🔐 Authentication failed. Please check your API credentials."
        elif "not found" in error_msg.lower():
            return "📂 Collection not found in the vector database."
        else:
            return f"⚠️ Search failed: {error_msg}. Please try again."

@retry_on_failure(max_retries=2, delay=1)
def generate_response(user_question: str, retrieved_content: str) -> str:
    """
    Generate a concise answer using the Gemini LLM, given the user question and retrieved content.
    Args:
        user_question (str): The user's question.
        retrieved_content (str): The context retrieved from Qdrant.
    Returns:
        str: The LLM's answer.
    """
    # Check if retrieved content indicates an error
    if retrieved_content.startswith(("⚠️", "🔌", "🔐", "📂", "🔍")):
        return retrieved_content
    
    try:
        system_prompt = (
            "You are an assistant for question-answering tasks about Vietnamese news. "
            "Use the following retrieved content to answer the user's question concisely and accurately. "
            "If the information is not sufficient to answer the question, say you don't know. "
            "Use three sentences maximum. Answer in Vietnamese if the question is in Vietnamese.\n\n"
            f"Retrieved content:\n{retrieved_content}"
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_question)
        ]
        
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        print(f"❌ LLM error in generate_response: {type(e).__name__}: {str(e)}")
        return f"⚠️ Failed to generate response: {str(e)}. Please try again."

from langsmith.run_helpers import traceable

@traceable(name="vnexpress_rag_manual_rag")
def run_pipeline(query: str = None):
    """
    End-to-end pipeline: retrieves relevant news articles and generates an answer.
    Args:
        query (str): User query. If None, uses a default example.
    """
    if not query:
        query = "Ủy ban Thường vụ Quốc hội đã làm gì ngày 10/7"
    
    print(f"\n🔍 Query: {query}")
    print("=" * 50)
    
    # Test connection before proceeding
    connection_status = test_qdrant_connection()
    if connection_status["status"] != "connected":
        print(f"❌ Cannot proceed: {connection_status.get('error', 'Connection failed')}")
        return
    
    print("🔎 Retrieving documents...")
    retrieved_text = retrieve_documents(query)
    
    if retrieved_text.startswith(("⚠️", "🔌", "🔐", "📂")):
        print(f"❌ Retrieval failed: {retrieved_text}")
        return
    
    print("🧠 Generating response...")
    final_answer = generate_response(query, retrieved_text)
    
    print("\n🧠 Final Answer:")
    print(final_answer)
    
    print(f"\n📚 Retrieved Content Preview:")
    print(retrieved_text[:500] + "..." if len(retrieved_text) > 500 else retrieved_text)

def get_system_status():
    """Get comprehensive system status for debugging."""
    status = {
        "environment": {
            "qdrant_url": QDRANT_URL,
            "collection": QDRANT_COLLECTION,
            "has_api_key": bool(QDRANT_API_KEY),
            "embedding_model": EMBEDDING_MODEL
        },
        "connection": test_qdrant_connection(),
        "vectorstore_initialized": vectorstore is not None
    }
    return status

if __name__ == "__main__":
    # Print system status
    print("🔍 System Status Check:")
    print("=" * 50)
    status = get_system_status()
    for category, details in status.items():
        print(f"{category.upper()}: {details}")
    print("=" * 50)
    
    # Run pipeline
    run_pipeline()