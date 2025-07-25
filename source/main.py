# source/main.py

import time
from functools import wraps
from typing import Optional, Generator
from langchain_core.messages import SystemMessage, HumanMessage
from config import validate_environment
from model import load_llm
from storage.qdrant_store import initialize_vectorstore
from langchain_qdrant import QdrantVectorStore

llm = load_llm()

# --- Retry Decorator ---
def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    print(f"⚠️ Retry {attempt + 1} failed: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

# --- Document Retrieval ---
def get_dynamic_k(user_query: str) -> int:
    if any(k in user_query.lower() for k in ["summary", "tổng hợp", "tất cả", "toàn bộ", "tổng quan", "all", "overview"]):
        return 100
    return 5

@retry_on_failure()
def retrieve_documents(query: str, vectorstore, client, k: Optional[int] = None):
    if not vectorstore:
        return {
            "success": False,
            "content": "Vectorstore unavailable",
            "error_type": "VectorstoreUnavailable"
        }

    k = k or get_dynamic_k(query)
    try:
        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            return {
                "success": False,
                "content": "No relevant documents found",
                "error_type": "NoResults"
            }

        formatted = [
            f"📌 {doc.metadata.get('title', 'Untitled')}\n{doc.page_content.strip()}"
            for doc in docs if doc.page_content.strip()
        ]
        return {
            "success": True,
            "content": "\n\n".join(formatted),
            "error_type": None
        }

    except Exception as e:
        return {
            "success": False,
            "content": str(e),
            "error_type": type(e).__name__
        }

# --- Streamed Response Generator ---
def generate_response_stream(user_question: str, retrieved_content: str) -> Generator[str, None, None]:
    try:
        system_prompt = (
            "You are an assistant for question-answering tasks about Vietnamese news. "
            "Use the following retrieved content to answer the user's question concisely and accurately. "
            "If the information is not sufficient to answer the question, say you don't know. "
            "Answer in the same language as the user used."
            "Keep your answer concise but informative. Avoid overly short or overly verbose responses."
            "Do not add information that is not in the retrieved content.\n\n"
            f"Retrieved content:\n{retrieved_content}"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_question)
        ]
        for chunk in llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"\n❌ Streaming failed: {e}"

# --- Combined Pipeline & App ---
def run_pipeline_stream(query: str = "Ủy ban Thường vụ Quốc hội đã làm gì ngày 10/7"):
    print(f"\n🔍 Query: {query}")
    print("=" * 50)

    vectorstore, client = initialize_vectorstore()
    retrieval = retrieve_documents(query, vectorstore, client)

    if not retrieval["success"]:
        print(f"❌ Retrieval failed: {retrieval['content']}")
        return

    print("🧠 Streaming Gemini response:")
    for chunk in generate_response_stream(query, retrieval["content"]):
        print(chunk, end="", flush=True)

    print("\n\n📚 Retrieved Context (preview):")
    preview = retrieval["content"]
    print(preview[:500] + "..." if len(preview) > 500 else preview)

if __name__ == "__main__":
    validate_environment()
    run_pipeline_stream()