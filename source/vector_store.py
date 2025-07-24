from typing import Optional, TypedDict
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant

from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION
from model import load_embeddings

class RetrievalResult(TypedDict):
    success: bool
    content: str
    error_type: Optional[str]

def test_qdrant_connection(client: QdrantClient) -> RetrievalResult:
    try:
        collections = client.get_collections()
        names = [col.name for col in collections.collections]
        if QDRANT_COLLECTION not in names:
            return {"success": False, "content": f"Collection '{QDRANT_COLLECTION}' not found.", "error_type": "CollectionNotFound"}
        return {"success": True, "content": f"Connected to collection '{QDRANT_COLLECTION}'", "error_type": None}
    except Exception as e:
        return {"success": False, "content": str(e), "error_type": type(e).__name__}

def initialize_vectorstore():
    embedding = load_embeddings()
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    status = test_qdrant_connection(client)
    if not status["success"]:
        print(f"❌ Qdrant connection failed: {status['content']}")
        return None, None
    vs = Qdrant(collection_name=QDRANT_COLLECTION, embeddings=embedding, client=client)
    return vs, client
