# source/storage/qdrant_store.py

import os
import time
from uuid import uuid4
from crawler.utils import validate_env
from storage.embed import load_embedding_model
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

def initialize_vectorstore():
    """Load embedding model and initialize Qdrant vector store."""
    embedding_model, vector_size = load_embedding_model()
    vectorstore = connect_to_qdrant(vector_size, embedding_model)
    return vectorstore, vectorstore.client  # Use .client instead of ._client

def connect_to_qdrant(vector_size, embedding_model):
    """Connect to Qdrant and ensure the collection exists with correct vector settings."""
    validate_env(["QDRANT_URL", "QDRANT_COLLECTION"])

    url = os.getenv("QDRANT_URL")
    collection = os.getenv("QDRANT_COLLECTION")
    api_key = os.getenv("QDRANT_API_KEY")
    distance = os.getenv("QDRANT_DISTANCE", "cosine").upper()
    distance_enum = getattr(Distance, distance, Distance.COSINE)

    client = QdrantClient(url=url, api_key=api_key)
    collections = client.get_collections()
    print(f"🔗 Target Qdrant collection: {collection}")

    # Create or validate collection
    if collection not in [c.name for c in collections.collections]:
        print("📁 Creating new Qdrant collection...")
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=distance_enum)
        )
    else:
        info = client.get_collection(collection)
        current_size = info.config.params.vectors.size
        if current_size != vector_size:
            print(f"⚠️ Vector size mismatch: {current_size} vs {vector_size}. Recreating collection...")
            client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vector_size, distance=distance_enum)
            )

    print(f"✅ Connected to Qdrant collection: {collection}")
    
    # Fix: Use the correct constructor instead of from_client
    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embedding_model
    )


def upload_to_qdrant(documents, embedding_model, vector_size):
    """Upload document chunks to Qdrant in batches."""
    if not documents:
        print("❌ No documents to upload")
        return

    vectorstore = connect_to_qdrant(vector_size, embedding_model)

    ids = [str(uuid4()) for _ in documents]
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        try:
            vectorstore.add_documents(documents=batch, ids=batch_ids)
            print(f"✅ Uploaded batch {i // batch_size + 1} ({len(batch)} items)")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Failed batch {i // batch_size + 1}: {e}")