"""
A script to crawl news articles, extract content, generate embeddings, and upload to Qdrant vector database.
Designed for extensibility and clarity.
"""

import os
import requests
from uuid import uuid4
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant
# Load environment variables from .env file
load_dotenv()


from langchain_google_genai import GoogleGenerativeAIEmbeddings  # CLOUD: use Gemini for embedding

def store_documents_to_qdrant(documents: list):
    """
    Store a list of Document objects into a Qdrant vector database.
    Args:
        documents (list): List of Document objects to store.
    """
    # Load configuration from environment variables
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "vnexpress_articles")
    vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", 768))
    distance = os.getenv("QDRANT_DISTANCE", "cosine").upper()
    embedding_model = os.getenv("OLLAMA_EMBEDDING", "nomic-embed-text")

    # CLOUD: Use Qdrant Cloud API key if provided
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    distance_enum = getattr(Distance, distance, Distance.COSINE)

    # LOCAL: use Qdrant running on localhost
    # client = QdrantClient(url=qdrant_url)

    # CLOUD: use Qdrant Cloud with optional API key
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key  # Leave None for localhost
    )

    # LOCAL: use Ollama for local embedding (not supported on Streamlit Cloud)
    # from langchain_community.embeddings import OllamaEmbeddings
    # embedding = OllamaEmbeddings(model=embedding_model)

    # CLOUD: use Google embeddings for compatibility with Streamlit Cloud
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # LOCAL: always recreate collection (dangerous in production)
    # client.recreate_collection(
    #     collection_name=collection_name,
    #     vectors_config=VectorParams(size=vector_size, distance=distance_enum)
    # )

    # CLOUD: only create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance_enum)
        )

    # Split text and generate embeddings
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    ids = [str(uuid4()) for _ in splits]

    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embedding
    )

    vectorstore.add_documents(
        documents=splits,
        ids=ids
    )
    print(f"[Success] Stored {len(splits)} chunks to Qdrant collection: {collection_name}")
