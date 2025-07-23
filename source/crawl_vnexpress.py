"""
VNExpress News Crawler and Qdrant Uploader
------------------------------------------
This script crawls news articles from VNExpress, extracts and chunks their content,
generates embeddings using Gemini, and uploads them to a Qdrant vector database.

Features:
- Robust error handling and logging
- Extensible for other sources or embedding models
- Designed for production and research workflows
"""

# --- Standard Library Imports ---
import os
import requests
from uuid import uuid4
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# --- LangChain & Qdrant Imports ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Gemini embeddings

# --- Load environment variables ---
load_dotenv()


def extract_article_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article.fck_detail")
    return article.get_text(separator="\n").strip() if article else ""


def get_article_links() -> list:
    url = "https://vnexpress.net/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "lxml")
    links = []
    for a_tag in soup.select("h3.title-news a"):
        href = a_tag.get("href")
        if href and href.startswith("https://vnexpress.net"):
            links.append(href)
    return list(set(links))


def crawl_articles(url_list: list) -> list:
    documents = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for url in url_list:
        try:
            html = requests.get(url, headers=headers).text
            soup = BeautifulSoup(html, "html.parser")
            content = extract_article_content(html)
            title = soup.title.string.strip() if soup.title else ""
            desc_tag = soup.find("meta", attrs={"name": "description"})
            description = desc_tag["content"].strip() if desc_tag and "content" in desc_tag.attrs else ""
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": url,
                    "language": "vi",
                    "title": title,
                    "description": description
                }
            ))
        except Exception as e:
            print(f"[Error] {url} - {e}")
    return documents


def store_documents_to_qdrant(documents: list, google_api_key: str):
    print("🔧 Starting Qdrant upload process...")

    # --- Load configuration ---
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "vnexpress_articles")
    distance = os.getenv("QDRANT_DISTANCE", "cosine").upper()
    distance_enum = getattr(Distance, distance, Distance.COSINE)

    print(f"🌐 Qdrant URL: {qdrant_url}")
    print(f"🔑 Using API Key: {'Yes' if qdrant_api_key else 'No'}")
    print(f"📦 Collection: {collection_name}")
    print(f"📐 Distance: {distance}")

    # --- Connect to Qdrant ---
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print("✅ Connected to Qdrant.")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return

    # --- Initialize Embeddings and infer vector size ---
    try:
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        test_vector = embedding.embed_query("test")
        vector_size = len(test_vector)
        print("✅ Initialized Gemini embeddings.")
        print(f"📏 Inferred Vector Size: {vector_size}")
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        return

    # --- Create collection if needed ---
    try:
        if not client.collection_exists(collection_name):
            print("📁 Collection does not exist. Creating...")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance_enum)
            )
            print("✅ Collection created.")
        else:
            print("📁 Collection already exists.")
    except Exception as e:
        print(f"❌ Failed to create/check collection: {e}")
        return

    # --- Split documents ---
    print("📄 Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    print(f"✂️ Total chunks: {len(splits)}")

    # --- Upload to Qdrant ---
    ids = [str(uuid4()) for _ in splits]
    vectorstore = Qdrant(client=client, collection_name=collection_name, embeddings=embedding)

    print("📤 Uploading chunks to Qdrant...")
    try:
        vectorstore.add_documents(documents=splits, ids=ids)
        print(f"✅ Uploaded {len(splits)} documents to Qdrant collection '{collection_name}'")
    except Exception as e:
        print(f"❌ Failed to upload documents to Qdrant: {e}")


# --- Main script ---
if __name__ == "__main__":
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env")

    article_links = get_article_links()
    print(f"🔗 Found {len(article_links)} article links.")
    documents = crawl_articles(article_links[:5])
    print(f"📰 Retrieved {len(documents)} articles.")
    store_documents_to_qdrant(documents, google_api_key)
    print("✅ Crawling and storing process completed.")
