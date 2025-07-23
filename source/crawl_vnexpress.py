
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
    """
    Extracts the main content from a VNExpress article HTML.
    Args:
        html (str): Raw HTML of the article page.
    Returns:
        str: Cleaned article text, or empty string if not found.
    """
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article.fck_detail")
    return article.get_text(separator="\n").strip() if article else ""


def get_article_links() -> list:
    """
    Fetches article links from the VNExpress homepage.
    Returns:
        list: List of unique article URLs.
    """
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
    """
    Crawls a list of article URLs and extracts their content and metadata.
    Args:
        url_list (list): List of article URLs to crawl.
    Returns:
        list: List of Document objects with content and metadata.
    """
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


def store_documents_to_qdrant(documents: list):
    """
    Stores a list of Document objects into a Qdrant vector database.
    Handles collection creation, embedding, and chunk upload.
    Args:
        documents (list): List of Document objects to store.
    """
    print("🔧 Starting Qdrant upload process...")

    # --- Load configuration ---
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "vnexpress_articles")
    vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", 768))
    distance = os.getenv("QDRANT_DISTANCE", "cosine").upper()
    embedding_model = os.getenv("OLLAMA_EMBEDDING", "nomic-embed-text")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    distance_enum = getattr(Distance, distance, Distance.COSINE)

    print(f"🌐 Qdrant URL: {qdrant_url}")
    print(f"🔑 Using API Key: {'Yes' if qdrant_api_key else 'No'}")
    print(f"📦 Collection: {collection_name}")
    print(f"📏 Vector Size: {vector_size}")
    print(f"📐 Distance: {distance}")

    # --- Connect to Qdrant ---
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print("✅ Connected to Qdrant.")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return

    # --- Initialize Embeddings ---
    try:
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        print("✅ Initialized Gemini embeddings.")
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        return

    # --- Create or check Qdrant collection ---
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

    # --- Split documents into chunks ---
    print("📄 Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    print(f"✂️ Total chunks: {len(splits)}")

    # --- Upload chunks to Qdrant ---
    ids = [str(uuid4()) for _ in splits]
    vectorstore = Qdrant(client=client, collection_name=collection_name, embeddings=embedding)

    print("📤 Uploading chunks to Qdrant...")
    try:
        vectorstore.add_documents(documents=splits, ids=ids)
        print(f"✅ Uploaded {len(splits)} documents to Qdrant collection '{collection_name}'")
    except Exception as e:
        print(f"❌ Failed to upload documents to Qdrant: {e}")


# --- Main script entry point ---
if __name__ == "__main__":
    # Crawl and store top 5 articles for demo
    article_links = get_article_links()
    print(f"🔗 Found {len(article_links)} article links.")
    documents = crawl_articles(article_links[:5])
    print(f"📰 Retrieved {len(documents)} articles.")
    store_documents_to_qdrant(documents)
    print("✅ Crawling and storing process completed.")