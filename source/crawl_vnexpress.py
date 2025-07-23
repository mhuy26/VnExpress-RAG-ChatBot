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

def extract_article_content(html: str) -> str:
    """
    Extract the main article content from the provided HTML string.
    Args:
        html (str): HTML content of the page.
    Returns:
        str: Extracted article text, or empty string if not found.
    """
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article.fck_detail")
    return article.get_text(separator="\n").strip() if article else ""

def get_article_links() -> list:
    """
    Fetch article links from the VnExpress homepage.
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
    Crawl a list of article URLs and extract their content and metadata.
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
            # Extract meta description if available
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
    
    distance_enum = getattr(Distance, distance, Distance.COSINE)
    client = QdrantClient(url=qdrant_url)
    embedding = OllamaEmbeddings(model=embedding_model)
    # Recreate collection (deletes if exists)
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

def main():
    """
    Main function to crawl articles and store them in Qdrant.
    """
    article_links = get_article_links()
    print(f"[Info] Found {len(article_links)} links")
    documents = crawl_articles(article_links[:5])  # Limit to top 5 for demo
    print(f"[Info] Crawled {len(documents)} articles")
    store_documents_to_qdrant(documents)

if __name__ == "__main__":
    main()
