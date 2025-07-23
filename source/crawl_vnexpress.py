"""
VNExpress News Crawler and Qdrant Uploader
------------------------------------------
This script crawls news articles from VNExpress, extracts and chunks their content,
generates embeddings using Gemini, and uploads them to a Qdrant vector database.

Features:
- Robust error handling and logging
- Extensible for other sources or embedding models
- Designed for production and research workflows
- Connection testing and validation
"""

# --- Standard Library Imports ---
import os
import requests
import time
from uuid import uuid4
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from functools import wraps

# --- LangChain & Qdrant Imports ---
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- Load environment variables ---
load_dotenv()

def validate_environment():
    """Validate all required environment variables are present."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    
    if not google_api_key:
        raise ValueError("Missing GOOGLE_API_KEY in .env file")
    
    print("✅ Environment validation passed")
    print(f"🌐 Qdrant URL: {qdrant_url}")
    print(f"🔑 Google API Key: {'Set' if google_api_key else 'Not set'}")
    
    return google_api_key, qdrant_url

def retry_on_failure(max_retries=3, delay=2):
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

def extract_article_content(html: str) -> str:
    """Extract article content from VNExpress HTML."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        article = soup.select_one("article.fck_detail")
        if article:
            # Remove unwanted elements
            for element in article.select('script, style, .ads, .advertisement'):
                element.decompose()
            return article.get_text(separator="\n").strip()
        return ""
    except Exception as e:
        print(f"⚠️ Error extracting content: {e}")
        return ""

@retry_on_failure(max_retries=2, delay=1)
def get_article_links(max_links=None) -> list:
    """Get article links from VNExpress homepage."""
    print("🔗 Fetching article links from VNExpress...")
    
    try:
        url = "https://vnexpress.net/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "lxml")
        links = []
        
        # Get links from various sections
        selectors = [
            "h3.title-news a",
            "h2.title-news a", 
            "h4.title-news a",
            ".item-news h3 a",
            ".item-news h2 a"
        ]
        
        for selector in selectors:
            for a_tag in soup.select(selector):
                href = a_tag.get("href")
                if href and href.startswith("https://vnexpress.net") and "/video-" not in href:
                    links.append(href)
        
        # Remove duplicates while preserving order
        unique_links = list(dict.fromkeys(links))
        
        if max_links:
            unique_links = unique_links[:max_links]
        
        print(f"✅ Found {len(unique_links)} unique article links")
        return unique_links
        
    except Exception as e:
        print(f"❌ Failed to get article links: {e}")
        return []

@retry_on_failure(max_retries=2, delay=1)
def crawl_single_article(url: str, headers: dict) -> Document:
    """Crawl a single article and return a Document object."""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract content
        content = extract_article_content(response.text)
        if not content or len(content.strip()) < 100:  # Skip very short articles
            print(f"⚠️ Skipping short article: {url}")
            return None
        
        # Extract metadata
        title = ""
        if soup.title:
            title = soup.title.string.strip()
        
        description = ""
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and "content" in desc_tag.attrs:
            description = desc_tag["content"].strip()
        
        # Extract publish date if available
        publish_date = ""
        date_tag = soup.find("span", class_="date")
        if date_tag:
            publish_date = date_tag.get_text().strip()
        
        # Extract category if available
        category = ""
        breadcrumb = soup.find("ul", class_="breadcrumb")
        if breadcrumb:
            category_links = breadcrumb.find_all("a")
            if category_links:
                category = category_links[-1].get_text().strip()
        
        document = Document(
            page_content=content,
            metadata={
                "source": url,
                "language": "vi",
                "title": title,
                "description": description,
                "publish_date": publish_date,
                "category": category,
                "content_length": len(content)
            }
        )
        
        return document
        
    except Exception as e:
        print(f"❌ Error crawling {url}: {e}")
        return None

def crawl_articles(url_list: list, max_workers=5) -> list:
    """Crawl multiple articles with improved error handling."""
    print(f"📰 Starting to crawl {len(url_list)} articles...")
    
    documents = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    successful_crawls = 0
    failed_crawls = 0
    
    for i, url in enumerate(url_list, 1):
        print(f"📄 Crawling article {i}/{len(url_list)}: {url[:80]}...")
        
        try:
            document = crawl_single_article(url, headers)
            if document:
                documents.append(document)
                successful_crawls += 1
                print(f"✅ Successfully crawled: {document.metadata.get('title', 'No title')[:50]}...")
            else:
                failed_crawls += 1
            
            # Add delay to be respectful to the server
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ Failed to crawl {url}: {e}")
            failed_crawls += 1
    
    print(f"📊 Crawling completed: {successful_crawls} successful, {failed_crawls} failed")
    return documents

def assert_vector_size(docs, embedding, expected_dim):
    """
    Check if all documents produce vectors of the expected dimension.
    Returns the filtered list of valid documents.
    """
    print("🔍 Validating vector dimensions...")
    valid_docs = []
    invalid_count = 0
    
    for i, doc in enumerate(docs):
        try:
            # Test with first 100 characters to save API calls
            test_text = doc.page_content[:100] if doc.page_content else "test"
            vec = embedding.embed_query(test_text)
            
            if len(vec) != expected_dim:
                print(f"❌ Chunk {i} has invalid vector size {len(vec)}, expected {expected_dim}")
                invalid_count += 1
            else:
                valid_docs.append(doc)
                
        except Exception as e:
            print(f"⚠️ Error embedding chunk {i}: {e}")
            invalid_count += 1
    
    print(f"✅ {len(valid_docs)} of {len(docs)} chunks passed vector validation")
    if invalid_count > 0:
        print(f"⚠️ {invalid_count} chunks failed validation")
    
    return valid_docs

def test_qdrant_connection(client, collection_name):
    """Test Qdrant connection and collection status."""
    try:
        # Test basic connection
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        print(f"✅ Connected to Qdrant successfully")
        print(f"📋 Available collections: {collection_names}")
        
        # Check if target collection exists
        collection_exists = collection_name in collection_names
        
        if collection_exists:
            info = client.get_collection(collection_name)
            print(f"📊 Collection '{collection_name}' exists with {info.points_count} documents")
            return True, info
        else:
            print(f"📂 Collection '{collection_name}' does not exist - will be created")
            return False, None
            
    except Exception as e:
        print(f"❌ Qdrant connection test failed: {e}")
        raise

def store_documents_to_qdrant(documents: list, google_api_key: str):
    """Store documents to Qdrant with comprehensive error handling."""
    print("🔧 Starting Qdrant upload process...")
    
    if not documents:
        print("❌ No documents to upload")
        return
    
    # --- Load configuration ---
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "vnexpress_articles")
    distance = os.getenv("QDRANT_DISTANCE", "cosine").upper()
    
    try:
        distance_enum = getattr(Distance, distance, Distance.COSINE)
    except AttributeError:
        print(f"⚠️ Invalid distance metric '{distance}', using COSINE")
        distance_enum = Distance.COSINE

    print(f"🌐 Qdrant URL: {qdrant_url}")
    print(f"🔑 Using API Key: {'Yes' if qdrant_api_key else 'No'}")
    print(f"📦 Collection: {collection_name}")
    print(f"📐 Distance: {distance}")

    # --- Connect to Qdrant ---
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        collection_exists, collection_info = test_qdrant_connection(client, collection_name)
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return

    # --- Initialize Embeddings and infer vector size ---
    try:
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Test embedding to get vector size
        test_vector = embedding.embed_query("test")
        vector_size = len(test_vector)
        print("✅ Initialized Gemini embeddings")
        print(f"📏 Vector dimension: {vector_size}")
        
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        return

    # --- Create or verify collection ---
    try:
        if not collection_exists:
            print("📁 Creating new collection...")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance_enum)
            )
            print("✅ Collection created successfully")
        else:
            # Verify vector dimensions match
            if collection_info.config.params.vectors.size != vector_size:
                print(f"⚠️ Vector size mismatch! Collection: {collection_info.config.params.vectors.size}, Model: {vector_size}")
                print("🔄 Recreating collection with correct dimensions...")
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance_enum)
                )
                print("✅ Collection recreated with correct dimensions")
            else:
                print("✅ Collection dimensions match")
                
    except Exception as e:
        print(f"❌ Failed to create/verify collection: {e}")
        return

    # --- Split documents ---
    print("📄 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )
    
    try:
        splits = splitter.split_documents(documents)
        print(f"✂️ Created {len(splits)} chunks from {len(documents)} documents")
        
        # Filter out very short chunks
        valid_splits = [doc for doc in splits if len(doc.page_content.strip()) >= 50]
        print(f"📝 {len(valid_splits)} chunks after filtering short content")
        
        if not valid_splits:
            print("❌ No valid chunks to upload after filtering")
            return
            
    except Exception as e:
        print(f"❌ Error splitting documents: {e}")
        return

    # --- Validate vector dimensions ---
    valid_splits = assert_vector_size(valid_splits, embedding, vector_size)
    if not valid_splits:
        print("❌ No valid documents to upload after vector validation")
        return

    # --- Upload to Qdrant ---
    try:
        print("📤 Uploading chunks to Qdrant...")
        
        # Generate unique IDs
        ids = [str(uuid4()) for _ in valid_splits]
        
        # Initialize vectorstore
        vectorstore = Qdrant(
            client=client, 
            collection_name=collection_name, 
            embeddings=embedding
        )
        
        # Upload in batches to handle large datasets
        batch_size = 10
        total_batches = (len(valid_splits) + batch_size - 1) // batch_size
        
        for i in range(0, len(valid_splits), batch_size):
            batch = valid_splits[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"📤 Uploading batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            try:
                vectorstore.add_documents(documents=batch, ids=batch_ids)
                print(f"✅ Batch {batch_num} uploaded successfully")
                time.sleep(1)  # Small delay between batches
                
            except Exception as e:
                print(f"❌ Failed to upload batch {batch_num}: {e}")
                continue
        
        # Verify upload
        final_info = client.get_collection(collection_name)
        print(f"✅ Upload completed! Collection now has {final_info.points_count} documents")
    except Exception as e:
        print(f"❌ Failed to upload to Qdrant: {e}")
        return []

if __name__ == "__main__":
    try:
        # Step 1: Validate environment
        google_api_key, qdrant_url = validate_environment()

        # Step 2: Fetch article links
        article_links = get_article_links(max_links=15)  # Adjust max_links as needed
        if not article_links:
            print("❌ No article links found. Exiting.")
            exit()

        # Step 3: Crawl articles
        documents = crawl_articles(article_links)
        if not documents:
            print("❌ No articles successfully crawled. Exiting.")
            exit()

        # Step 4: Upload to Qdrant
        store_documents_to_qdrant(documents, google_api_key)

    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")
