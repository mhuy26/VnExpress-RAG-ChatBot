# source/crawl.py
"""
Main pipeline orchestration for VNExpress RAG ingestion.
"""

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crawler.links import get_article_links
from crawler.content import crawl_single_article
from storage.embed import load_embedding_model, validate_vector_size
from storage.qdrant_store import initialize_vectorstore, upload_to_qdrant


def crawl_articles(urls: list) -> list:
    """Crawl full articles from a list of URLs and return enriched LangChain Documents."""
    headers = {"User-Agent": "Mozilla/5.0"}
    documents = []
    for url in urls:
        print(f"🔗 Crawling: {url}")
        doc = crawl_single_article(url, headers)
        if doc:
            documents.append(doc)
    return documents


def run():
    load_dotenv()
    print("🔍 Starting VNExpress RAG pipeline...")

    # Step 0: Reset Qdrant collection
    vectorstore, client = initialize_vectorstore(reset=True)

    # Step 1: Load article links
    links = get_article_links(max_links=15)
    if not links:
        print("❌ No links found. Exiting.")
        return

    # Step 2: Crawl content from URLs
    documents = crawl_articles(links)
    if not documents:
        print("❌ No articles crawled. Exiting.")
        return

    # Step 3: Load embedding model
    embedding_model, vector_size = load_embedding_model()

    # Step 4: Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 50]
    print(f"✂️ {len(chunks)} valid chunks from {len(documents)} articles")

    # Step 5: Validate chunk embeddings
    chunks = validate_vector_size(chunks, embedding_model, vector_size)
    if not chunks:
        print("❌ No valid chunks for upload. Exiting.")
        return

    # Step 6: Upload to Qdrant
    upload_to_qdrant(chunks, vectorstore)
    print("✅ Pipeline completed successfully.")


if __name__ == "__main__":
    run()
