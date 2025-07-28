# source/crawl.py
"""
Enhanced pipeline orchestration for VNExpress RAG ingestion with Playwright crawler.
"""

import os
import json
import logging
from typing import List, Optional
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from crawler.links import get_article_links
from crawler.playwright_crawler import crawl_articles as crawl_articles_playwright
from storage.embed import load_embedding_model, validate_vector_size
from storage.qdrant_store import initialize_vectorstore, upload_to_qdrant

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_crawled_data(documents: List[Document], filename: str = "crawled_articles.json"):
    """Save crawled documents for backup/debugging"""
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    
    data = []
    for doc in documents:
        data.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Saved {len(documents)} documents to {filepath}")


def load_cached_articles(filename: str = "crawled_articles.json") -> Optional[List[Document]]:
    """Load previously crawled articles from cache"""
    filepath = os.path.join("data", filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        for item in data:
            doc = Document(
                page_content=item["content"],
                metadata=item["metadata"]
            )
            documents.append(doc)
        
        logger.info(f"📂 Loaded {len(documents)} cached articles from {filepath}")
        return documents
    
    except Exception as e:
        logger.error(f"❌ Failed to load cached articles: {e}")
        return None


def filter_and_validate_documents(documents: List[Document], min_length: int = 100) -> List[Document]:
    """Filter out documents with insufficient content"""
    valid_docs = []
    
    for doc in documents:
        content = doc.page_content.strip()
        
        # Skip if content is too short
        if len(content) < min_length:
            logger.warning(f"⚠️ Skipping short document: {len(content)} chars from {doc.metadata.get('source', 'unknown')}")
            continue
        
        # Skip if content seems to be blocked/error page
        blocked_indicators = [
            "access denied",
            "403 forbidden",
            "404 not found",
            "please enable javascript",
            "cloudflare",
            "captcha",
            "blocked"
        ]
        
        content_lower = content.lower()
        if any(indicator in content_lower for indicator in blocked_indicators):
            logger.warning(f"⚠️ Skipping potentially blocked content from {doc.metadata.get('source', 'unknown')}")
            continue
        
        valid_docs.append(doc)
    
    logger.info(f"✅ {len(valid_docs)}/{len(documents)} documents passed validation")
    return valid_docs


def enhance_document_metadata(documents: List[Document]) -> List[Document]:
    """Add additional metadata to documents"""
    for doc in documents:
        # Add word count
        word_count = len(doc.page_content.split())
        doc.metadata["word_count"] = word_count
        
        # Add estimated reading time (average 200 words per minute)
        reading_time = max(1, word_count // 200)
        doc.metadata["estimated_reading_time_minutes"] = reading_time
        
        # Extract domain from source URL
        source_url = doc.metadata.get("source", "")
        if source_url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(source_url).netloc
                doc.metadata["domain"] = domain
            except:
                doc.metadata["domain"] = "unknown"
        
        # Add processing timestamp if not present
        if "timestamp" not in doc.metadata:
            from datetime import datetime
            doc.metadata["timestamp"] = datetime.now().isoformat()
    
    return documents


def create_intelligent_chunks(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Create chunks with intelligent splitting based on content structure"""
    
    # Define separators prioritizing Vietnamese text structure
    vietnamese_separators = [
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ". ",    # Sentence endings
        "! ",    # Exclamation
        "? ",    # Question
        "; ",    # Semicolon
        ", ",    # Comma
        " ",     # Space
        ""       # Character level
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=vietnamese_separators,
        keep_separator=True,
        add_start_index=True
    )
    
    all_chunks = []
    
    for doc in documents:
        try:
            chunks = splitter.split_documents([doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
                chunk.metadata["chunk_id"] = f"{doc.metadata.get('source', 'unknown')}_{i}"
                
                # Calculate chunk position in original document
                if "start_index" in chunk.metadata:
                    original_length = len(doc.page_content)
                    position_percentage = (chunk.metadata["start_index"] / original_length) * 100
                    chunk.metadata["position_in_document"] = round(position_percentage, 2)
            
            all_chunks.extend(chunks)
            
        except Exception as e:
            logger.error(f"❌ Failed to chunk document from {doc.metadata.get('source', 'unknown')}: {e}")
            continue
    
    # Filter out chunks that are too short
    valid_chunks = [chunk for chunk in all_chunks if len(chunk.page_content.strip()) >= 50]
    
    logger.info(f"✂️ Created {len(valid_chunks)} valid chunks from {len(documents)} documents")
    return valid_chunks


def run(use_cache: bool = False, max_links: int = 15, reset_collection: bool = True):
    """
    Main pipeline execution
    
    Args:
        use_cache: Whether to use previously crawled articles
        max_links: Maximum number of articles to crawl
        reset_collection: Whether to reset the Qdrant collection
    """
    load_dotenv()
    logger.info("🚀 Starting Enhanced VNExpress RAG pipeline...")

    # Step 0: Initialize Qdrant
    try:
        vectorstore, client = initialize_vectorstore(reset=reset_collection)
        logger.info("✅ Qdrant vectorstore initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Qdrant: {e}")
        return

    # Step 1: Get documents (either from cache or fresh crawl)
    documents = None
    
    if use_cache:
        documents = load_cached_articles()
    
    if not documents:
        logger.info(f"🔍 Getting article links (max: {max_links})...")
        
        # Get article links
        try:
            links = get_article_links(max_links=max_links)
            if not links:
                logger.error("❌ No links found. Exiting.")
                return
            
            logger.info(f"📝 Found {len(links)} article links")
        except Exception as e:
            logger.error(f"❌ Failed to get article links: {e}")
            return

        # Crawl articles using Playwright
        logger.info("🕸️ Starting enhanced crawling with Playwright...")
        try:
            documents = crawl_articles_playwright(
                links, 
                delay_range=(3, 8)  # 3-8 second delays between requests
            )
            
            if not documents:
                logger.error("❌ No articles crawled successfully. Exiting.")
                return
            
            logger.info(f"📄 Successfully crawled {len(documents)} articles")
            
            # Save crawled data for future use
            save_crawled_data(documents)
            
        except Exception as e:
            logger.error(f"❌ Crawling failed: {e}")
            return

    # Step 2: Validate and filter documents
    logger.info("🔍 Validating and filtering documents...")
    documents = filter_and_validate_documents(documents, min_length=200)
    
    if not documents:
        logger.error("❌ No valid documents after filtering. Exiting.")
        return

    # Step 3: Enhance metadata
    logger.info("📊 Enhancing document metadata...")
    documents = enhance_document_metadata(documents)

    # Step 4: Load embedding model
    logger.info("🤖 Loading embedding model...")
    try:
        embedding_model, vector_size = load_embedding_model()
        logger.info(f"✅ Embedding model loaded (vector size: {vector_size})")
    except Exception as e:
        logger.error(f"❌ Failed to load embedding model: {e}")
        return

    # Step 5: Create intelligent chunks
    logger.info("✂️ Creating document chunks...")
    try:
        chunks = create_intelligent_chunks(documents, chunk_size=1000, chunk_overlap=200)
        
        if not chunks:
            logger.error("❌ No chunks created. Exiting.")
            return
            
        # Log chunking statistics
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)
        logger.info(f"📊 Chunk statistics: {len(chunks)} chunks, avg size: {avg_chunk_size:.0f} chars")
        
    except Exception as e:
        logger.error(f"❌ Failed to create chunks: {e}")
        return

    # Step 6: Validate embeddings
    logger.info("🔬 Validating chunk embeddings...")
    try:
        chunks = validate_vector_size(chunks, embedding_model, vector_size)
        
        if not chunks:
            logger.error("❌ No valid chunks after embedding validation. Exiting.")
            return
            
        logger.info(f"✅ {len(chunks)} chunks passed embedding validation")
        
    except Exception as e:
        logger.error(f"❌ Embedding validation failed: {e}")
        return

    # Step 7: Upload to Qdrant
    logger.info("⬆️ Uploading to Qdrant...")
    try:
        upload_to_qdrant(chunks, vectorstore, use_point_ids=True)
        
        # Log final statistics
        total_docs = len(documents)
        total_chunks = len(chunks)
        success_rate = (total_docs / max_links) * 100 if max_links > 0 else 0
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📊 Final statistics:")
        logger.info(f"   • Articles crawled: {total_docs}/{max_links} ({success_rate:.1f}%)")
        logger.info(f"   • Chunks created: {total_chunks}")
        logger.info(f"   • Average chunks per article: {total_chunks/total_docs:.1f}")
        
    except Exception as e:
        logger.error(f"❌ Failed to upload to Qdrant: {e}")
        return


def run_with_custom_urls(urls: List[str], reset_collection: bool = True):
    """
    Run pipeline with custom URLs instead of fetching from VNExpress
    
    Args:
        urls: List of URLs to crawl
        reset_collection: Whether to reset the Qdrant collection
    """
    load_dotenv()
    logger.info(f"🚀 Starting pipeline with {len(urls)} custom URLs...")

    # Initialize Qdrant
    vectorstore, client = initialize_vectorstore(reset=reset_collection)

    # Crawl articles
    documents = crawl_articles_playwright(urls, delay_range=(3, 8))
    
    if not documents:
        logger.error("❌ No articles crawled successfully.")
        return

    # Process documents through the pipeline
    documents = filter_and_validate_documents(documents)
    documents = enhance_document_metadata(documents)
    
    embedding_model, vector_size = load_embedding_model()
    chunks = create_intelligent_chunks(documents)
    chunks = validate_vector_size(chunks, embedding_model, vector_size)
    
    if chunks:
        upload_to_qdrant(chunks, vectorstore)
        logger.info("✅ Custom URL pipeline completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced VNExpress RAG Pipeline')
    parser.add_argument('--use-cache', action='store_true', help='Use cached articles instead of crawling')
    parser.add_argument('--max-links', type=int, default=15, help='Maximum number of articles to crawl')
    parser.add_argument('--no-reset', action='store_true', help='Do not reset Qdrant collection')
    
    args = parser.parse_args()
    
    run(
        use_cache=args.use_cache,
        max_links=args.max_links,
        reset_collection=not args.no_reset
    )