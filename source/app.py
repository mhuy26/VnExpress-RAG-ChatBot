# source/app.py

import asyncio
import sys
import time
import streamlit as st
from datetime import datetime

# --- Fix for grpc.aio "no event loop" in Streamlit ---
if sys.platform == "darwin":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Cache-based timestamp store (survives refresh) ---
@st.cache_data
def get_last_crawled():
    return None  # Initially empty

@st.cache_data
def update_last_crawled():
    return datetime.now().isoformat()

# --- Page Setup ---
st.set_page_config(page_title="VNExpress RAG QA", layout="wide")
st.title("📰 VNExpress Chatbot")
st.markdown("Ask questions about the current VNExpress articles.")

# --- News Refresh Section ---
st.subheader("🕸️ VNExpress Data Update")

# Show last updated timestamp
last_crawled = get_last_crawled()
if last_crawled:
    st.markdown(f"🕒 Last updated: `{last_crawled}`")
else:
    st.markdown("🕒 Last updated: *not yet this session*")

# Run crawler on button click
if st.button("📰 Get News"):
    from crawl import run as crawl_news
    with st.spinner("🕷️ Crawling VNExpress and updating vector database..."):
        try:
            crawl_news()
            update_last_crawled.clear()  # Clear cache
            new_time = update_last_crawled()  # Re-cache new time
            st.success(f"✅ News updated at `{new_time}`")
        except Exception as e:
            st.error(f"❌ Failed to crawl news: {e}")

# --- Environment Setup ---
from config import validate_environment
from storage.qdrant_store import initialize_vectorstore
from main import retrieve_documents, generate_response_stream

validate_environment()
vectorstore, client = initialize_vectorstore()

# --- User Input ---
query = st.text_input(
    "💬 Ask your question:",
    placeholder="e.g. Hãy hỏi tôi về tin tức hôm nay"
)

# --- Main QA Logic ---
if query:
    with st.spinner("🔎 Searching..."):
        retrieval = retrieve_documents(query, vectorstore, client)

    if not retrieval["success"]:
        st.error(f"❌ Retrieval failed: {retrieval['content']}")
    else:
        st.markdown("### 🤖 Answer")
        response_placeholder = st.empty()
        full_response = ""

        for chunk in generate_response_stream(query, retrieval["content"]):
            full_response += chunk
            response_placeholder.markdown(full_response + "|")  # Blinking cursor

        response_placeholder.markdown(full_response)

        with st.expander("📚 Retrieved Context"):
            st.markdown(retrieval["content"][:3000])  # Avoid overwhelming display
