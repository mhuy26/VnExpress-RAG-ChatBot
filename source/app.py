# source/app.py
import streamlit as st
from config import validate_environment
from main import retrieve_documents, generate_response_stream  # streaming pipeline logic
from storage.qdrant_store import initialize_vectorstore
from langchain_qdrant import QdrantVectorStore


# --- Environment Setup ---
validate_environment()
vectorstore, client = initialize_vectorstore()

# --- Streamlit Page Config ---
st.set_page_config(page_title="VNExpress RAG QA", layout="wide")

# --- App Title and Description ---
st.title("📰 VNExpress Chatbot")
st.markdown("Ask questions about the VNExpress articles you've embedded.")

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
            response_placeholder.markdown(full_response + "▌")  # Blinking cursor

        response_placeholder.markdown(full_response)

        with st.expander("📚 Retrieved Context"):
            st.markdown(retrieval["content"][:3000])  # Avoid overwhelming display
