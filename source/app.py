
# app.py
# Streamlit web app for VNExpress Retrieval-Augmented Generation (RAG) chatbot.
#
# This app allows users to ask questions about embedded VNExpress news articles.
# It retrieves relevant articles and generates concise answers using a Gemini LLM.

import streamlit as st
from agent import retrieve_documents, generate_response


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
    # Retrieve relevant articles and generate answer
    with st.spinner("🔎 Searching..."):
        retrieved = retrieve_documents(query)
        answer = generate_response(query, retrieved)

    # Display retrieved context
    st.markdown("### 📑 Retrieved Context")
    st.code(retrieved, language="markdown")

    # Display generated answer
    st.markdown("### 🤖 Answer")
    st.success(answer)
