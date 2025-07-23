# app.py
# Streamlit web app for VNExpress Retrieval-Augmented Generation (RAG) chatbot.

import streamlit as st
from agent import retrieve_documents, generate_response_stream  # use stream version!

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
        retrieved = retrieve_documents(query)

    st.markdown("### 🤖 Answer")
    response_placeholder = st.empty()
    full_response = ""

    # --- Stream the response from Gemini ---
    for chunk in generate_response_stream(query, retrieved):
        full_response += chunk
        response_placeholder.markdown(full_response + "▌")  # Blinking cursor effect

    # Final output without cursor
    response_placeholder.markdown(full_response)
