# source/model.py

from config import GOOGLE_API_KEY, EMBEDDING_MODEL
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def load_llm():
    return init_chat_model("gemini-2.0-flash", model_provider="google_genai")

def load_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
