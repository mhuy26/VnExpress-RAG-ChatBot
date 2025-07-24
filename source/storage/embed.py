# source/storage/embed.py

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from crawler.utils import validate_env
import os
from dotenv import load_dotenv
load_dotenv()

def load_embedding_model():
    """Load and validate Gemini embedding model."""
    validate_env(["GOOGLE_API_KEY"])
    model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    test_vec = model.embed_query("test")
    print(f"✅ Gemini embedding model loaded. Vector size: {len(test_vec)}")
    return model, len(test_vec)


def validate_vector_size(documents, embedding, expected_dim):
    """Ensure all documents embed to expected dimension."""
    valid_docs, invalid = [], 0
    for i, doc in enumerate(documents):
        try:
            test = doc.page_content[:100]
            vec = embedding.embed_query(test)
            if len(vec) != expected_dim:
                print(f"❌ Chunk {i} size mismatch: {len(vec)} != {expected_dim}")
                invalid += 1
            else:
                valid_docs.append(doc)
        except Exception as e:
            print(f"⚠️ Embedding error on chunk {i}: {e}")
            invalid += 1
    print(f"✅ {len(valid_docs)} valid chunks | ⚠️ {invalid} failed")
    return valid_docs
