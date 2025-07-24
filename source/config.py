import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vnexpress_articles")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL_NAME", "models/embedding-001")

def validate_environment():
    required = {
        "QDRANT_URL": QDRANT_URL,
        "GOOGLE_API_KEY": GOOGLE_API_KEY,
        "QDRANT_COLLECTION": QDRANT_COLLECTION
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")
    
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
