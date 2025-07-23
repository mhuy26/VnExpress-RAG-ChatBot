from datetime import timezone, datetime, timedelta
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vnexpress_articles")

client = QdrantClient(url=QDRANT_URL)

def delete_old_articles(days_to_keep=1):
    cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")

    filter_condition = {
        "must": [
            {
                "key": "published_date",
                "range": {
                    "lt": cutoff_date
                }
            }
        ]
    }

    deleted_count = client.delete(
        collection_name=QDRANT_COLLECTION,
        filter=filter_condition
    )

    print(f"🗑️ Deleted old articles older than {cutoff_date}: {deleted_count}")
