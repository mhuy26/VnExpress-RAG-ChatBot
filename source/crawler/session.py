# crawler/session.py
"""
Session tracking and retry state management
"""
import time
import hashlib

# Cache for failed URLs to avoid repeated retries
FAILED_URLS = set()

def generate_session_id(url: str) -> str:
    return hashlib.md5(f"{url}{time.time()}".encode()).hexdigest()[:8]