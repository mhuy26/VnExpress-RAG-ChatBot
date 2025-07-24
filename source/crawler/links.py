# source/crawler/links.py

import requests
from bs4 import BeautifulSoup
from functools import wraps
import time


def retry_on_failure(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


@retry_on_failure(max_retries=2, delay=1)
def get_article_links(max_links=None):
    print("🔗 Fetching article links from VNExpress homepage...")
    url = "https://vnexpress.net/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    links = []

    selectors = [
        "h3.title-news a",
        "h2.title-news a",
        "h4.title-news a",
        ".item-news h3 a",
        ".item-news h2 a"
    ]

    for selector in selectors:
        for tag in soup.select(selector):
            href = tag.get("href")
            if href and href.startswith("https://vnexpress.net") and "/video-" not in href:
                links.append(href)

    # Remove duplicates and limit
    unique_links = list(dict.fromkeys(links))
    if max_links:
        unique_links = unique_links[:max_links]

    print(f"✅ Found {len(unique_links)} article links")
    return unique_links
