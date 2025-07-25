# source/crawler/content.py
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from datetime import datetime
from .utils import retry_on_failure


def extract_article_content(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article.fck_detail")
    if not article:
        return ""

    for tag in article.select("script, style, .ads, .advertisement"):
        tag.decompose()

    return article.get_text(separator="\n").strip()


@retry_on_failure(max_retries=2, delay=1)
def crawl_single_article(url: str, headers: dict) -> Document | None:
    if headers is None:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }


    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch URL: {url} → {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    content = extract_article_content(response.text)

    if not content or len(content.strip()) < 100:
        print(f"⚠️ Skipping short article: {url}")
        return None

    title = soup.title.string.strip() if soup.title else ""

    description_meta = soup.find("meta", attrs={"name": "description"})
    description = description_meta["content"].strip() if description_meta and "content" in description_meta.attrs else ""

    # --- Parse publish date ---
    dt = None
    publish_date = ""
    try:
        raw_date = soup.find("span", class_="date").get_text().strip()
        # Strip anything after GMT or in parentheses
        raw_date = raw_date.split("GMT")[0].split("(")[0].strip()
        parts = [p.strip() for p in raw_date.split(",") if p.strip()]

        if len(parts) == 2:
            date_part, time_part = parts[1], "00:00"
        elif len(parts) == 1:
            date_part, time_part = parts[0], "00:00"
        else:
            date_part, time_part = parts[1], parts[2].split(" ")[0]

        dt = datetime.strptime(f"{date_part} {time_part}", "%d/%m/%Y %H:%M")
        publish_date = f"{dt.strftime('%Y-%m-%d %H:%M')} +0700"
    except Exception as e:
        print(f"⚠️ Failed to parse publish date for {url}: {e}")

    # --- Parse category ---
    category = ""
    breadcrumb = soup.find("ul", class_="breadcrumb")
    if breadcrumb:
        links = breadcrumb.find_all("a")
        if links:
            category = links[-1].get_text().strip()

    return Document(
    page_content=content,
    metadata={
        "source": url,
        "language": "vi",
        "title": title,
        "description": description,
        "publish_ts": dt.timestamp() if dt else None,                 
        "publish_date": publish_date if dt else None,
        "category": category,
        "content_length": len(content)
    }
)