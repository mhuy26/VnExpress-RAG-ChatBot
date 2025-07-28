# crawler/playwright_crawler.py
"""
Playwright-based crawler integrated with the RAG pipeline
"""

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from datetime import datetime
import random
import time
from typing import Optional, List
import logging

from crawler.header import get_random_user_agent, get_random_screen_size, get_random_proxy
from crawler.session import generate_session_id, FAILED_URLS
from crawler.stealth import add_stealth_scripts
from crawler.vnexpress import extract_vnexpress_content

logger = logging.getLogger(__name__)

# ---------- Main Crawling Functions ----------

def crawl_single_article_playwright(url: str, max_retries: int = 3) -> Optional[Document]:
    if url in FAILED_URLS:
        logger.info(f"⏭️ Skipping previously failed URL: {url}")
        return None

    session_id = generate_session_id(url)

    for attempt in range(max_retries):
        with sync_playwright() as p:
            proxy = get_random_proxy()
            user_agent = get_random_user_agent()
            screen_size = get_random_screen_size()

            launch_args = {
                "headless": True,
                "args": [
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-extensions",
                    "--disable-plugins-discovery"
                ]
            }

            if proxy:
                launch_args["proxy"] = {"server": proxy}

            try:
                browser = p.chromium.launch(**launch_args)
                context = browser.new_context(
                    user_agent=user_agent,
                    viewport=screen_size,
                    locale="vi-VN",
                    timezone_id="Asia/Ho_Chi_Minh",
                    java_script_enabled=True,
                    extra_http_headers={
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
                        "Accept-Encoding": "gzip, deflate",
                        "Connection": "keep-alive"
                    }
                )

                page = context.new_page()
                add_stealth_scripts(page)

                logger.info(f"🌐 [{session_id}] Attempt {attempt + 1}: {url}")

                response = page.goto(url, timeout=30000, wait_until="domcontentloaded")

                if response.status >= 400:
                    logger.warning(f"⚠️ [{session_id}] HTTP {response.status}")
                    if attempt == max_retries - 1:
                        FAILED_URLS.add(url)
                    continue

                try:
                    page.wait_for_selector("article.fck_detail", timeout=5000)
                except:
                    try:
                        page.wait_for_selector("article", timeout=3000)
                    except:
                        page.wait_for_timeout(2000)

                page.mouse.wheel(0, random.randint(100, 500))
                time.sleep(random.uniform(0.5, 1.5))

                html = page.content()

            except Exception as e:
                logger.error(f"❌ [{session_id}] Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    FAILED_URLS.add(url)
                continue
            finally:
                try:
                    browser.close()
                except:
                    pass

            content, title = extract_vnexpress_content(html)

            if not content or len(content.strip()) < 100:
                logger.warning(f"⚠️ [{session_id}] Insufficient content ({len(content)} chars)")
                if attempt == max_retries - 1:
                    FAILED_URLS.add(url)
                continue

            logger.info(f"✅ [{session_id}] Success: {len(content)} chars")

            return Document(
                page_content=content,
                metadata={
                    "source": url,
                    "title": title,
                    "user_agent": user_agent,
                    "proxy": proxy,
                    "timestamp": datetime.now().isoformat(),
                    "content_length": len(content),
                    "session_id": session_id,
                    "attempts": attempt + 1,
                    "crawler_type": "playwright"
                }
            )

    logger.error(f"💀 [{session_id}] All attempts failed for {url}")
    return None

def crawl_articles(urls: List[str], delay_range: tuple = (3, 8)) -> List[Document]:
    documents = []
    total = len(urls)

    logger.info(f"🚀 Starting crawl of {total} URLs")

    for i, url in enumerate(urls, 1):
        logger.info(f"📄 Processing {i}/{total}: {url}")

        doc = crawl_single_article_playwright(url)
        if doc:
            documents.append(doc)
            success_rate = len(documents) / i * 100
            logger.info(f"✅ Success rate: {len(documents)}/{i} ({success_rate:.1f}%)")
        else:
            logger.warning(f"❌ Failed: {url}")

        if i < total:
            recent_failures = len([u for u in urls[max(0, i-5):i] if u in FAILED_URLS])
            base_delay = delay_range[0] if recent_failures < 2 else delay_range[1]
            delay = random.uniform(base_delay, base_delay + 2)
            logger.info(f"⏱️ Waiting {delay:.1f}s...")
            time.sleep(delay)

    success_rate = len(documents) / total * 100 if total > 0 else 0
    logger.info(f"🎯 Crawl completed: {len(documents)}/{total} ({success_rate:.1f}%)")

    return documents

def crawl_single_article(url: str, headers: dict = None) -> Optional[Document]:
    return crawl_single_article_playwright(url)