# crawler/vnexpress.py
"""
VNExpress-specific content and title extraction
"""
from bs4 import BeautifulSoup

def extract_vnexpress_content(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    selectors = [
        "article.fck_detail", ".fck_detail", "article",
        ".article-content", ".content_detail"
    ]

    article = None
    for selector in selectors:
        article = soup.select_one(selector)
        if article:
            break

    if not article:
        article = soup.find("body")
        if not article:
            return "", ""

    for tag in article.select("""
        script, style, nav, header, footer, aside, 
        .ads, .advertisement, .social-share, .comments, .comment,
        .related-news, .tags, .author-info, .breadcrumb,
        .social-plugin, .fb-comments, .zalo-share-button,
        .newsletter-signup, .promotion, .banner
    """):
        tag.decompose()

    text = article.get_text(separator="\n").strip()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    content = '\n'.join(lines)

    title = ""
    title_selectors = ["h1", ".title", ".article-title", "title"]
    for selector in title_selectors:
        title_elem = soup.select_one(selector)
        if title_elem:
            title = title_elem.get_text().strip()
            break

    return content, title