from __future__ import annotations

import re
from urllib.parse import urlparse

import requests


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _run_async(coro):
    """Run a coroutine safely on Windows (ProactorEventLoop) even inside Streamlit."""
    import asyncio
    import sys
    import concurrent.futures

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # Already inside a running event loop (some Streamlit configs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


def _scrape_with_crawl4ai(url: str) -> str:
    """Fetch and extract markdown from a URL using Crawl4AI (handles JS, cookie banners)."""
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

    async def _crawl():
        browser_cfg = BrowserConfig(headless=True, verbose=False)
        run_cfg = CrawlerRunConfig(
            word_count_threshold=10,
            remove_overlay_elements=True,
        )
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            result = await crawler.arun(url=url, config=run_cfg)
            if not result.success:
                raise RuntimeError(result.error_message or "Crawl4AI returned no result")
            md = result.markdown
            return (md.fit_markdown or md.raw_markdown or "").strip()

    return _run_async(_crawl())


def scrape_url_to_markdown(url: str) -> str:
    """Scrape any HTML URL to markdown. Crawl4AI primary, trafilatura/html2text fallback."""
    try:
        text = _scrape_with_crawl4ai(url)
        if text and len(text) > 100:
            return text
    except Exception:
        pass

    # Fallback: plain HTTP + trafilatura + html2text + raw tag strip
    response = requests.get(url, timeout=15, headers={"User-Agent": _USER_AGENT})
    html_bytes = response.content if response.ok else b""

    if html_bytes:
        try:
            import trafilatura
            result = trafilatura.extract(
                html_bytes, url=url, output_format="markdown",
                include_tables=True, favor_recall=True,
            )
            if result and result.strip():
                return result.strip()
        except Exception:
            pass

        try:
            import html2text
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.body_width = 0
            result = h.handle(html_bytes.decode("utf-8", errors="replace"))
            if result and result.strip():
                return result.strip()
        except Exception:
            pass

        # Last resort: strip all HTML tags
        text = html_bytes.decode("utf-8", errors="replace")
        text = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and len(text) > 50:
            return text

    raise RuntimeError(f"Could not extract any content from {url}")


def is_pdf_url(url: str) -> bool:
    """Return True if the URL serves a PDF (by path extension or Content-Type)."""
    if urlparse(url).path.lower().endswith(".pdf"):
        return True
    try:
        head = requests.head(url, timeout=10, headers={"User-Agent": _USER_AGENT},
                             allow_redirects=True)
        return "application/pdf" in head.headers.get("Content-Type", "").lower()
    except Exception:
        return False


def fetch_pdf_bytes(url: str) -> bytes:
    """Download a PDF URL and return raw bytes."""
    response = requests.get(url, timeout=30, headers={"User-Agent": _USER_AGENT})
    if not response.ok:
        raise ValueError(f"HTTP {response.status_code} fetching {url}")
    return response.content


def url_to_document_name(url: str) -> str:
    """Derive a stable, slug-safe document name from a URL.

    Dots are replaced with underscores so slugify_filename (which strips
    Path.stem) doesn't truncate domain names.

    e.g. https://docs.python.org/3/library/os.html → docs_python_org_3_library_os
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.replace(".", "_").replace(":", "_")
    path_parts = [p for p in parsed.path.split("/") if p]
    if path_parts:
        last = re.sub(r"\.(html?|pdf|php|aspx?)$", "", path_parts[-1], flags=re.IGNORECASE)
        path_parts[-1] = last
    path_parts = [p for p in path_parts if p]
    combined = "_".join([netloc] + path_parts) if path_parts else netloc
    combined = re.sub(r"[^a-zA-Z0-9_]+", "_", combined)
    combined = re.sub(r"_+", "_", combined).strip("_")
    return combined[:80] or "web_document"
