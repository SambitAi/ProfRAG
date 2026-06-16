from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_BAD_EXTRACT_PATTERNS = [
    re.compile(r"i am sorry,? but i cannot access external websites", re.IGNORECASE),
    re.compile(r"unable to read the content of the section", re.IGNORECASE),
    re.compile(r"if you can copy and paste the text here", re.IGNORECASE),
    re.compile(r"sorry,? i can(?:not|'t) read the url", re.IGNORECASE),
]

_BLOCKED_ACCESS_PATTERNS = [
    re.compile(r"\baccess denied\b", re.IGNORECASE),
    re.compile(r"\byou don't have permission to access\b", re.IGNORECASE),
    re.compile(r"\berrors\.edgesuite\.net\b", re.IGNORECASE),
    re.compile(r"\bakamai\b", re.IGNORECASE),
    re.compile(r"\brequest blocked\b", re.IGNORECASE),
    re.compile(r"\bbot detection\b", re.IGNORECASE),
    re.compile(r"\bsecurity check\b", re.IGNORECASE),
    re.compile(r"\bforbidden\b", re.IGNORECASE),
]

def _run_async(coro_factory):
    """Run an async callable safely on Windows (ProactorEventLoop) even inside Streamlit.

    Takes a zero-arg callable returning a fresh coroutine — not a coroutine object —
    so the worker-thread fallback never re-awaits an already-started coroutine.
    """
    import asyncio
    import sys
    import concurrent.futures

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())
    # Already inside a running event loop (some Streamlit configs): run on a worker thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro_factory())).result()


def _html_to_markdown(html_text: str, url: str) -> str:
    if not html_text.strip():
        return ""

    try:
        import trafilatura

        result = trafilatura.extract(
            html_text,
            url=url,
            output_format="markdown",
            include_tables=True,
            favor_recall=True,
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
        result = h.handle(html_text)
        if result and result.strip():
            return result.strip()
    except Exception:
        pass

    text = re.sub(
        r"<(script|style|noscript)[^>]*>.*?</(script|style|noscript)>",
        "",
        html_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_bad_extraction(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return True
    if len(normalized) < 120:
        return True
    return any(pattern.search(normalized) for pattern in _BAD_EXTRACT_PATTERNS)


def _looks_like_blocked_access_page(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _BLOCKED_ACCESS_PATTERNS)


def _best_markdown_candidate(*candidates: str) -> str:
    usable = [str(candidate or "").strip() for candidate in candidates if str(candidate or "").strip()]
    if not usable:
        return ""
    good = [candidate for candidate in usable if not _looks_like_bad_extraction(candidate)]
    pool = good or usable
    return max(pool, key=len)


def _scrape_with_playwright(url: str) -> str:
    """Fetch rendered HTML with Playwright, click common consent buttons, then extract markdown."""
    from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    from playwright.async_api import async_playwright

    async def _crawl() -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=_USER_AGENT)
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)
                try:
                    await page.wait_for_load_state("networkidle", timeout=8000)
                except PlaywrightTimeoutError:
                    pass

                await page.evaluate(
                    """
                    () => {
                      const textMatches = (value) => {
                        const text = (value || '').replace(/\\s+/g, ' ').trim().toLowerCase();
                        if (!text || text.length > 40) return false;
                        // Covers the common real-world labels: "Accept All Cookies"
                        // (OneTrust default), "Allow all cookies", "I accept",
                        // "Agree and continue", "Accept & close", plus the bare forms.
                        return (
                          /^(i )?(accept|allow|agree|consent)( (all|cookies|all cookies|necessary cookies))?( ?(&|and) ?(continue|close|proceed))?$/.test(text)
                          || /^(got it|ok|okay|yes|continue|understood|i understand)$/.test(text)
                        );
                      };
                      for (const root of [document, ...Array.from(document.querySelectorAll('iframe')).map(f => {
                        try { return f.contentDocument; } catch { return null; }
                      }).filter(Boolean)]) {
                        const nodes = root.querySelectorAll('button, a[role="button"], input[type="button"], input[type="submit"], [aria-label]');
                        for (const node of nodes) {
                          const label = node.innerText || node.value || node.getAttribute('aria-label') || '';
                          if (textMatches(label)) {
                            try { node.click(); } catch {}
                          }
                        }
                      }
                    }
                    """
                )
                await page.wait_for_timeout(1200)
                await page.evaluate(
                    """
                    () => {
                      const selectors = [
                        '[id*="cookie" i]',
                        '[class*="cookie" i]',
                        '[id*="consent" i]',
                        '[class*="consent" i]',
                        '[aria-label*="cookie" i]',
                        '[aria-label*="consent" i]',
                        '[data-testid*="cookie" i]',
                        '[data-testid*="consent" i]'
                      ];
                      for (const selector of selectors) {
                        for (const node of document.querySelectorAll(selector)) {
                          try {
                            node.remove();
                          } catch {}
                        }
                      }
                      if (document.documentElement) document.documentElement.style.overflow = 'auto';
                      if (document.body) document.body.style.overflow = 'auto';
                    }
                    """
                )
                html_text = await page.content()
                article_text = await page.evaluate(
                    """
                    () => {
                      const candidates = [
                        'main article',
                        'article',
                        'main',
                        '[role="main"]',
                        '#main-content',
                        '.article',
                        '.content'
                      ];
                      for (const selector of candidates) {
                        const node = document.querySelector(selector);
                        const text = (node?.innerText || '').replace(/\\s+/g, ' ').trim();
                        if (text.length >= 400) return text;
                      }
                      return '';
                    }
                    """
                )
            finally:
                await browser.close()
        article_markdown = article_text.strip() if article_text else ""
        html_markdown = _html_to_markdown(html_text, url)
        return _best_markdown_candidate(article_markdown, html_markdown)

    return _run_async(_crawl)


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

    return _run_async(_crawl)


def scrape_url_to_markdown(url: str) -> str:
    """Scrape any HTML URL to markdown. Rendered-browser scrape first, plain HTTP last."""
    try:
        text = _scrape_with_playwright(url)
        if _looks_like_blocked_access_page(text):
            raise RuntimeError(
                "This site appears to block automated access (for example via bot protection or permission checks)."
            )
        if text and not _looks_like_bad_extraction(text):
            return text
    except Exception:
        pass

    try:
        text = _scrape_with_crawl4ai(url)
        if _looks_like_blocked_access_page(text):
            raise RuntimeError(
                "This site appears to block automated access (for example via bot protection or permission checks)."
            )
        if text and not _looks_like_bad_extraction(text):
            return text
    except Exception:
        pass

    # Fallback: plain HTTP + trafilatura + html2text + raw tag strip
    response = requests.get(url, timeout=15, headers={"User-Agent": _USER_AGENT})
    html_bytes = response.content if response.ok else b""

    if html_bytes:
        text = _html_to_markdown(html_bytes.decode("utf-8", errors="replace"), url)
        if _looks_like_blocked_access_page(text):
            raise RuntimeError(
                f"Could not extract any content from {url}: the site appears to block automated access."
            )
        if text and not _looks_like_bad_extraction(text):
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
