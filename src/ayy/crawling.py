import asyncio
from typing import Set
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler
from loguru import logger

MAX_DEPTH = 2
MAX_LINKS = 5


async def recursive_crawl(
    url: str,
    max_depth: int = MAX_DEPTH,
    max_links: int = MAX_LINKS,
    same_domain_only: bool = True,
    prefixes: list[str] | None = None,
) -> dict:
    """
    Recursively crawl starting from a URL up to a specified depth.

    Args:
        url: The URL to start crawling from
        max_depth: Maximum depth of recursion (default: 2)
        max_links: Maximum number of links to follow (default: 100)
        same_domain_only: Only follow links within the same domain (default: True)
        prefixes: List of prefixes to follow (default: None). So only links starting with these prefixes will be followed. If same_domain_only is True, it will be automatically added.
    """
    visited: Set[str] = set()
    results = {}
    start_domain = urlparse(url).netloc
    prefixes = prefixes or []
    if same_domain_only:
        prefixes = [p for p in prefixes + [start_domain] if urlparse(p).netloc == start_domain]
    logger.info(f"Crawling {url} with prefixes {prefixes}")

    async def crawl_url(url: str, depth: int):
        logger.info(f"Crawling {url} at depth {depth} with netloc {urlparse(url).netloc}")
        if depth > max_depth or url in visited or len(results) > max_links:
            return

        if prefixes and not any(url.startswith(p) for p in prefixes):
            logger.warning(f"Skipping {url} because it is not in prefixes")
            return

        visited.add(url)

        async with AsyncWebCrawler(verbose=True) as crawler:
            try:
                result = await crawler.arun(url=url)
                results[url] = result.markdown

                # Extract links from the page
                if result.links and depth < max_depth:
                    for _, links in result.links.items():
                        for link in links:
                            if "href" in link:
                                next_url = urljoin(url, link["href"])
                                await crawl_url(url=next_url, depth=depth + 1)
            except Exception:
                logger.exception(f"Error crawling {url}")

    await crawl_url(url, 1)
    return results


def crawl_url(
    url: str,
    max_depth: int = MAX_DEPTH,
    max_links: int = MAX_LINKS,
    same_domain_only: bool = True,
    prefixes: list[str] | None = None,
) -> dict:
    """
    Recursively crawl starting from a URL up to a specified depth.

    Args:
        url: The URL to start crawling from
        max_depth: Maximum depth of recursion (default: 2)
        max_links: Maximum number of links to follow (default: 100)
        same_domain_only: Only follow links within the same domain (default: True)
        prefixes: List of prefixes to follow (default: None). So only links starting with these prefixes will be followed. If same_domain_only is True, it will be automatically added.
    """
    return asyncio.run(
        recursive_crawl(
            url=url, max_depth=max_depth, max_links=max_links, same_domain_only=same_domain_only, prefixes=prefixes
        )
    )
