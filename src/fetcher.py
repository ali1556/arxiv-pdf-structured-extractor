from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
import httpx

logger = logging.getLogger("arxiv_extractor.fetcher")

ARXIV_ID_PATTERN = re.compile(
    r"(?:arxiv(?:\.org)?/(?:abs|pdf)/|arxiv:)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?|[a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?/[0-9]{7}(?:v[0-9]+)?)",
    re.IGNORECASE,
)
DOI_ARXIV_PATTERN = re.compile(r"10\.48550/arxiv\.([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", re.IGNORECASE)

DEFAULT_USER_AGENT = "AcademicPaperExtractor/1.0 (Apify Actor; research-pdf-bot; +https://apify.com)"


def extract_arxiv_id(raw_input: str) -> Optional[str]:
    """Extract canonical ArXiv ID from a URL, DOI, or raw ID string."""
    clean = raw_input.strip()
    doi_match = DOI_ARXIV_PATTERN.search(clean)
    if doi_match:
        return doi_match.group(1)

    match = ARXIV_ID_PATTERN.search(clean)
    if match:
        return match.group(1)

    return None


def resolve_paper_url(raw_input: str) -> Tuple[str, Optional[str]]:
    """
    Resolves any paper input (ArXiv ID, ArXiv URL, DOI, or direct PDF URL)
    into a direct downloadable PDF URL and optional ArXiv ID.
    """
    clean = raw_input.strip()
    arxiv_id = extract_arxiv_id(clean)

    if arxiv_id:
        # Strip trailing .pdf if mistakenly included in id
        clean_id = arxiv_id[:-4] if arxiv_id.endswith(".pdf") else arxiv_id
        pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
        return pdf_url, clean_id

    # Bare DOI (e.g. 10.1000/182 or 10.1016/j.cell...)
    if re.match(r"^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$", clean):
        return f"https://doi.org/{clean}", None

    # If it's already a full HTTP/HTTPS URL
    if clean.startswith("http://") or clean.startswith("https://"):
        return clean, None

    # Fallback assumption
    return clean, None


async def fetch_arxiv_api_metadata(
    client: httpx.AsyncClient,
    arxiv_id: str,
    timeout_secs: float = 12.0,
) -> Optional[dict]:
    """
    Queries the official ArXiv Atom API to extract high-accuracy metadata.
    Returns dict with title, authors, abstract, published date, doi, etc.
    """
    # Remove version suffix (e.g. 2312.12345v2 -> 2312.12345) for canonical API search
    base_id = re.sub(r"v\d+$", "", arxiv_id)
    api_url = f"https://export.arxiv.org/api/query?id_list={base_id}&max_results=1"

    try:
        resp = await client.get(api_url, timeout=timeout_secs)
        if resp.status_code != 200:
            logger.warning(f"ArXiv API returned status {resp.status_code} for {arxiv_id}")
            return None

        # Parse Atom XML
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title_elem = entry.find("atom:title", ns)
        title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else ""

        summary_elem = entry.find("atom:summary", ns)
        abstract = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else ""

        published_elem = entry.find("atom:published", ns)
        published = published_elem.text if published_elem is not None else None

        doi_elem = entry.find("arxiv:doi", ns)
        doi = doi_elem.text if doi_elem is not None else None

        authors = []
        affiliations = []
        for author in entry.findall("atom:author", ns):
            name_el = author.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())
            affil_el = author.find("arxiv:affiliation", ns)
            if affil_el is not None and affil_el.text:
                affil = affil_el.text.strip()
                if affil not in affiliations:
                    affiliations.append(affil)

        return {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "affiliations": affiliations,
            "publication_date": published,
            "doi": doi,
        }
    except Exception as exc:
        logger.warning(f"Failed to query ArXiv API for {arxiv_id}: {exc}")
        return None


async def download_pdf_bytes(
    client: httpx.AsyncClient,
    pdf_url: str,
    timeout_secs: float = 45.0,
    max_retries: int = 3,
) -> bytes:
    """
    Downloads PDF content with retries and exponential backoff.
    Ensures received payload is valid PDF (%PDF header).
    """
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading PDF (attempt {attempt}/{max_retries}): {pdf_url}")
            response = await client.get(
                pdf_url,
                timeout=timeout_secs,
                follow_redirects=True,
            )

            if response.status_code == 429:
                # Rate limited
                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after and retry_after.isdigit() else delay
                logger.warning(f"HTTP 429 rate limit hit for {pdf_url}. Backing off {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                delay *= 2
                continue

            if response.status_code == 503:
                logger.warning(f"HTTP 503 service unavailable for {pdf_url}. Retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay *= 2
                continue

            response.raise_for_status()

            pdf_bytes = response.content
            if not pdf_bytes.startswith(b"%PDF"):
                raise ValueError(
                    f"URL returned non-PDF content (starts with: {pdf_bytes[:30]!r}). Expected '%PDF' header."
                )

            return pdf_bytes

        except (httpx.TimeoutException, httpx.NetworkError) as err:
            logger.warning(f"Network error downloading {pdf_url} (attempt {attempt}): {err}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2

    raise RuntimeError(f"Exceeded max retries downloading PDF from {pdf_url}")
