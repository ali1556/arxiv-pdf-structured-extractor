import pytest
from src.fetcher import extract_arxiv_id, resolve_paper_url


def test_extract_arxiv_id_various_formats():
    # Canonical new format
    assert extract_arxiv_id("1706.03762") == "1706.03762"
    assert extract_arxiv_id("2312.12345v2") == "2312.12345v2"
    
    # Prefixed
    assert extract_arxiv_id("arXiv:1706.03762") == "1706.03762"
    assert extract_arxiv_id("arxiv:2103.00020v1") == "2103.00020v1"
    
    # Abstract and PDF URLs
    assert extract_arxiv_id("https://arxiv.org/abs/1706.03762") == "1706.03762"
    assert extract_arxiv_id("http://arxiv.org/pdf/1706.03762.pdf") == "1706.03762"
    assert extract_arxiv_id("https://arxiv.org/abs/2312.12345v3") == "2312.12345v3"

    # Old format
    assert extract_arxiv_id("hep-th/9901001") == "hep-th/9901001"
    assert extract_arxiv_id("https://arxiv.org/abs/math.GT/0309136") == "math.GT/0309136"

    # DOI format
    assert extract_arxiv_id("10.48550/arXiv.1706.03762") == "1706.03762"

    # Generic URLs should return None
    assert extract_arxiv_id("https://nature.com/articles/s41586-020.pdf") is None
    assert extract_arxiv_id("invalid-text-string") is None


def test_resolve_paper_url():
    # ArXiv ID to PDF URL
    pdf_url, arxiv_id = resolve_paper_url("1706.03762")
    assert pdf_url == "https://arxiv.org/pdf/1706.03762.pdf"
    assert arxiv_id == "1706.03762"

    # Direct ArXiv PDF URL
    pdf_url, arxiv_id = resolve_paper_url("https://arxiv.org/pdf/2312.12345.pdf")
    assert pdf_url == "https://arxiv.org/pdf/2312.12345.pdf"
    assert arxiv_id == "2312.12345"

    # Direct non-arxiv PDF URL
    pdf_url, arxiv_id = resolve_paper_url("https://example.com/deeplearning.pdf")
    assert pdf_url == "https://example.com/deeplearning.pdf"
    assert arxiv_id is None
