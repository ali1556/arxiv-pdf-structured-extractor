import pymupdf as fitz
from src.parser import (
    clean_text,
    extract_citations_from_text,
    parse_pdf_document,
)


def create_synthetic_paper_pdf() -> bytes:
    """Generates a minimal valid academic paper PDF in memory using fitz."""
    doc = fitz.open()
    page = doc.new_page()

    # Title & Authors
    page.insert_text((50, 60), "Self-Attention in Neural Networks", fontsize=18)
    page.insert_text((50, 90), "Alice Researcher, Bob Scientist", fontsize=11)
    
    # Abstract
    page.insert_text((50, 130), "Abstract", fontsize=12)
    page.insert_text(
        (50, 150),
        "In this paper, we propose a novel transformer-based neural architecture that eliminates recurrence.",
        fontsize=10,
    )

    # Section 1
    page.insert_text((50, 200), "1. Introduction", fontsize=13)
    page.insert_text(
        (50, 220),
        "Recurrent models typically factor computation along the symbol positions of the input sequence.",
        fontsize=10,
    )
    # Equation label
    page.insert_text((450, 260), "y = Wx + b   (1)", fontsize=10)

    # Table caption
    page.insert_text((50, 310), "Table 1: BLEU scores on WMT 2014 English-to-German test set.", fontsize=10)
    page.insert_text((50, 330), "Model | BLEU | Training Cost (FLOPs)", fontsize=9)

    # Section 2 References
    page.insert_text((50, 380), "References", fontsize=13)
    page.insert_text((50, 400), "[1] Vaswani, A., et al. Attention Is All You Need. NeurIPS, 2017.", fontsize=9)
    page.insert_text((50, 420), "[2] Devlin, J., et al. BERT: Pre-training of Deep Bidirectional Transformers. arXiv:1810.04805, 2018.", fontsize=9)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_clean_text():
    raw = "trans-\nformer archi-\ntecture is fast."
    assert clean_text(raw) == "transformer architecture is fast."


def test_extract_citations():
    ref_text = (
        "[1] Vaswani, A., et al. Attention Is All You Need. NeurIPS, 2017.\n"
        "[2] Devlin, J. BERT Pre-training. arXiv:1810.04805, 2018."
    )
    citations = extract_citations_from_text(ref_text)
    assert len(citations) == 2
    assert citations[0].year == 2017
    assert citations[1].year == 2018
    assert citations[1].arxiv_id_if_present == "1810.04805"


def test_parse_synthetic_pdf():
    pdf_bytes = create_synthetic_paper_pdf()
    output = parse_pdf_document(
        pdf_bytes=pdf_bytes,
        source_input="test-input",
        pdf_url="https://example.com/test.pdf",
        arxiv_id="2312.00001",
        api_metadata={
            "title": "Self-Attention in Neural Networks",
            "authors": ["Alice Researcher", "Bob Scientist"],
            "abstract": "In this paper, we propose a novel transformer-based neural architecture.",
            "publication_date": "2023-12-01T00:00:00Z",
            "doi": "10.48550/arXiv.2312.00001",
        },
    )

    assert output.status == "SUCCESS"
    assert output.metadata.title == "Self-Attention in Neural Networks"
    assert len(output.metadata.authors) == 2
    assert output.metadata.arxiv_id == "2312.00001"
    assert "transformer-based neural architecture" in output.abstract
    assert output.equations_count >= 1
    assert len(output.tables_summary) >= 1
    assert "Table 1" in output.tables_summary[0].caption
    assert len(output.citations) >= 1
    assert output.stats.pages == 1


def test_parse_pure_pdf_without_api_metadata():
    pdf_bytes = create_synthetic_paper_pdf()
    output = parse_pdf_document(
        pdf_bytes=pdf_bytes,
        source_input="https://example.com/standalone.pdf",
        pdf_url="https://example.com/standalone.pdf",
        arxiv_id=None,
        api_metadata=None,
    )

    assert output.status == "SUCCESS"
    assert "Self-Attention in Neural Networks" in output.metadata.title
    assert output.stats.pages == 1
    assert output.stats.characters > 50


def test_parse_corrupted_empty_pdf_raises():
    import pytest
    # A valid PDF header and trailer with 0 pages
    empty_bytes = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"

    with pytest.raises(ValueError, match="0 pages"):
        parse_pdf_document(
            pdf_bytes=empty_bytes,
            source_input="bad-pdf",
            pdf_url="https://example.com/empty.pdf",
        )
