from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import pymupdf as fitz

from src.models import (
    Citation,
    PaperMetadata,
    PaperSection,
    PaperStats,
    PaperStructuredOutput,
    TableSummary,
)

logger = logging.getLogger("arxiv_extractor.parser")

HEADING_PATTERN = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2})*|\b[IVXLCDM]+\b\.?)\s+[A-Z][A-Za-z0-9\s,\-–:]{2,80}$"
    r"|^(?:Abstract|Introduction|Background|Related Work|Methodology|Methods|System Architecture|Experiments|Experimental Setup|Results|Discussion|Conclusion|Conclusions|Limitations|Broader Impacts|Acknowledgements|Acknowledgments|References|Bibliography|Appendix(?:\s+[A-Z0-9]+)?)$",
    re.IGNORECASE,
)

TABLE_CAPTION_PATTERN = re.compile(
    r"^(?:Table\s+\d+|TABLE\s+[IVXLCDM]+)[\s.:–-]+(.*)",
    re.IGNORECASE,
)

EQUATION_LABEL_PATTERN = re.compile(r"\((?:\d{1,3}(?:\.\d{1,3})?|[a-z]{1,2})\)\s*$")
CITATION_YEAR_PATTERN = re.compile(r"\b(19\d{2}|20[0-2]\d)\b")
CITATION_ARXIV_PATTERN = re.compile(r"(?:arXiv:\s*|arxiv\.org/(?:abs|pdf)/)([0-9]{4}\.[0-9]{4,5})", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalize whitespace and fix hyphenation at line breaks."""
    # Fix hyphenated words at line wrap: e.g. "trans- \nformer" -> "transformer"
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Convert multiple spaces / newlines to single space
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(lines)


def extract_citations_from_text(ref_text: str) -> List[Citation]:
    """
    Parses references text into structured Citation items.
    Handles numeric markers ([1], 1.) and author-year markers ([Vaswani2017]).
    """
    citations: List[Citation] = []
    if not ref_text:
        return citations

    # Split by reference markers like [1], [2], or 1., 2. (independent of line breaks)
    raw_entries = re.split(r"(?:\[\d{1,3}\]|\b\d{1,3}\.\s+|\[[A-Za-z\-]+(?:\s*et\s*al\.)?,?\s*\d{4}\])", ref_text)
    
    # Filter out header or empty chunks
    entries = [e.strip() for e in raw_entries if len(e.strip()) > 15]

    for entry in entries:
        cleaned_entry = clean_text(entry)
        if not cleaned_entry:
            continue

        # Extract year
        year_match = CITATION_YEAR_PATTERN.search(cleaned_entry)
        year = int(year_match.group(1)) if year_match else None

        # Extract ArXiv ID if cited
        arxiv_match = CITATION_ARXIV_PATTERN.search(cleaned_entry)
        arxiv_id = arxiv_match.group(1) if arxiv_match else None

        # Guess title: Look for text inside quotes, or second clause
        title_guess = None
        quote_match = re.search(r"[\"“](.*?)[\"”]", cleaned_entry)
        if quote_match and len(quote_match.group(1).strip()) > 8:
            title_guess = quote_match.group(1).strip()
        else:
            # Fallback: parts split by period
            parts = [p.strip() for p in cleaned_entry.split(".") if len(p.strip()) > 10]
            if len(parts) >= 2:
                title_guess = parts[1]

        citations.append(
            Citation(
                raw_text=cleaned_entry,
                title_guess=title_guess,
                year=year,
                arxiv_id_if_present=arxiv_id,
            )
        )

    return citations


def parse_pdf_document(
    pdf_bytes: bytes,
    source_input: str,
    pdf_url: str,
    arxiv_id: Optional[str] = None,
    api_metadata: Optional[Dict[str, Any]] = None,
    duration_secs: float = 0.0,
) -> PaperStructuredOutput:
    """
    Performs fast, high-fidelity PDF extraction using PyMuPDF.
    Combines layout analysis with ArXiv API metadata for high precision.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    doc_meta = doc.metadata or {}

    # 1. Title Extraction
    title = ""
    if api_metadata and api_metadata.get("title"):
        title = api_metadata["title"]
    elif doc_meta.get("title") and len(doc_meta["title"].strip()) > 4:
        title = doc_meta["title"].strip()
    
    # 2. Page-by-Page Extraction
    all_blocks: List[Dict[str, Any]] = []
    equations_count = 0
    tables: List[TableSummary] = []
    first_page_spans: List[Dict[str, Any]] = []

    for page_idx in range(page_count):
        page = doc[page_idx]
        text_page = page.get_text("dict")
        
        for block in text_page.get("blocks", []):
            if block.get("type") == 0:  # Text block
                block_lines: List[str] = []
                max_font_size = 0.0
                is_bold = False

                for line in block.get("lines", []):
                    line_spans_text: List[str] = []
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        line_spans_text.append(span_text)
                        size = span.get("size", 0.0)
                        if size > max_font_size:
                            max_font_size = size
                        flags = span.get("flags", 0)
                        if flags & 2 or "bold" in span.get("font", "").lower():
                            is_bold = True

                        if page_idx == 0:
                            first_page_spans.append(span)

                    line_str = "".join(line_spans_text).strip()
                    if line_str:
                        block_lines.append(line_str)
                        # Check for equation labels like (1) or (2.3) at end of line
                        if EQUATION_LABEL_PATTERN.search(line_str):
                            equations_count += 1

                raw_block_text = "\n".join(block_lines)
                clean_block = clean_text(raw_block_text)

                if clean_block:
                    all_blocks.append({
                        "page": page_idx,
                        "text": clean_block,
                        "raw_text": raw_block_text,
                        "max_font_size": max_font_size,
                        "is_bold": is_bold,
                        "bbox": block.get("bbox"),
                    })

                    # Table detection
                    table_match = TABLE_CAPTION_PATTERN.match(clean_block)
                    if table_match:
                        tables.append(
                            TableSummary(
                                table_index=len(tables) + 1,
                                caption=clean_block[:250],
                                context_snippet=clean_block[:400],
                            )
                        )

    # Fallback title from page 0 font sizes if still empty
    if not title and first_page_spans:
        # Sort spans by font size descending
        valid_spans = [s for s in first_page_spans if len(s.get("text", "").strip()) > 3]
        if valid_spans:
            valid_spans.sort(key=lambda s: s.get("size", 0), reverse=True)
            top_size = valid_spans[0].get("size", 0)
            title_parts = [
                s.get("text", "").strip()
                for s in valid_spans
                if abs(s.get("size", 0) - top_size) < 1.0
            ]
            title = " ".join(title_parts)

    title = clean_text(title) or "Untitled Document"

    # 3. Authors & Metadata
    authors: List[str] = []
    affiliations: List[str] = []
    pub_date: Optional[str] = None
    doi: Optional[str] = None

    if api_metadata:
        authors = api_metadata.get("authors", [])
        affiliations = api_metadata.get("affiliations", [])
        pub_date = api_metadata.get("publication_date")
        doi = api_metadata.get("doi")

    if not authors and doc_meta.get("author"):
        # Split author string
        authors = [a.strip() for a in re.split(r"[,;]", doc_meta["author"]) if a.strip()]

    # 4. Sections & Abstract Detection
    sections: List[PaperSection] = []
    current_heading = "Header"
    current_paragraphs: List[str] = []
    abstract_text = ""
    references_text = ""

    for block in all_blocks:
        b_text = block["text"]
        is_heading = False

        # Rule 1: Regex match
        if HEADING_PATTERN.match(b_text) and len(b_text) < 100:
            is_heading = True
        # Rule 2: Short bold line
        elif block["is_bold"] and len(b_text.split()) <= 8 and len(b_text) < 60:
            if re.match(r"^[0-9IVXLCDM]+\.?\s+[A-Z]", b_text):
                is_heading = True

        if is_heading:
            if current_paragraphs:
                section_body = " ".join(current_paragraphs).strip()
                sections.append(PaperSection(heading=current_heading, text=section_body))
                if current_heading.lower() == "abstract":
                    abstract_text = section_body
                elif "reference" in current_heading.lower() or "bibliography" in current_heading.lower():
                    references_text = section_body
                current_paragraphs = []
            current_heading = b_text
        else:
            current_paragraphs.append(b_text)

    # Flush final section
    if current_paragraphs:
        section_body = " ".join(current_paragraphs).strip()
        sections.append(PaperSection(heading=current_heading, text=section_body))
        if current_heading.lower() == "abstract":
            abstract_text = section_body
        elif "reference" in current_heading.lower() or "bibliography" in current_heading.lower():
            references_text = section_body

    # If abstract was provided by ArXiv API, use API abstract for 100% accuracy
    if api_metadata and api_metadata.get("abstract"):
        abstract_text = api_metadata["abstract"]

    # If abstract still not found, search in first section
    if not abstract_text and sections:
        first_sec = sections[0].text
        ab_match = re.search(r"\babstract\b[:\s]*(.*)", first_sec, re.IGNORECASE)
        if ab_match:
            abstract_text = ab_match.group(1)[:2500].strip()

    # 5. Citations extraction
    citations = extract_citations_from_text(references_text)

    # 6. Overall stats
    full_text = " ".join(s.text for s in sections)
    total_chars = len(full_text)
    total_words = len(full_text.split())

    metadata = PaperMetadata(
        title=title,
        authors=authors,
        affiliations=affiliations,
        publication_date=pub_date or doc_meta.get("creationDate"),
        arxiv_id=arxiv_id,
        doi=doi,
        source_url=pdf_url,
    )

    stats = PaperStats(
        pages=page_count,
        characters=total_chars,
        words=total_words,
        citations_count=len(citations),
        sections_count=len(sections),
        parse_duration_secs=round(duration_secs, 3),
    )

    return PaperStructuredOutput(
        status="SUCCESS",
        source_input=source_input,
        metadata=metadata,
        abstract=abstract_text,
        sections=sections,
        equations_count=equations_count,
        tables_summary=tables,
        citations=citations,
        stats=stats,
    )
