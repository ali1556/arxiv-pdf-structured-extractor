from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field, HttpUrl


class ActorInput(BaseModel):
    urls_or_ids: List[str] = Field(
        ...,
        description="List of ArXiv IDs, ArXiv URLs, or direct academic PDF links.",
        min_length=1,
    )
    extract_chunks: bool = Field(
        default=True,
        description="Whether to generate overlapping chunks for RAG pipelines.",
    )
    chunk_size: int = Field(
        default=1200,
        ge=200,
        le=8000,
        description="Target character size per chunk.",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap character size between chunks.",
    )
    max_papers: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum papers to parse in this run.",
    )
    timeout_secs: int = Field(
        default=45,
        ge=10,
        le=300,
        description="Timeout for HTTP requests.",
    )
    proxyConfiguration: Optional[dict[str, Any]] = Field(
        default=None,
        description="Apify proxy configuration.",
    )


class Citation(BaseModel):
    raw_text: str = Field(description="Full text of the citation entry.")
    title_guess: Optional[str] = Field(default=None, description="Guessed paper title from reference entry.")
    year: Optional[int] = Field(default=None, description="Publication year.")
    arxiv_id_if_present: Optional[str] = Field(default=None, description="Extracted ArXiv ID if cited.")


class PaperSection(BaseModel):
    heading: str = Field(description="Section heading or title.")
    text: str = Field(description="Clean text content within this section.")


class Chunk(BaseModel):
    chunk_id: int = Field(description="Zero-based sequential chunk index.")
    section_heading: str = Field(description="Section where this chunk originates.")
    text: str = Field(description="Chunk content.")
    char_count: int = Field(description="Character count.")
    approx_token_count: int = Field(description="Estimated token count (~4 chars/token).")


class PaperMetadata(BaseModel):
    title: str = Field(description="Paper title.")
    authors: List[str] = Field(default_factory=list, description="List of paper authors.")
    affiliations: List[str] = Field(default_factory=list, description="Author affiliations if detected.")
    publication_date: Optional[str] = Field(default=None, description="Publication or submission date.")
    arxiv_id: Optional[str] = Field(default=None, description="Canonical ArXiv ID if available.")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier (DOI) if available.")
    source_url: str = Field(description="Direct URL where the PDF was fetched from.")


class TableSummary(BaseModel):
    table_index: int = Field(description="Sequential index of detected table.")
    caption: str = Field(description="Table caption or heading.")
    context_snippet: Optional[str] = Field(default=None, description="Surrounding text or table header preview.")


class PaperStats(BaseModel):
    pages: int = Field(description="Total PDF page count.")
    characters: int = Field(description="Total extracted character count.")
    words: int = Field(description="Total word count.")
    citations_count: int = Field(description="Total parsed citations.")
    sections_count: int = Field(description="Total sections identified.")
    parse_duration_secs: float = Field(description="Time taken to fetch and parse in seconds.")


class PaperStructuredOutput(BaseModel):
    status: str = Field(default="SUCCESS", description="SUCCESS or FAILED.")
    source_input: str = Field(description="Raw input item requested.")
    metadata: PaperMetadata = Field(description="Academic metadata.")
    abstract: str = Field(description="Paper abstract.")
    sections: List[PaperSection] = Field(default_factory=list, description="Extracted sections.")
    equations_count: int = Field(default=0, description="Estimated count of display equations/formulas.")
    tables_summary: List[TableSummary] = Field(default_factory=list, description="Detected tables.")
    citations: List[Citation] = Field(default_factory=list, description="Parsed references.")
    chunks: Optional[List[Chunk]] = Field(default=None, description="LLM-ready RAG chunks.")
    stats: PaperStats = Field(description="Extraction statistics.")
