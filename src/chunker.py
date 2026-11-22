from __future__ import annotations

import re
from typing import List
from src.models import Chunk, PaperSection


def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences using punctuation boundaries."""
    # Split by period/question/exclamation followed by space and uppercase letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def generate_rag_chunks(
    sections: List[PaperSection],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[Chunk]:
    """
    Creates token-bounded overlapping chunks preserving section context
    for RAG (Retrieval-Augmented Generation) pipelines and vector databases.
    """
    chunks: List[Chunk] = []
    chunk_id = 0

    for section in sections:
        # Skip empty sections or references list from vector chunks (optional, keep references separate)
        if not section.text:
            continue

        sec_heading = section.heading.strip()
        sentences = split_into_sentences(section.text)
        if not sentences:
            continue

        current_sentences: List[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)

            # If adding this sentence exceeds chunk_size, create chunk
            if current_len + sent_len > chunk_size and current_sentences:
                chunk_body = " ".join(current_sentences)
                # Add section breadcrumb for LLM grounding
                final_text = f"[{sec_heading}]\n{chunk_body}"
                
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        section_heading=sec_heading,
                        text=final_text,
                        char_count=len(final_text),
                        approx_token_count=max(1, round(len(final_text) / 4)),
                    )
                )
                chunk_id += 1

                # Calculate overlap: retain trailing sentences that fit into chunk_overlap
                overlap_sentences: List[str] = []
                overlap_len = 0
                for prev_sent in reversed(current_sentences):
                    if overlap_len + len(prev_sent) <= chunk_overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_len += len(prev_sent)
                    else:
                        break

                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(sent)
            current_len += sent_len

        # Flush any remaining sentences in this section
        if current_sentences:
            chunk_body = " ".join(current_sentences)
            final_text = f"[{sec_heading}]\n{chunk_body}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    section_heading=sec_heading,
                    text=final_text,
                    char_count=len(final_text),
                    approx_token_count=max(1, round(len(final_text) / 4)),
                )
            )
            chunk_id += 1

    return chunks
