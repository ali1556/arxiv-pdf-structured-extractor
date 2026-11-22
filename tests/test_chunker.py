from src.chunker import generate_rag_chunks, split_into_sentences
from src.models import PaperSection


def test_split_into_sentences():
    text = "Attention is all you need. It replaced recurrent layers! Did it improve performance? Yes, significantly."
    sentences = split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "Attention is all you need."
    assert sentences[1] == "It replaced recurrent layers!"


def test_generate_rag_chunks_bounded_and_overlapping():
    sections = [
        PaperSection(
            heading="1. Introduction",
            text=(
                "Sentence one introduces the foundational transformer architecture. "
                "Sentence two discusses the multi-head self-attention mechanism in detail. "
                "Sentence three evaluates cross-attention mechanisms against traditional RNNs. "
                "Sentence four elaborates on feedforward sub-layers and positional encodings."
            ),
        ),
        PaperSection(
            heading="2. Methodology",
            text=(
                "Methodology outlines the model dimension of 512 and 8 parallel attention heads. "
                "Residual connections and layer normalization are applied after each sub-layer. "
                "Optimization is performed using the Adam optimizer with warmup steps."
            ),
        ),
    ]

    # Target small chunk size to trigger chunking and overlap
    chunks = generate_rag_chunks(sections, chunk_size=120, chunk_overlap=40)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.section_heading in ["1. Introduction", "2. Methodology"]
        assert chunk.text.startswith(f"[{chunk.section_heading}]")
        assert chunk.char_count == len(chunk.text)
        assert chunk.approx_token_count > 0
