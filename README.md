# Academic Paper & ArXiv PDF-to-Structured-JSON Extractor 📄⚡

> **Fast, production-grade academic paper extractor converting ArXiv papers and scientific PDFs into clean, schema-validated JSON with references, equations, tables, and LLM-ready RAG chunks.**

[![Apify Actor](https://img.shields.io/badge/Apify-Actor-blue?logo=apify)](https://apify.com)
[![Pay-Per-Event](https://img.shields.io/badge/Pricing-Pay--Per--Event%20($0.005/paper)-success)](https://apify.com)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 💡 Why This Actor?

Raw scientific PDFs are notoriously difficult for AI pipelines. They contain multi-column layouts, hyphenated line wraps, scattered mathematical formulas, and complex reference lists that destroy context when ingested into RAG or LLMs.

This Actor solves that problem by combining **ArXiv's authoritative metadata API** with **PyMuPDF (`fitz`) layout analysis**:
- **0% Hallucination**: Pure deterministic parsing with zero GPU overhead and sub-3-second execution per paper.
- **Fair Pay-Per-Event (PPE) Pricing**: Charged at **\$0.005 per successfully parsed paper**. You are **never charged** for failed downloads, 404s, or invalid files.
- **LLM & Vector DB Ready**: Includes pre-chunked, token-bounded overlapping text blocks tagged with section headers (`[Section: Methodology] ...`), ready to drop into Pinecone, Qdrant, Chroma, or Weaviate.

---

## 🚀 Key Features

| Capability | Description |
| :--- | :--- |
| **Hybrid Metadata Resolution** | Accepts ArXiv IDs (`1706.03762`), ArXiv URLs, DOIs, or direct PDF links. Automatically queries ArXiv API for exact author list, publication date, and canonical DOI. |
| **Section Decomposition** | Detects abstract, numbered headings (e.g. `1. Introduction`, `2. Methodology`), and subsections with automated hyphenation repair. |
| **Math & Table Analytics** | Identifies display equation counts and table captions with header context snippets. |
| **Citation Network Parsing** | Splits bibliography entries into structured citations with publication year, guessed paper title, and cited ArXiv IDs. |
| **RAG-Ready Chunks** | Generates overlapping token-bounded chunks preserving section hierarchy breadcrumbs for optimal embedding similarity. |
| **Resilient Networking** | Built with polite headers, intelligent rate limiting, and exponential backoff. |

---

## 📥 Input Configuration

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `urls_or_ids` | `array` | *Required* | List of ArXiv IDs (`1706.03762`), ArXiv URLs, or direct academic PDF links. |
| `extract_chunks` | `boolean` | `true` | Generate pre-tokenized overlapping chunks for RAG pipelines. |
| `chunk_size` | `integer` | `1200` | Target chunk character size (~300 tokens). |
| `chunk_overlap` | `integer` | `200` | Character overlap between consecutive chunks. |
| `max_papers` | `integer` | `50` | Maximum papers to process in a single batch. |
| `timeout_secs` | `integer` | `45` | Download timeout per PDF in seconds. |
| `proxyConfiguration` | `object` | `null` | Optional Apify Proxy configuration for large scraping runs. |

### Input Example
```json
{
  "urls_or_ids": [
    "1706.03762",
    "https://arxiv.org/abs/2312.12345"
  ],
  "extract_chunks": true,
  "chunk_size": 1200,
  "chunk_overlap": 200,
  "max_papers": 10
}
```

---

## 📤 Output Schema

Each dataset item provides structured JSON validated against Pydantic v2:

```json
{
  "status": "SUCCESS",
  "source_input": "1706.03762",
  "metadata": {
    "title": "Attention Is All You Need",
    "authors": [
      "Ashish Vaswani",
      "Noam Shazeer",
      "Niki Parmar",
      "Jakob Uszkoreit",
      "Llion Jones",
      "Aidan N. Gomez",
      "Lukasz Kaiser",
      "Illia Polosukhin"
    ],
    "affiliations": ["Google Brain", "Google Research", "University of Toronto"],
    "publication_date": "2017-06-12T17:57:34Z",
    "arxiv_id": "1706.03762",
    "doi": "10.48550/arXiv.1706.03762",
    "source_url": "https://arxiv.org/pdf/1706.03762.pdf"
  },
  "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks... We propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.",
  "sections": [
    {
      "heading": "1. Introduction",
      "text": "Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling..."
    },
    {
      "heading": "2. Background",
      "text": "The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building blocks..."
    }
  ],
  "equations_count": 14,
  "tables_summary": [
    {
      "table_index": 1,
      "caption": "Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types.",
      "context_snippet": "Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length"
    }
  ],
  "citations": [
    {
      "raw_text": "Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.",
      "title_guess": "Layer normalization",
      "year": 2016,
      "arxiv_id_if_present": "1607.06450"
    }
  ],
  "chunks": [
    {
      "chunk_id": 0,
      "section_heading": "1. Introduction",
      "text": "[1. Introduction]\nRecurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches...",
      "char_count": 940,
      "approx_token_count": 235
    }
  ],
  "stats": {
    "pages": 11,
    "characters": 38450,
    "words": 6210,
    "citations_count": 39,
    "sections_count": 8,
    "parse_duration_secs": 2.45
  }
}
```

---

## 🔌 Integration Examples

### Python (Apify Client)
```python
from apify_client import ApifyClient

client = ApifyClient("YOUR_APIFY_API_TOKEN")

run_input = {
    "urls_or_ids": ["2312.12345", "1706.03762"],
    "extract_chunks": True,
}

# Run Actor and wait for completion
run = client.actor("your-username/arxiv-pdf-structured-extractor").call(run_input=run_input)

# Fetch structured results
for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    print(f"Title: {item['metadata']['title']}")
    print(f"Chunks: {len(item.get('chunks', []))}")
```

### Ready-to-Import n8n Workflow Node
Copy and paste this into any n8n canvas to trigger paper parsing on schedule or webhook:

```json
{
  "nodes": [
    {
      "parameters": {
        "method": "POST",
        "url": "https://api.apify.com/v2/acts/your-username~arxiv-pdf-structured-extractor/run-sync-get-dataset-items",
        "authentication": "genericCredentialType",
        "genericAuthType": "httpHeaderAuth",
        "sendHeaders": true,
        "headerParameters": {
          "parameters": [
            {
              "name": "Authorization",
              "value": "Bearer YOUR_APIFY_API_TOKEN"
            }
          ]
        },
        "sendBody": true,
        "specifyBody": "json",
        "jsonBody": "={\n  \"urls_or_ids\": [\"{{ $json.paper_id_or_url }}\"],\n  \"extract_chunks\": true,\n  \"chunk_size\": 1200,\n  \"chunk_overlap\": 200\n}"
      },
      "id": "e4f5a6b7-c8d9-40e1-b2c3-d4e5f6a7b8c9",
      "name": "Parse Paper with Apify Actor",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [460, 300]
    }
  ]
}
```

---

## 💰 Pricing: Pay-Per-Event (PPE)

- **\$0.005 per paper parsed**: Charged only when a paper is successfully parsed and written to the dataset.
- **\$0.00 on failures**: If an invalid PDF or 404 is encountered, you pay nothing.
- No recurring subscription or monthly minimums.

---

## 🛡️ License

MIT License. Free for commercial and non-commercial automation pipelines.
