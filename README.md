# Texas Education Local Knowledge Base

A local-first knowledge-base starter for Texas education policy, guidance, law, accountability, graduation, and special education material.

The goal is citation-first retrieval, not blind generation. The project downloads or reads trusted sources, chunks them locally, builds a searchable JSONL index, and returns cited passages plus a prompt that can be sent to a local model.

## What You Learn

- Catalog Texas education sources in YAML.
- Download or ingest local TEA, Texas Education Code, Texas Administrative Code, and district policy documents.
- Parse text, HTML, and PDFs.
- Chunk documents while preserving source metadata.
- Search the corpus locally without hosted APIs.
- Serve a small FastAPI query endpoint for local applications.
- Keep citations attached to every answer workflow.

## Project Layout

```text
sources/                    Source catalogs for Texas education documents
data/raw/                   Downloaded source files
data/processed/             Generated chunk indexes
src/texas_education_kb/     Python package for ingestion, chunking, retrieval, and serving
tests/                      Lightweight behavior tests
```

## Quick Start

Use Python 3.10, 3.11, or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
pytest
```

Build the sample local index:

```bash
python -m texas_education_kb.ingest \
  --sources sources/sample_local_sources.yaml \
  --raw-dir data/raw \
  --index data/processed/chunks.jsonl
```

Query it:

```bash
python -m texas_education_kb.query \
  "How are special education services handled in Texas?" \
  --index data/processed/chunks.jsonl
```

The output includes citations and an `answer_prompt` that can be passed to a local model such as Llama, Qwen, Mistral, or Nemotron.

## Texas Source Catalog

The starter catalog is [sources/texas_education_sources.yaml](sources/texas_education_sources.yaml). It includes entry points for:

- TEA graduation guidance
- TEA accountability guidance
- TEA special education guidance
- Texas Education Code
- Texas Administrative Code Title 19

Build an index from the public source catalog:

```bash
scripts/build_index.sh sources/texas_education_sources.yaml data/raw data/processed/chunks.jsonl
```

Some public sites change HTML structure or link to PDFs from landing pages. For a production corpus, download the exact PDFs/manuals you want, save them under `data/raw/`, and point the YAML catalog at local paths. That gives reproducible indexes.

## Serve Locally

Install the serving extra:

```bash
pip install -e ".[serve]"
```

Start the API:

```bash
python -m texas_education_kb.serve \
  --index data/processed/chunks.jsonl
```

Send a query:

```bash
curl http://127.0.0.1:8000/query \
  -H 'content-type: application/json' \
  -d '{"question":"What sources explain Texas school accountability?","top_k":3}'
```

## Why This Direction

For education policy, retrieval with citations matters more than fine-tuning. Texas education content changes by school year, rulemaking cycle, legislative session, and district. A local RAG knowledge base makes updates explicit: refresh the source catalog, rebuild the index, and compare cited passages.

Fine-tuning can still be added later for repeated workflows, such as classifying questions by topic or formatting answers for a help desk. The knowledge should remain in the local corpus.
