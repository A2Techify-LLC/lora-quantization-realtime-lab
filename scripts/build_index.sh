#!/usr/bin/env bash
set -euo pipefail

python -m texas_education_kb.ingest \
  --sources "${1:-sources/texas_education_sources.yaml}" \
  --raw-dir "${2:-data/raw}" \
  --index "${3:-data/processed/chunks.jsonl}"
