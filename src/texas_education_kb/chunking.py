from __future__ import annotations

import re
from collections.abc import Iterable
from hashlib import sha1

from .schema import DocumentChunk, RawDocument


PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def normalize_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.replace("\r\n", "\n").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def chunk_text(text: str, max_chars: int = 1400, overlap_chars: int = 180) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")

    paragraphs = [part.strip() for part in PARAGRAPH_SPLIT_RE.split(normalize_text(text)) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.extend(_split_oversized(current, max_chars, overlap_chars))
            overlap = current[-overlap_chars:].strip() if overlap_chars else ""
            current = f"{overlap}\n\n{paragraph}" if overlap else paragraph
    if current:
        chunks.extend(_split_oversized(current, max_chars, overlap_chars))
    return chunks


def chunks_from_documents(
    documents: Iterable[RawDocument],
    max_chars: int = 1400,
    overlap_chars: int = 180,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in documents:
        for ordinal, text in enumerate(chunk_text(document.text, max_chars=max_chars, overlap_chars=overlap_chars)):
            digest = sha1(f"{document.source_id}:{ordinal}:{text}".encode("utf-8")).hexdigest()[:16]
            chunks.append(
                DocumentChunk(
                    id=f"{document.source_id}-{digest}",
                    source_id=document.source_id,
                    title=document.title,
                    text=text,
                    ordinal=ordinal,
                    source_url=document.source_url,
                    source_path=document.source_path,
                    kind=document.kind,
                    agency=document.agency,
                    topics=document.topics,
                    year=document.year,
                )
            )
    return chunks


def _split_oversized(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap_chars)
    return [chunk for chunk in chunks if chunk]
