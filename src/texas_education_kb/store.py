from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .schema import Citation, DocumentChunk


TOKEN_RE = re.compile(r"[a-z0-9']+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "with",
}


def load_chunks(path: Path) -> list[DocumentChunk]:
    with path.open("r", encoding="utf-8") as handle:
        return [DocumentChunk(**json.loads(line)) for line in handle if line.strip()]


class LocalRetrievalStore:
    """Small dependency-light retriever for local demos and tests.

    It uses TF-IDF style lexical similarity. Swap this behind the same API for
    Chroma/Qdrant plus sentence-transformer embeddings when the corpus grows.
    """

    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.vectors = [_term_counts(chunk.text) for chunk in chunks]
        document_frequency: Counter[str] = Counter()
        for vector in self.vectors:
            document_frequency.update(vector.keys())
        count = max(len(chunks), 1)
        self.idf = {term: math.log((1 + count) / (1 + frequency)) + 1 for term, frequency in document_frequency.items()}

    @classmethod
    def from_jsonl(cls, path: Path) -> "LocalRetrievalStore":
        return cls(load_chunks(path))

    def search(self, question: str, top_k: int = 5, topic: str | None = None) -> list[Citation]:
        query = _term_counts(question)
        scored: list[tuple[float, DocumentChunk]] = []
        for chunk, vector in zip(self.chunks, self.vectors, strict=True):
            if topic and topic.lower() not in {item.lower() for item in chunk.topics}:
                continue
            score = _cosine(query, vector, self.idf)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            Citation(
                chunk_id=chunk.id,
                title=chunk.title,
                source_id=chunk.source_id,
                source_url=chunk.source_url,
                source_path=chunk.source_path,
                score=round(score, 4),
                excerpt=_excerpt(chunk.text, question),
            )
            for score, chunk in scored[:top_k]
        ]


def _term_counts(text: str) -> Counter[str]:
    tokens = [token for token in TOKEN_RE.findall(text.lower()) if token not in STOP_WORDS and len(token) > 1]
    return Counter(tokens)


def _cosine(left: Counter[str], right: Counter[str], idf: dict[str, float]) -> float:
    shared = set(left) & set(right)
    numerator = sum(left[term] * right[term] * idf.get(term, 1) ** 2 for term in shared)
    left_norm = math.sqrt(sum((count * idf.get(term, 1)) ** 2 for term, count in left.items()))
    right_norm = math.sqrt(sum((count * idf.get(term, 1)) ** 2 for term, count in right.items()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _excerpt(text: str, question: str, max_chars: int = 420) -> str:
    query_terms = set(_term_counts(question))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    best = max(sentences, key=lambda sentence: len(query_terms & set(_term_counts(sentence))), default=text)
    best = best.strip() or text.strip()
    if len(best) <= max_chars:
        return best
    return best[: max_chars - 3].rstrip() + "..."
