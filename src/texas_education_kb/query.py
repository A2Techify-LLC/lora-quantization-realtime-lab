from __future__ import annotations

import argparse
import json
from pathlib import Path

from .schema import RetrievalResult
from .store import LocalRetrievalStore


def build_answer_prompt(question: str, citations: list) -> str:
    context = "\n\n".join(
        f"[{index}] {citation.title} ({citation.source_id})\n{citation.excerpt}"
        for index, citation in enumerate(citations, start=1)
    )
    return (
        "Answer the question using only the cited Texas education context below. "
        "If the context is insufficient, say what source is missing. Cite sources by bracket number.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )


def query_index(index_path: Path, question: str, top_k: int = 5, topic: str | None = None) -> RetrievalResult:
    store = LocalRetrievalStore.from_jsonl(index_path)
    citations = store.search(question, top_k=top_k, topic=topic)
    return RetrievalResult(
        question=question,
        citations=citations,
        answer_prompt=build_answer_prompt(question, citations),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local Texas education KB.")
    parser.add_argument("question")
    parser.add_argument("--index", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--topic")
    args = parser.parse_args()
    result = query_index(args.index, args.question, top_k=args.top_k, topic=args.topic)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
