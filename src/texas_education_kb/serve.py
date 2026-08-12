from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from .query import build_answer_prompt
from .schema import QueryRequest, RetrievalResult
from .store import LocalRetrievalStore


def create_app(index_path: Path) -> FastAPI:
    store = LocalRetrievalStore.from_jsonl(index_path)
    app = FastAPI(title="Texas Education Local Knowledge Base")

    @app.post("/query", response_model=RetrievalResult)
    def query(request: QueryRequest) -> RetrievalResult:
        citations = store.search(request.question, top_k=request.top_k, topic=request.topic)
        return RetrievalResult(
            question=request.question,
            citations=citations,
            answer_prompt=build_answer_prompt(request.question, citations),
        )

    @app.get("/health")
    def health() -> dict[str, int | str]:
        return {"status": "ok", "chunks": len(store.chunks)}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.index), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
