from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from .chunking import chunks_from_documents
from .schema import DocumentSource, RawDocument
from .sources import load_sources


def document_from_source(source: DocumentSource, raw_dir: Path) -> RawDocument:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if source.path:
        path = source.path
    elif source.url:
        path = download_source(source, raw_dir)
    else:
        raise ValueError(f"{source.id} must define either url or path")

    text = extract_text(path)
    return RawDocument(
        source_id=source.id,
        title=source.title,
        text=text,
        source_url=str(source.url) if source.url else None,
        source_path=str(path),
        kind=source.kind,
        agency=source.agency,
        topics=source.topics,
        year=source.year,
    )


def download_source(source: DocumentSource, raw_dir: Path) -> Path:
    assert source.url is not None
    response = requests.get(str(source.url), timeout=60)
    response.raise_for_status()
    suffix = _suffix_from_content_type(response.headers.get("content-type", ""), str(source.url))
    path = raw_dir / f"{source.id}{suffix}"
    path.write_bytes(response.content)
    return path


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if suffix in {".html", ".htm"}:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text("\n", strip=True)
    return path.read_text(encoding="utf-8", errors="ignore")


def build_index(sources_path: Path, raw_dir: Path, index_path: Path) -> None:
    sources = load_sources(sources_path)
    documents = [document_from_source(source, raw_dir) for source in sources]
    chunks = chunks_from_documents(documents)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")


def _suffix_from_content_type(content_type: str, url: str) -> str:
    lower_url = url.lower()
    if lower_url.endswith(".pdf") or "pdf" in content_type:
        return ".pdf"
    if lower_url.endswith(".html") or lower_url.endswith(".htm") or "html" in content_type:
        return ".html"
    return ".txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local Texas education KB retrieval index.")
    parser.add_argument("--sources", type=Path, default=Path("sources/texas_education_sources.yaml"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--index", type=Path, default=Path("data/processed/chunks.jsonl"))
    args = parser.parse_args()
    build_index(args.sources, args.raw_dir, args.index)
    print(json.dumps({"index": str(args.index)}, indent=2))


if __name__ == "__main__":
    main()
