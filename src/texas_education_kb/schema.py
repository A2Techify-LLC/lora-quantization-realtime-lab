from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


SourceKind = Literal["law", "rule", "manual", "guidance", "district_policy", "reference"]


class DocumentSource(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    title: str
    url: HttpUrl | None = None
    path: Path | None = None
    kind: SourceKind = "reference"
    agency: str = "Texas Education Agency"
    topics: list[str] = Field(default_factory=list)
    year: int | None = None


class RawDocument(BaseModel):
    source_id: str
    title: str
    text: str = Field(min_length=1)
    source_url: str | None = None
    source_path: str | None = None
    kind: SourceKind = "reference"
    agency: str = "Texas Education Agency"
    topics: list[str] = Field(default_factory=list)
    year: int | None = None
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    id: str
    source_id: str
    title: str
    text: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    source_url: str | None = None
    source_path: str | None = None
    kind: SourceKind = "reference"
    agency: str = "Texas Education Agency"
    topics: list[str] = Field(default_factory=list)
    year: int | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    topic: str | None = None


class Citation(BaseModel):
    chunk_id: str
    title: str
    source_id: str
    source_url: str | None = None
    source_path: str | None = None
    score: float
    excerpt: str


class RetrievalResult(BaseModel):
    question: str
    citations: list[Citation]
    answer_prompt: str
