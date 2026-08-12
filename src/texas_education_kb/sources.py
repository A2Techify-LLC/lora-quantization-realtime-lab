from __future__ import annotations

from pathlib import Path

import yaml

from .schema import DocumentSource


def load_sources(path: Path) -> list[DocumentSource]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return [DocumentSource(**item) for item in payload.get("sources", [])]
