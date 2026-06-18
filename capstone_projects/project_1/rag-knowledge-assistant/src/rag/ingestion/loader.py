"""Load mock internal documents from the data directory.

Each markdown file carries a tiny YAML-ish front-matter block (title/category)
which we parse without a hard pyyaml dependency. In a real deployment this layer
is where connectors live (Confluence, SharePoint, S3, Google Drive) — the rest
of the pipeline only ever sees `Document` objects, so swapping the source is a
one-file change.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..logging_config import get_logger
from ..models import Document

logger = get_logger(__name__)


def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("---", 3)
    if end == -1:
        return {}, raw
    header = raw[3:end].strip()
    body = raw[end + 3 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body


def load_documents(data_dir: str | Path) -> list[Document]:
    data_dir = Path(data_dir)
    docs: list[Document] = []
    for path in sorted(data_dir.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(raw)
        rel = str(path.relative_to(data_dir))
        # Deterministic id so re-ingestion overwrites rather than duplicates.
        doc_id = hashlib.sha1(rel.encode()).hexdigest()[:12]
        docs.append(
            Document(
                doc_id=doc_id,
                title=meta.get("title", path.stem.replace("-", " ").title()),
                source=rel,
                category=meta.get("category", "general"),
                text=body,
                metadata={"path": rel},
            )
        )
    logger.info("loaded documents", extra={"extra": {"count": len(docs)}})
    return docs
