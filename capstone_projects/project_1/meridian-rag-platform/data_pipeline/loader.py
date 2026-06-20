"""Step 1 of loading: read the markdown files with a LangChain loader.

We use `DirectoryLoader` + `TextLoader` to read everything under
`documents/<category>/*.md`, then enrich each Document's metadata with the
category (the folder name), a clean relative `source`, and a `title` (the first
markdown heading). Downstream everything speaks LangChain `Document`.
"""
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from common.logging_setup import get_logger

logger = get_logger(__name__)

DOCS_DIR = Path(__file__).parent / "documents"


def _title_from(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    loader = DirectoryLoader(
        str(docs_dir), glob="**/*.md",
        loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    # Enrich metadata: category = folder, source = relative path, title = first heading.
    for doc in docs:
        full = Path(doc.metadata.get("source", ""))
        category = full.parent.name
        rel = f"{category}/{full.name}"
        doc.metadata.update({
            "source": rel,
            "category": category,
            "title": _title_from(doc.page_content, full.stem),
        })
    logger.info("loaded %d documents", len(docs))
    return docs
