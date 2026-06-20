"""Run the data-loading pipeline from the command line.

    python -m data_pipeline.run_ingest

load (LangChain loader) -> split (header + token splitters) -> embed + index
(Chroma). Safe to re-run; chunks upsert by id.
"""
from common.config import settings
from common.logging_setup import get_logger, setup_logging
from data_pipeline.indexer import index_chunks
from data_pipeline.loader import load_documents
from data_pipeline.splitter import split_documents

logger = get_logger("ingest")


def main() -> None:
    setup_logging(settings.log_level)
    logger.info("starting ingestion")

    docs = load_documents()
    chunks = split_documents(docs)
    total = index_chunks(chunks)

    logger.info("done: %d documents -> %d chunks -> %d in store",
                len(docs), len(chunks), total)


if __name__ == "__main__":
    main()
