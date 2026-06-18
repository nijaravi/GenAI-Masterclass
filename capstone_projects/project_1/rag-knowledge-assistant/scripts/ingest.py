"""CLI: ingest documents into the configured vector store.

Usage:
    python scripts/ingest.py --data-dir data/raw
"""
from __future__ import annotations

import argparse

from rag.config import settings
from rag.ingestion.pipeline import run_ingestion
from rag.logging_config import configure_logging


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    args = ap.parse_args()
    configure_logging(settings.log_level, json_logs=False)
    n = run_ingestion(args.data_dir)
    print(f"Ingested {n} chunks into {settings.vector_store}.")


if __name__ == "__main__":
    main()
