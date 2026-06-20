"""Step 3 of loading: embed and store the chunks in the vector store.

Builds the LangChain embeddings + Chroma vector store and adds the chunk
Documents. We pass explicit ids (the chunk_id from the splitter) so re-running
ingest upserts in place instead of duplicating.
"""
from langchain_core.documents import Document

from common.embedding import build_embeddings
from common.logging_setup import get_logger
from common.vector_store import build_vectorstore

logger = get_logger(__name__)


def index_chunks(chunks: list[Document]) -> int:
    embeddings = build_embeddings()
    vectorstore = build_vectorstore(embeddings)

    ids = [c.metadata["chunk_id"] for c in chunks]
    logger.info("embedding + indexing %d chunks", len(chunks))
    vectorstore.add_documents(chunks, ids=ids)

    total = vectorstore._collection.count()
    logger.info("vector store now holds %d chunks", total)
    return total
