"""Reranking — a cross-encoder via LangChain's ContextualCompressionRetriever.

The ensemble retrieves a wide set of candidates; the cross-encoder then re-scores
each (question, chunk) pair *together* and keeps the best few. We wrap it as a
ContextualCompressionRetriever so the whole thing is still a single LangChain
retriever: ask it for documents and you get the reranked top-n.

If the cross-encoder model can't load (offline / not installed), we skip
reranking and return the base retriever so the system still works — just without
the precision boost.
"""
from langchain.retrievers import ContextualCompressionRetriever

from common.config import settings
from common.logging_setup import get_logger

logger = get_logger(__name__)


def build_reranking_retriever(base_retriever):
    if not settings.use_reranker:
        return base_retriever
    try:
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=model, top_n=settings.rerank_top_n)
        logger.info("cross-encoder reranker enabled")
        return ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever)
    except Exception as e:
        logger.warning("reranker unavailable, using base retriever: %s", e)
        return base_retriever
