"""Hybrid retrieval — LangChain's EnsembleRetriever.

EnsembleRetriever runs the dense and sparse retrievers and fuses their results
with Reciprocal Rank Fusion (RRF) under the hood — combining on rank, so we
don't have to normalise cosine scores against BM25 scores. `weights` lets us
lean slightly toward one retriever; 0.5/0.5 is a balanced default.
"""
from langchain.retrievers import EnsembleRetriever


def build_hybrid_retriever(dense_retriever, sparse_retriever) -> EnsembleRetriever:
    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5],
    )
