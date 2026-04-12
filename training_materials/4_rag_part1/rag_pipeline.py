#!/usr/bin/env python3
"""
rag_pipeline.py — Complete RAG pipeline over TechStore documents.

GenAI Decoded by Nij — Section 4: RAG Part 1

Usage:
    python rag_pipeline.py "What is the return policy for electronics?"
    python rag_pipeline.py "How many PTO days after 4 years?" --debug
    python rag_pipeline.py "Best laptop for ML?" --chunks 5
"""

import argparse
import os
import sys
import glob
import textwrap
import chromadb
from openai import OpenAI


# ── Configuration ──────────────────────────────────────────────
DOCS_DIR = "sample_docs"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DEFAULT_K = 3

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about TechStore.

RULES:
- Answer ONLY using the provided context
- If the answer is not in the context, say "I don't have that information in my knowledge base."
- Do NOT use your general knowledge — only what's in the context
- Cite the source document when possible
- Be concise and direct"""


# ── Document Processing ───────────────────────────────────────

def load_documents(docs_dir):
    """Load all .txt files from the docs directory."""
    documents = []
    for filepath in sorted(glob.glob(f"{docs_dir}/*.txt")):
        with open(filepath, "r") as f:
            content = f.read()
        documents.append({
            "name": os.path.basename(filepath),
            "content": content,
        })
    return documents


def chunk_recursive(text, max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text by paragraphs, merge small ones, respect structure."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + "\n\n" + para if overlap > 0 else para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ── Embedding & Storage ───────────────────────────────────────

def build_index(client, documents):
    """Chunk, embed, and store all documents in ChromaDB."""
    chroma = chromadb.Client()

    # Delete existing collection if it exists
    try:
        chroma.delete_collection("techstore_docs")
    except Exception:
        pass

    collection = chroma.create_collection(
        name="techstore_docs",
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []
    for doc in documents:
        chunks = chunk_recursive(doc["content"])
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['name']}_chunk_{i}",
                "text": chunk_text,
                "source": doc["name"],
                "chunk_index": i,
            })

    # Embed all chunks
    chunk_texts = [c["text"] for c in all_chunks]
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk_texts)
    embeddings = [item.embedding for item in response.data]

    # Store in ChromaDB
    collection.add(
        ids=[c["id"] for c in all_chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in all_chunks],
    )

    return collection, all_chunks


# ── Search & Generation ───────────────────────────────────────

def search(client, collection, query, n_results=DEFAULT_K):
    """Search the vector database for relevant chunks."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results


def generate_answer(client, question, results):
    """Assemble prompt from retrieved chunks and generate answer."""
    context_parts = []
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        source = results['metadatas'][0][i]['source']
        score = 1 - results['distances'][0][i]
        context_parts.append(f"[Source: {source}, Relevance: {score:.3f}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""Context (retrieved from TechStore knowledge base):
---
{context}
---

Question: {question}"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=500
    )

    return response.choices[0].message.content.strip(), response.usage


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about TechStore using RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_pipeline.py "What is the return policy for electronics?"
  python rag_pipeline.py "How many PTO days after 4 years?" --debug
  python rag_pipeline.py "Best laptop for ML?" --chunks 5
        """
    )
    parser.add_argument("question", help="Your question about TechStore")
    parser.add_argument("--chunks", "-k", type=int, default=DEFAULT_K,
                        help=f"Number of chunks to retrieve (default: {DEFAULT_K})")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Show retrieved chunks before generating answer")

    args = parser.parse_args()

    # Setup
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY first.")
        sys.exit(1)

    client = OpenAI()

    # Load and index documents
    print("📄 Loading documents...", end=" ", flush=True)
    documents = load_documents(DOCS_DIR)
    print(f"{len(documents)} found.")

    print("🔢 Building index...", end=" ", flush=True)
    collection, all_chunks = build_index(client, documents)
    print(f"{len(all_chunks)} chunks indexed.")

    # Search
    results = search(client, collection, args.question, n_results=args.chunks)

    # Debug output
    if args.debug:
        print(f"\n🔍 Retrieved {args.chunks} chunks:")
        print("─" * 50)
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            source = results['metadatas'][0][i]['source']
            score = 1 - results['distances'][0][i]
            relevance = "🟢" if score > 0.5 else "🟡" if score > 0.35 else "🔴"
            print(f"\n  [{i+1}] {relevance} {score:.4f} | {source}")
            print(f"      \"{doc[:120]}...\"")
        print()

    # Generate
    print("─" * 50)
    answer, usage = generate_answer(client, args.question, results)
    print(answer)

    # Footer
    cost = usage.prompt_tokens * 0.15e-6 + usage.completion_tokens * 0.60e-6
    print(f"\n{'─' * 50}")
    print(f"Tokens: {usage.total_tokens} | Cost: ~${cost:.6f} | Chunks: {args.chunks}")


if __name__ == "__main__":
    main()
