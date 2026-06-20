# Data Pipeline (Layer 1 — Data Loading)

Everything needed to get documents **into** the vector database, built with
LangChain. Runs **offline**, on its own, separate from the live request path. Run
it once to populate the store, and again whenever the documents change.

## What's here

| File | Job | LangChain pieces |
|------|-----|------------------|
| `documents/` | the source corpus — markdown grouped by category folder | — |
| `loader.py` | read the markdown into `Document`s, enrich metadata | `DirectoryLoader`, `TextLoader` |
| `splitter.py` | header-aware + token-based chunking | `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter` |
| `indexer.py` | embed and store the chunks | `OpenAIEmbeddings`/fake, `Chroma` |
| `run_ingest.py` | the CLI: loader → splitter → indexer | — |

## Run it

```bash
make ingest
# or:
PYTHONPATH=. python -m data_pipeline.run_ingest
```

Every markdown file is loaded, split, embedded, and written to ChromaDB at
`./.chroma`. Safe to re-run — chunks upsert by a stable `chunk_id`.

## How splitting works (the important part)

Two LangChain splitters, the markdown approach from the course:

1. **`MarkdownHeaderTextSplitter`** splits on the heading structure (`#`, `##`,
   `###`). A chunk rarely mixes two unrelated sections, and the headings are
   captured into metadata.
2. **`RecursiveCharacterTextSplitter.from_tiktoken_encoder`** further splits any
   section longer than the token budget (`chunk_tokens`, default 512), with
   overlap (`chunk_overlap_tokens`, default 64) so a sentence cut at a boundary
   survives into the next chunk. Using the tiktoken encoder means the budget is
   counted in real GPT tokens.

The captured heading is prepended back onto each chunk so it stays
self-describing for the embedding model and for citations.

## Adding your own documents

Drop more `.md` files into a category folder under `documents/` (or add a new
folder) and re-run `make ingest`. Nothing else changes.

## Embeddings & the keyless path

`indexer.py` uses `common/embedding.py`. With `OPENAI_API_KEY` set it uses
`OpenAIEmbeddings` (`text-embedding-3-small`); with no key it uses LangChain's
`DeterministicFakeEmbedding` so ingestion still runs for a demo or tests (lower
quality, clearly labelled).
