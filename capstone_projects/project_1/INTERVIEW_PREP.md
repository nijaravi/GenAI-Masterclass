# Interview Prep — RAG Knowledge Assistant

How to use this: don't memorise answers. Build the project, run it, break it,
re-run the eval. The goal is that you can answer any of these because you
*understand* the system — and so you can survive the third follow-up, which is
where rehearsed answers fall apart.

**Framing this honestly in the interview.** Present it as: *"I built a
production-shaped RAG system to get hands-on with the full stack — hybrid
retrieval, reranking, evaluation, deployment. Let me walk you through the
decisions."* That is a strong, defensible position. A well-built personal
project you fully understand beats a vague secondhand "prod" story every time,
and you never risk getting caught overstating.

---

## A. Opening "walk me through it"

> *Give me the 60-second overview.*

It's a RAG service over an internal knowledge base. Ingestion loads documents,
chunks them with heading-aware + token-window splitting, embeds them, and indexes
them in a vector store — Chroma for dev, pgvector for prod, behind one interface.
At query time I embed the question, run **dense vector search and BM25 lexical
search in parallel**, fuse the two with **Reciprocal Rank Fusion**, then **rerank
the fused candidates with a cross-encoder** and pass the top 5 to the LLM with a
grounding prompt that forces citations. There's a **semantic cache** in front,
**model routing** between GPT-4o and 4o-mini/Groq for cost, and an **evaluation
harness** measuring retrieval hit-rate/MRR, answer faithfulness, and latency.
It's a containerised FastAPI service.

---

## B. Chunking

> *How did you choose your chunk size?*

512 tokens with 64 overlap, measured in tiktoken so it matches the model's
tokenizer and the context budget. The reasoning is a trade-off: too large and
the embedding blurs across multiple topics and you burn context; too small and
you cut the context a fact needs to be understood. I split on headings first so
chunks follow the document's natural structure, then token-window only the
oversized sections. I'd tune it per corpus with the eval harness — code and
tables want different settings than prose.

> *Why the overlap?*

So a sentence or fact sitting on a chunk boundary still appears whole in a
neighbouring chunk; without overlap you can sever a fact and neither chunk
retrieves well for it.

> *(Hard follow-up) Your chunk straddles two facts and retrieval returns the
> wrong half. What do you do?*

That's a precision-vs-recall tuning problem. Options: reduce chunk size, increase
overlap, or move to smaller chunks plus a "parent-document" strategy where you
retrieve on small chunks but feed the larger parent section to the LLM. I'd
decide based on what the eval shows — is it a retrieval miss (Hit@k drops) or a
generation miss (faithfulness drops)?

---

## C. Embeddings & vector store

> *Why text-embedding-3-small and not large?*

Cost/quality. Small is cheap, fast, and strong enough for this corpus; I'd only
move to large if eval showed retrieval quality was the bottleneck. I kept a local
MiniLM option behind the same interface for offline/cost-sensitive use.

> *ChromaDB or pgvector — when would you pick each?*

Chroma when I want zero ops and a single node — great for dev and small/medium
corpora. pgvector when the org already runs Postgres: vectors sit next to
relational data, I reuse existing backup/HA/access control, and I can filter on
metadata in SQL. I used an HNSW index on the pgvector side for approximate NN at
scale. Both sit behind one `VectorStore` interface so it's a config flag.

> *(Hard) What's HNSW and what does it cost you?*

Hierarchical Navigable Small World — a graph index for approximate nearest
neighbour. It's fast and scales well, but it's *approximate*: you trade a little
recall for a lot of speed, and it has build-time/memory cost and tuning knobs
(`m`, `ef_construction`, `ef_search`). For a small corpus an exact flat scan is
fine; HNSW earns its keep at scale.

---

## D. Hybrid search & RRF — *expect the deepest probing here*

> *Why hybrid? Why not just embeddings?*

Dense embeddings are semantic but weak on exact tokens the model never trained
on — error codes, policy numbers, acronyms, product SKUs. In my mock data,
"GP-1107" is exactly that case: BM25 nails it, pure dense search can miss it.
Hybrid gets both semantic recall and lexical precision.

> *How do you combine the two result sets?*

Reciprocal Rank Fusion. I deliberately don't add the scores, because cosine
similarity is bounded `[0,1]` and BM25 is unbounded — combining raw scores is
fragile and needs constant re-normalisation. RRF fuses on rank instead:
`1/(k + rank)` summed across retrievers, k=60. A doc ranked well by either
retriever rises; ranked well by both, it wins. It's simple and robust.

> *(Hard) Why k=60? What does k actually do?*

k damps the contribution of deep ranks — it controls how quickly the `1/(k+rank)`
weight flattens out. 60 is the value from the original RRF paper and a strong
default; smaller k makes top ranks dominate more aggressively. It's tunable, but
RRF is famously insensitive to it, which is part of its appeal. I'd only sweep it
if eval suggested the fusion was the weak link.

> *(Hard) Walk me through fusing [A,B] and [B,C].*

B appears in both lists. Say it's rank 2 in the first and rank 1 in the second:
its RRF score is `1/(60+2) + 1/(60+1)`. A and C each appear once, so each gets a
single `1/(60+rank)` term. B's two contributions put it on top — that's RRF
rewarding cross-retriever agreement. (This is literally `test_rrf_rewards_agreement`
in the repo.)

---

## E. Reranking

> *You already retrieved and fused — why rerank?*

Because retrieval scores the query and each chunk *independently*; it's fast but
never looks at the pair together. A cross-encoder takes `(query, chunk)` jointly
and scores true relevance — much better at telling "contains the keyword" from
"answers the question". I run it only on the ~12 fused candidates, not the whole
corpus: retrieve wide, rerank narrow. It's the biggest precision win in the
pipeline.

> *(Hard) Why not cross-encode everything then? Why retrieve at all?*

Cost. A cross-encoder is a full transformer forward pass per (query, chunk) pair
— running it over the whole corpus per query is computationally hopeless. The
bi-encoder retrieval stage exists to cheaply narrow thousands of chunks to a
dozen good candidates, then the expensive model does the precise ordering.

---

## F. Generation, grounding & hallucination

> *How do you stop it hallucinating?*

Layered. The system prompt restricts the model to the provided context and tells
it to say "I don't have that information" when the context is silent rather than
guess. I number the context passages and require citations, so an answer is
traceable to a source. Low temperature (0.1). And I *measure* it — the eval
harness scores faithfulness, so hallucination is a number I can watch, not a
vibe.

> *(Hard) The model still makes something up that isn't in the context. How do
> you catch it in production?*

Offline faithfulness eval catches regressions pre-release. Online, I'd add a
post-generation check — an LLM or NLI model verifying each answer claim against
the retrieved context, and either flag or suppress unsupported answers. Plus
user feedback signals (thumbs down, escalations) feeding back into the eval set.

> *Why model routing?*

Cost and latency. Most internal questions are simple lookups that 4o-mini or
Llama-8b on Groq handle fine; I reserve 4o for long or multi-part questions. The
router here is a simple heuristic on query length/complexity — in a fuller build
I'd train a small classifier. Both providers share an OpenAI-compatible API so
one client handles both.

---

## G. Semantic cache

> *How is a semantic cache different from a normal cache?*

A normal cache keys on the exact string, so "leave policy?" and "how many
vacation days?" miss each other despite identical intent. The semantic cache
embeds the query and serves a stored answer when cosine similarity to a prior
query clears a threshold.

> *(Hard) What threshold, and what's the failure mode?*

0.95, deliberately strict. The failure mode of a loose threshold is the worst
kind: a confidently wrong cached answer to a subtly different question. I'd
rather miss the cache than serve a wrong hit, so I set it high and would tune it
against real query logs. In production I'd back it with Redis so it's shared
across replicas, not per-process.

---

## H. Evaluation — *the question that separates real engineers*

> *How do you know any of this works?*

The eval harness. Retrieval layer: Hit@k and MRR against a gold set of
question→source mappings — that isolates whether the right chunk even arrives.
Answer layer: LLM-as-judge scoring relevance and faithfulness, plus p50/p95
latency from the pipeline's own timings. When I added the reranker I re-ran and
compared — that's how I'd defend "reranking improved relevance" with a number
instead of an opinion.

> *(Hard) LLM-as-judge — isn't that circular / unreliable?*

It's correlated with human judgement, not identical to it, so I treat it as a
fast proxy, not ground truth. I'd validate the judge against a human-labelled
sample, use a strong model as judge, and keep a human-reviewed gold set for the
metrics that matter. And I'd pair offline eval with online signals — offline
metrics can look great while users are unhappy.

> *What's your single most important metric?*

For this system, faithfulness — a wrong answer with a confident citation is
worse than "I don't know", because people act on internal-policy answers. I'd
watch faithfulness and the "no answer" rate together.

---

## I. Production, scale & failure

> *This gets 100x the documents. What breaks first?*

The in-memory BM25 index — I'd push sparse retrieval into the engine
(OpenSearch, or Postgres full-text) rather than rebuild it in process. Then
ingestion throughput (needs to be incremental, not full re-index) and the HNSW
index parameters. The fusion and rerank logic don't change.

> *A downstream dependency (the LLM API) is down. What happens?*

Retrieval still works, so I can degrade to returning the top reranked passages
with a "couldn't generate a summary" notice rather than failing the request. The
reranker already degrades to pass-through if its model won't load. The global
exception handler ensures no stack trace leaks and the service returns a clean
error.

> *(Hard, security) A user asks about a document they shouldn't see. What stops
> the LLM leaking it?*

This is the big one for enterprise RAG and I'm honest that the learning project
doesn't implement it: retrieval must filter by the caller's permissions *before*
chunks reach the LLM, because a RAG system that ignores ACLs is an exfiltration
channel — the model will happily summarise a restricted doc it was handed. In
production I'd enforce document-level access control at the retrieval query
(metadata filter on the user's entitlements) and never rely on the prompt to
withhold content.

---

## J. The honesty questions (be ready, answer plainly)

> *Is this something you shipped in production?*

No — it's a project I built to learn production RAG end to end. The company and
data are synthetic. But I made the engineering decisions, wrote the code, and can
walk you through any part of it and the trade-offs behind it.

> *What was the hardest part / what did you learn?*

Pick something true from actually building it — e.g. "RRF was a revelation:
I started by trying to normalise and add dense+BM25 scores and it was brittle;
switching to rank fusion made the hybrid step both simpler and better." Or "the
eval harness changed how I worked — before it I was guessing whether changes
helped; after it I could prove it." Specific, real, and it shows judgement.

> *What would you do differently / what's missing?*

Section 9 of ARCHITECTURE.md is your honest answer: auth, rate limiting,
Redis-backed cache, async retrieval, incremental ingestion, document-level
access control, full metrics/tracing, eval as a CI gate. Knowing what you
*didn't* build, and why, reads as senior. Pretending it's complete reads as
junior.

---

## K. 30-second glossary (don't fumble the basics)

- **RAG**: retrieve relevant context, then generate an answer grounded in it —
  so the model uses your data, not just its training.
- **Embedding**: a vector representing meaning; similar texts sit close in vector
  space.
- **Dense vs sparse retrieval**: dense = embedding similarity (semantic); sparse
  = BM25 keyword overlap (lexical).
- **BM25**: a classic lexical ranking function — term frequency × inverse
  document frequency, length-normalised.
- **RRF**: combine ranked lists by summing `1/(k+rank)`; fuses on rank, not score.
- **Cross-encoder vs bi-encoder**: bi-encoder embeds query and doc separately
  (fast, for retrieval); cross-encoder reads them together (accurate, for
  reranking).
- **HNSW**: graph-based approximate nearest-neighbour index.
- **MRR / Hit@k**: retrieval quality metrics — rank of first relevant result /
  whether a relevant result is in the top-k.
- **Faithfulness**: degree to which an answer's claims are supported by retrieved
  context (the inverse of hallucination).
