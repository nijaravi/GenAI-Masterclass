# Interview Prep — Meridian RAG Platform

The honest way to use this: **run it, then break it.** `make ingest && make
backend`, open the Streamlit app, ask questions as different users, watch the
admin dashboard, try a prompt injection, run `make eval`. Once you've operated
each layer, the questions below are easy because you're describing something you
actually built.

**Framing line:** *"I built a layered RAG platform on LangChain to understand how
a production system fits together — the data pipeline, the retrieval stack, an
LCEL chain for generation, guardrails, cost tracking, and an async eval loop.
LangChain handles the RAG plumbing; the gateway, guardrails, and monitoring are
custom. Let me walk through a request."*

---

## A. The 60-second walkthrough

A user asks a question in Streamlit. FastAPI authenticates the API key to a user,
rate-limits them, and validates the request. Input guardrails redact PII and block
injection. The orchestration layer composes a LangChain retriever — dense vectors
from Chroma plus a BM25 retriever, fused by an EnsembleRetriever and reranked by a
cross-encoder — formats the top passages into a prompt, routes to a model
(cheap/smart), and runs an LCEL chain `prompt | model | StrOutputParser`. Output
guardrails check toxicity, PII leaks, and groundedness. Monitoring writes one
trace row — latency, tokens (captured by a LangChain callback), cost — attributed
to that user. Separately, an eval pipeline samples traces and uses an
LLM-as-judge (an LCEL chain with a Pydantic output parser) to score relevance and
faithfulness, which show on the admin dashboard.

---

## B. Why LangChain — and why not everywhere?

> *Why use LangChain here?*

Because it gives tested, swappable components for the parts of RAG that are
fiddly to hand-roll: loaders, the markdown + token splitters, embeddings, the
Chroma store, the dense/sparse/ensemble/rerank retriever stack, prompt templates,
the chain, and structured output parsing. Using them is faster, less buggy, and
is how the course says to build for real.

> *So why is the gateway / guardrails / monitoring custom?*

Because those aren't things a RAG framework has an opinion on — auth, per-user
rate limiting and cost, the trace store, anomaly alerts. Forcing LangChain in
there would be using it for its own sake. The clean line is: *framework for the
RAG plumbing, custom code for the production platform around it.*

---

## C. Retrieval (expect depth here)

> *Walk me through retrieval.*

It's hybrid. Dense retrieval is the Chroma store as a retriever — good at meaning.
BM25Retriever is lexical — good at exact tokens dense misses, like a policy number
"HR-LV-2024-03". I combine them with an `EnsembleRetriever`, which fuses the two
ranked lists with Reciprocal Rank Fusion, then wrap a cross-encoder reranker in a
`ContextualCompressionRetriever` to re-score the top candidates and keep the best
five. Retrieve wide, rerank narrow — the whole thing is still one LangChain
retriever I call `.invoke()` on.

> *What does the EnsembleRetriever actually do under the hood?*

RRF — Reciprocal Rank Fusion. It scores each document by summing `1 / (k + rank)`
across the retrievers it appears in, so it fuses on *rank*, not raw score. That
sidesteps the problem that cosine similarity and BM25 scores are on different
scales and can't just be added. A doc ranked highly by either retriever rises;
one ranked highly by both wins. I can speak to the formula because it's worth
understanding even though the library implements it.

> *Why a cross-encoder for reranking and not just the retriever scores?*

Dense and sparse score the query and a chunk separately. A cross-encoder sees the
(question, chunk) pair together and scores how well it actually answers — much
better at separating "mentions the keyword" from "answers it". It's slow, so it
only runs on the ~dozen fused candidates, not the whole corpus.

> *How did you chunk, and why that way?*

`MarkdownHeaderTextSplitter` first, so chunks follow the document's heading
structure and don't blend unrelated sections, then
`RecursiveCharacterTextSplitter.from_tiktoken_encoder` to enforce a token budget
with overlap. Token-based so the budget matches the model's real token count, and
overlap so a fact split across a boundary still appears whole in a neighbour.

---

## D. Generation, chains, routing, caching

> *How is generation wired?*

An LCEL chain: `ChatPromptTemplate | model | StrOutputParser()`. The prompt's
system message carries the anti-hallucination rules — answer only from the
numbered context, say "I don't have that information" otherwise, cite the passage
numbers. The model is chosen per request by a router.

> *Model routing?*

A length/complexity heuristic sends short, simple questions to a cheap model
(gpt-4o-mini or Groq Llama) and long/multi-part ones to gpt-4o. Most questions are
simple lookups, so this cuts cost a lot. In a fuller build I'd replace the
heuristic with a small classifier and measure the quality/cost trade-off on the
eval set.

> *Caching?*

LangChain's global LLM cache (`set_llm_cache(InMemoryCache())`), so identical
calls are served from memory. I detect a hit by checking whether the cost callback
saw any LLM run. For semantic caching (matching paraphrases, not exact strings)
LangChain has integrations like a Redis semantic cache; I used the in-memory exact
cache to stay dependency-light, and I'd swap in the semantic one for production.

---

## E. Safety / guardrails

> *You're putting an LLM in front of company data — what stops it going wrong?*

Guardrails on both ends, plus the prompt rules. Input: block prompt-injection
phrasing, redact PII before anything is sent or logged. Output: block toxicity and
any answer containing PII, and flag low groundedness (how much of the answer's
wording actually appears in the retrieved context) as possible hallucination.

> *(Honest follow-up) How good are those checks really?*

They're lightweight heuristics — regex, patterns, a wordlist, word-overlap. They
demonstrate *where* each check belongs. In production I'd swap each for a real
component (Presidio for PII, a trained injection/jailbreak classifier, a toxicity
model, an NLI or LLM-judge for groundedness) behind the same function signatures.
I didn't use LangChain here because it has no first-class guardrail without extra
deps — and I'd rather say that than pretend a regex is a security boundary.

> *Why is the expensive groundedness check not inline?*

Cost and latency. Inline I use the cheap overlap heuristic as a fast flag; the
real LLM-as-judge runs *asynchronously* over a sample of traffic in the eval
pipeline. Cheap guard on the hot path, expensive judge offline.

---

## F. Users, cost, monitoring, eval

> *How do you track cost per user?*

Every request is authenticated to a user, and a LangChain callback captures the
token usage from the model's response on `on_llm_end`. I price those tokens with a
per-model table and store the cost on the request's trace. The admin cost endpoint
sums per user. So cost falls out of identity at the gateway plus token counts from
the callback.

> *What's the eval pipeline?*

It samples unjudged traces and runs an LLM-as-judge built as an LCEL chain with a
`PydanticOutputParser`, so the judge's reply is parsed straight into a typed
`{relevance, faithfulness, reason}` instead of hand-parsing JSON. Scores are
written back onto the trace and show up in the admin metrics and the anomaly
checker. It runs on a schedule, off the request path, so judging never slows a
user.

> *How would you know quality is drifting?*

The anomaly checker fires on thresholds — p95 latency, block rate spiking, average
faithfulness dropping. In production those become LangSmith / Datadog monitors,
and I'd add a held-out gold set with Hit@k and MRR to measure retrieval directly.

---

## G. Honesty questions

> *Did this run in production?*

No — a project I built to learn how a production RAG platform is structured, on
the LangChain stack the course taught. The company and docs are synthetic. I wrote
every layer and chose every component, which is what I'd want to be judged on.

> *Weakest part?*

The guardrails are heuristics, and retrieval has no per-user access control — any
user can retrieve any document. For a real internal tool, document ACLs feeding
the retrieval filter would be the first thing I'd add.

> *What would you build next?*

Document-level access control in retrieval, real guardrail models, a Redis
semantic cache + shared rate limiter, streaming responses, enforced spend caps,
and proper retrieval eval (Hit@k/MRR on a gold set) on a schedule — plus LangSmith
tracing instead of the SQLite trace table.

---

## H. 30-second glossary

- **RAG** — retrieve relevant docs, put them in the prompt, generate a grounded
  answer.
- **LCEL** — LangChain Expression Language; composing steps with `|`
  (`prompt | model | parser`).
- **Splitters** — `MarkdownHeaderTextSplitter` (split on headings),
  `RecursiveCharacterTextSplitter` (size/token budget with overlap).
- **Dense vs sparse** — embedding/semantic search vs BM25 keyword search.
- **EnsembleRetriever / RRF** — fuses ranked lists by rank, not score.
- **ContextualCompressionRetriever** — wraps a reranker so reranking is part of
  the retriever.
- **Output parser** — `PydanticOutputParser` turns the model's reply into a typed
  object.
- **Callback** — a LangChain hook (`BaseCallbackHandler`) used here to capture
  token usage for cost.
- **LLM-as-judge** — an LLM scoring answer relevance/faithfulness, run async on
  sampled traffic.
- **Trace** — the one stored record per request powering monitoring, cost, and
  eval.
