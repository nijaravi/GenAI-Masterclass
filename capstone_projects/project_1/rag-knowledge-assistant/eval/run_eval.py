"""Evaluation harness (resume: relevance / hallucination / latency checks).

Two layers, because they answer different questions:

  RETRIEVAL metrics (no LLM, cheap, deterministic) — is the right chunk even
  reaching the model?
    * Hit@k          : did a known-relevant chunk appear in the top-k?
    * MRR            : how high did the first relevant chunk rank?

  ANSWER metrics (LLM-as-judge) — given good context, is the answer good?
    * Relevance      : does the answer address the question? (1-5)
    * Faithfulness   : is every claim supported by the retrieved context, i.e.
                       NOT hallucinated? (1-5)
    * Latency        : p50 / p95 end-to-end, from the pipeline's own timings.

The gold set lives in data/eval/eval_set.jsonl. This is the artefact you point
at in an interview when asked "how did you know reranking helped?" — you flip
USE_RERANKER and re-run.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rag.config import settings
from rag.generation.llm import LLMClient
from rag.logging_config import configure_logging, get_logger
from rag.pipeline import RAGPipeline
from rag.api.dependencies import _build_bm25

logger = get_logger("eval")

JUDGE_PROMPT = (
    "You are a strict evaluator. Given a QUESTION, the CONTEXT passages the "
    "system retrieved, and the ANSWER it produced, score two dimensions from 1-5.\n"
    "relevance: does the ANSWER address the QUESTION?\n"
    "faithfulness: is every factual claim in the ANSWER supported by the CONTEXT "
    "(5 = fully grounded, 1 = clearly hallucinated)?\n"
    'Respond ONLY with JSON: {"relevance": int, "faithfulness": int, "reason": str}'
)


def load_eval_set(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def retrieval_metrics(pipeline: RAGPipeline, item: dict, k: int) -> dict:
    """Compute Hit@k and reciprocal rank against expected source files."""
    q_emb = pipeline.embedder.embed_query(item["question"])
    dense = pipeline.store.search(q_emb, k)
    sparse = pipeline.bm25.search(item["question"], k) if pipeline.bm25 else []
    from rag.retrieval.hybrid import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion([dense, sparse], top_k=k)
    expected = set(item.get("relevant_sources", []))
    hit, rr = 0, 0.0
    for rank, sc in enumerate(fused, start=1):
        if sc.chunk.source in expected:
            hit = 1
            rr = 1.0 / rank
            break
    return {"hit": hit, "rr": rr}


def judge_answer(judge: LLMClient, item: dict, answer: str, context: str) -> dict:
    user = (
        f"QUESTION:\n{item['question']}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"
    )
    out = judge.generate(JUDGE_PROMPT, user, item["question"]).text
    out = out.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"relevance": 0, "faithfulness": 0, "reason": "unparseable judge output"}


def run(eval_path: str, judge_enabled: bool) -> dict:
    configure_logging(settings.log_level, json_logs=False)
    pipeline = RAGPipeline(bm25=_build_bm25())
    judge = LLMClient() if judge_enabled else None
    items = load_eval_set(eval_path)

    hits, rrs, rels, faiths, latencies = [], [], [], [], []
    for item in items:
        rm = retrieval_metrics(pipeline, item, settings.fused_top_k)
        hits.append(rm["hit"])
        rrs.append(rm["rr"])

        resp = pipeline.query(item["question"], use_cache=False)
        latencies.append(resp.timings_ms.get("total_ms", 0.0))

        if judge is not None:
            ctx = "\n\n".join(c.snippet for c in resp.citations)
            j = judge_answer(judge, item, resp.answer, ctx)
            rels.append(j.get("relevance", 0))
            faiths.append(j.get("faithfulness", 0))
            logger.info(
                "judged",
                extra={"extra": {"q": item["question"][:50],
                                 "rel": j.get("relevance"), "faith": j.get("faithfulness")}},
            )

    def p(vals, q):
        if not vals:
            return 0.0
        s = sorted(vals)
        return round(s[min(len(s) - 1, int(q * len(s)))], 2)

    report = {
        "n": len(items),
        "retrieval": {
            "hit_rate": round(sum(hits) / len(hits), 3) if hits else 0,
            "mrr": round(sum(rrs) / len(rrs), 3) if rrs else 0,
        },
        "latency_ms": {
            "p50": p(latencies, 0.5),
            "p95": p(latencies, 0.95),
            "mean": round(statistics.mean(latencies), 2) if latencies else 0,
        },
    }
    if judge_enabled and rels:
        report["answer"] = {
            "relevance_mean": round(statistics.mean(rels), 2),
            "faithfulness_mean": round(statistics.mean(faiths), 2),
        }
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", default="data/eval/eval_set.jsonl")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the LLM-as-judge answer scoring (retrieval+latency only)")
    args = ap.parse_args()
    report = run(args.eval_path, judge_enabled=not args.no_judge)
    print(json.dumps(report, indent=2))
