"""Run the eval pipeline over sampled production traffic.

    python -m eval.run_eval --sample 10

Samples unjudged traces from the store, scores each with the LLM-as-judge chain,
and writes the relevance/faithfulness scores back. Meant to run on a schedule,
off the request path, so judging never slows a user's request. The scores then
appear in the admin metrics and feed the anomaly check.
"""
import argparse

from backend.app.monitoring.store import TraceStore
from common.config import settings
from common.logging_setup import get_logger, setup_logging
from .judge import Judge
from .sampler import sample_traces

logger = get_logger("eval")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=10)
    args = ap.parse_args()

    setup_logging(settings.log_level)
    store = TraceStore()
    judge = Judge()

    traces = sample_traces(store, args.sample)
    if not traces:
        logger.info("no unjudged traces to evaluate (ask some questions first)")
        return

    rel_total, faith_total = 0, 0
    for t in traces:
        result = judge.score(t["question"], t["context_used"], t["answer"])
        rel, faith = result.get("relevance", 0), result.get("faithfulness", 0)
        store.update_scores(t["request_id"], rel, faith)
        rel_total += rel
        faith_total += faith
        logger.info("judged %s: relevance=%d faithfulness=%d",
                    t["request_id"], rel, faith)

    n = len(traces)
    logger.info("done: judged %d | avg relevance=%.2f | avg faithfulness=%.2f",
                n, rel_total / n, faith_total / n)


if __name__ == "__main__":
    main()
