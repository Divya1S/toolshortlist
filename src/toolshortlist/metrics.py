"""Retrieval metrics for one query, plus a bootstrap confidence interval.

Every function takes `ranked`, the retrieved keys in rank order (best first), and
`gold`, a dict from relevant key to its relevance grade (a positive integer).
The definitions follow trec_eval, which the ToolRet benchmark uses through
pytrec_eval; tests/test_metrics.py checks them against it.
"""

import math
import random


def recall_at_k(ranked: list[str], gold: dict[str, int], k: int) -> float:
    """Share of the gold tools that appear in the top k."""
    return sum(key in gold for key in ranked[:k]) / len(gold)


def completeness_at_k(ranked: list[str], gold: dict[str, int], k: int) -> float:
    """1 if every gold tool is in the top k, otherwise 0 (ToolRet's C@K)."""
    return float(recall_at_k(ranked, gold, k) == 1.0)


def ndcg_at_k(ranked: list[str], gold: dict[str, int], k: int) -> float:
    """Discounted gain of the top k, divided by the gain of an ideal ranking.

    The gain of a tool is its relevance grade and the discount at rank r
    (counting from 1) is log2(r + 1).
    """
    dcg = sum(gold.get(key, 0) / math.log2(rank + 2) for rank, key in enumerate(ranked[:k]))
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return dcg / idcg


# Name -> (function, k). These are the metrics PROJECT.md section 7 asks for.
METRICS = {
    "recall@1": (recall_at_k, 1),
    "recall@5": (recall_at_k, 5),
    "recall@10": (recall_at_k, 10),
    "ndcg@10": (ndcg_at_k, 10),
    "completeness@10": (completeness_at_k, 10),
}


def score_query(ranked: list[str], gold: dict[str, int]) -> dict[str, float]:
    """All metrics for one query."""
    return {name: fn(ranked, gold, k) for name, (fn, k) in METRICS.items()}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bootstrap_ci(values: list[float], n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean of per-query scores."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean(rng.choices(values, k=n)) for _ in range(n_resamples))
    return means[int(0.025 * n_resamples)], means[int(0.975 * n_resamples) - 1]
