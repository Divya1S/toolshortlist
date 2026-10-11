"""Metric tests on cases small enough to work out by hand, plus a cross-check against trec_eval."""

import math
import random

import pytest

from toolshortlist.metrics import (
    bootstrap_ci,
    completeness_at_k,
    ndcg_at_k,
    recall_at_k,
    score_query,
)


def test_two_gold_tools_at_ranks_2_and_4():
    ranked = ["a", "b", "c", "d", "e"]
    gold = {"b": 1, "d": 1}
    assert recall_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 2) == 0.5
    assert recall_at_k(ranked, gold, 5) == 1.0
    assert recall_at_k(ranked, gold, 10) == 1.0  # a list shorter than k is fine
    # DCG = 1/log2(3) + 1/log2(5) = 0.63093 + 0.43068 = 1.06161
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.63093 = 1.63093
    assert ndcg_at_k(ranked, gold, 10) == pytest.approx(0.65092, abs=1e-5)
    assert completeness_at_k(ranked, gold, 10) == 1.0
    assert completeness_at_k(ranked, gold, 2) == 0.0


def test_one_of_three_gold_tools_found():
    ranked = ["x", "a", "y", "z"]
    gold = {"a": 1, "b": 1, "c": 1}
    assert recall_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 5) == pytest.approx(1 / 3)
    # DCG = 1/log2(3) = 0.63093; IDCG = 1 + 0.63093 + 0.5 = 2.13093
    assert ndcg_at_k(ranked, gold, 10) == pytest.approx(0.29608, abs=1e-5)
    assert completeness_at_k(ranked, gold, 10) == 0.0


def test_graded_relevance():
    ranked = ["a", "b"]
    gold = {"a": 1, "b": 3}
    # DCG = 1/1 + 3/log2(3) = 2.89279; IDCG = 3/1 + 1/log2(3) = 3.63093
    assert ndcg_at_k(ranked, gold, 10) == pytest.approx(0.79671, abs=1e-5)


def test_perfect_and_empty_rankings():
    assert score_query(["a"], {"a": 1}) == {
        "recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0, "ndcg@10": 1.0, "completeness@10": 1.0,
    }
    assert set(score_query([], {"a": 1}).values()) == {0.0}


def test_recall_at_1_with_two_gold_tools_is_at_most_half():
    assert recall_at_k(["a", "b"], {"a": 1, "b": 1}, 1) == 0.5


def test_completeness_needs_every_gold_tool():
    gold = {f"t{i}": 1 for i in range(11)}
    ranked = [f"t{i}" for i in range(11)]
    assert completeness_at_k(ranked, gold, 10) == 0.0  # 11 gold tools cannot fit in 10
    assert completeness_at_k(ranked, gold, 11) == 1.0


def test_only_the_top_k_counts():
    ranked = ["x"] * 10 + ["a"]
    assert recall_at_k(ranked, {"a": 1}, 10) == 0.0
    assert ndcg_at_k(ranked, {"a": 1}, 10) == 0.0


def test_bootstrap_ci():
    assert bootstrap_ci([0.5] * 20) == (0.5, 0.5)
    low, high = bootstrap_ci([0.0, 1.0] * 50)
    assert 0.35 < low < 0.5 < high < 0.65
    assert bootstrap_ci([0.0, 1.0] * 50) == (low, high)  # same seed, same interval


def test_agrees_with_trec_eval():
    """ToolRet computes its metrics with pytrec_eval; ours must match it."""
    pytrec_eval = pytest.importorskip("pytrec_eval")
    rng = random.Random(0)
    docs = [f"d{i}" for i in range(50)]
    qrels, runs, ours = {}, {}, {}
    for q in range(30):
        gold = {d: rng.choice([1, 1, 2]) for d in rng.sample(docs, rng.randint(1, 6))}
        ranked = rng.sample(docs, 20)
        qrels[f"q{q}"] = gold
        # Distinct, decreasing scores so trec_eval keeps our order.
        runs[f"q{q}"] = {d: float(len(ranked) - i) for i, d in enumerate(ranked)}
        ours[f"q{q}"] = (ranked, gold)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.1,5,10"})
    theirs = evaluator.evaluate(runs)
    for q, (ranked, gold) in ours.items():
        assert ndcg_at_k(ranked, gold, 10) == pytest.approx(theirs[q]["ndcg_cut_10"], abs=1e-4)
        for k in (1, 5, 10):
            assert recall_at_k(ranked, gold, k) == pytest.approx(theirs[q][f"recall_{k}"], abs=1e-4)
        assert completeness_at_k(ranked, gold, 10) == float(math.isclose(theirs[q]["recall_10"], 1.0))
