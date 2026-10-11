"""Tests for rank fusion, query formatting, BM25 and the evaluate entry point on a tiny catalog."""

import json

import numpy as np

from toolshortlist.data import Benchmark, Query, load_handwritten
from toolshortlist.evaluate import evaluate, ranked_keys
from toolshortlist.retrievers import BM25Retriever, HybridRetriever, format_queries, rrf_fuse


def test_rrf_fuse_by_hand():
    # One query, two rankings, rrf_k = 60.
    #   doc 0: ranks 1 and 2 -> 1/61 + 1/62 = 0.032522
    #   doc 2: ranks 3 and 1 -> 1/63 + 1/61 = 0.032266
    #   doc 1: rank 2        -> 1/62        = 0.016129
    #   doc 3: rank 3        -> 1/63        = 0.015873
    fused = rrf_fuse([np.array([[0, 1, 2]]), np.array([[2, 0, 3]])], k=4, rrf_k=60)
    assert fused.tolist() == [[0, 2, 1, 3]]
    assert rrf_fuse([np.array([[0, 1, 2]]), np.array([[2, 0, 3]])], k=2).tolist() == [[0, 2]]


def test_rrf_fuse_breaks_ties_by_position():
    # Docs 5 and 3 each hold rank 1 in one list and are absent from the other.
    assert rrf_fuse([np.array([[5]]), np.array([[3]])], k=2).tolist() == [[3, 5]]


def test_format_queries():
    assert format_queries(["q"], None, "P: {query}", "{instruction} {query}") == ["P: q"]
    assert format_queries(["q"], ["i"], "P: {query}", "{instruction} | {query}") == ["i | q"]
    # Braces inside a query are left alone.
    assert format_queries(['{"a": 1}'], None, "{query}", "") == ['{"a": 1}']


def test_ranked_keys_keeps_first_occurrence():
    keys = ["search", "fetch", "search", "read"]
    assert ranked_keys([2, 0, 3, 1], keys) == ["search", "read", "fetch"]


DOCS = [
    "get the current weather forecast for a city",
    "send an email message to a contact",
    "convert an amount between two currencies",
]


def test_bm25_ranks_the_matching_tool_first():
    retriever = BM25Retriever()
    retriever.index(DOCS)
    positions = retriever.search(["weather forecast in Paris", "currencies conversion amount"], k=3)
    assert positions.shape == (2, 3)
    assert positions[:, 0].tolist() == [0, 2]


class FixedRetriever:
    """Returns the same ranking for every query."""

    def __init__(self, ranking):
        self.ranking = ranking

    def index(self, docs):
        pass

    def search(self, queries, k, instructions=None):
        return np.array([self.ranking[:k]] * len(queries))


def test_hybrid_fuses_its_parts():
    hybrid = HybridRetriever([FixedRetriever([0, 1, 2]), FixedRetriever([2, 0, 1])], depth=3)
    hybrid.index(DOCS)
    # doc 0: 1/61 + 1/62, doc 2: 1/63 + 1/61, doc 1: 1/62 + 1/63
    assert hybrid.search(["anything"], k=3).tolist() == [[0, 2, 1]]


def test_evaluate_writes_hand_checkable_results(tmp_path):
    bench = Benchmark(
        name="tiny", tool_ids=["t0", "t1", "t2"], tool_texts=DOCS, tool_keys=["t0", "t1", "t2"],
        queries=[Query("q0", "x", {"t0": 1}, "a"), Query("q1", "y", {"t1": 1}, "a"),
                 Query("q2", "z", {"t2": 1}, "b")],
        splits={"only_b": ["b"]},
    )
    # Every query gets the ranking t0, t1, t2, so recall@1 is 1, 0, 0.
    result = evaluate(FixedRetriever([0, 1, 2]), "fixed", [bench], out_dir=tmp_path)
    assert json.loads((tmp_path / "fixed.json").read_text()) == result
    summary = result["benchmarks"]["tiny"]["settings"]["query"]
    assert summary["all"]["queries"] == 3
    assert summary["all"]["query_mean"]["recall@1"] == 1 / 3  # mean of 1, 0, 0
    assert summary["all"]["group_mean"]["recall@1"] == 0.25  # group a: 0.5, group b: 0 -> 0.25
    assert summary["all"]["query_mean"]["recall@5"] == 1.0
    assert summary["splits"]["only_b"]["queries"] == 1
    assert summary["splits"]["only_b"]["query_mean"]["recall@1"] == 0.0
    assert summary["groups"]["a"] == {
        "queries": 2, "recall@1": 0.5, "recall@5": 1.0, "recall@10": 1.0,
        # nDCG@10: q0 at rank 1 -> 1; q1 at rank 2 -> 1/log2(3) = 0.63093; mean 0.81546
        "ndcg@10": summary["groups"]["a"]["ndcg@10"], "completeness@10": 1.0,
    }
    assert abs(summary["groups"]["a"]["ndcg@10"] - 0.81546) < 1e-5
    assert "instruction" not in result["benchmarks"]["tiny"]["settings"]


def test_load_handwritten(tmp_path):
    assert load_handwritten(tmp_path) is None  # nothing written yet
    (tmp_path / "tools.json").write_text(json.dumps([
        {"server": "files", "name": "read_file", "description": "Read a file.",
         "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
        {"server": "mail", "name": "send", "description": "Send an email."},
    ]))
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"id": "h1", "query": "open notes.txt", "tools": ["files/read_file"]}) + "\n")
    bench = load_handwritten(tmp_path)
    assert bench.tool_ids == ["files/read_file", "mail/send"]
    assert json.loads(bench.tool_texts[0]) == {
        "server": "files", "name": "read_file", "description": "Read a file.",
        "parameters": {"path": {"type": "string"}}}
    assert bench.queries[0].gold == {"files/read_file": 1}
    assert bench.report_ci
