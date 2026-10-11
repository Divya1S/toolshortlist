"""Tests for the training pieces that can be checked by hand: the loss, the split, negative picking."""

import math
import tomllib

import pytest
import torch

from toolshortlist.train import (
    MINE_RANKS, NEGATIVES_PER_QUERY, Row, infonce_loss, pick_negatives, split, write_eval_config,
)


def test_infonce_loss_by_hand():
    # Two queries, two tools, each query's own positive is the matching unit vector.
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    own = torch.eye(2, dtype=torch.bool)
    # scores = identity; each row: -log(e^1 / (e^1 + e^0)) = log(1 + e^-1) = 0.31326
    assert infonce_loss(q, d, own, temperature=1.0).item() == pytest.approx(math.log(1 + math.exp(-1)), abs=1e-5)
    # temperature 0.5 doubles the scores: log(1 + e^-2) = 0.12693
    assert infonce_loss(q, d, own, temperature=0.5).item() == pytest.approx(math.log(1 + math.exp(-2)), abs=1e-5)


def test_infonce_masks_other_positives_of_the_same_query():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    # Tool 1 is also a positive of query 0, so it must not count as query 0's negative:
    # row 0 then has a single candidate and loss 0; row 1 keeps log(1 + e^-1).
    mask = torch.tensor([[True, True], [False, True]])
    expected = (0.0 + math.log(1 + math.exp(-1))) / 2
    assert infonce_loss(q, d, mask, temperature=1.0).item() == pytest.approx(expected, abs=1e-5)


def test_pick_negatives():
    candidates = list(range(40))
    keys = [(f"tool{t}", f"description {t}") for t in range(40)]
    keys[12] = keys[7]  # tool 12 duplicates positive 7 under another id
    picked = pick_negatives(candidates, positives=[7, 9], keys=keys)
    assert picked == [4, 5, 6, 8, 10, 11, 13, 14]
    assert len(picked) == NEGATIVES_PER_QUERY
    assert all(t >= MINE_RANKS[0] for t in picked)


def test_split_holds_out_tools_not_queries():
    tools = [f"tool {t}" for t in range(400)]
    rows = [Row(f"r{i}", f"query {i}", [i % 400, (i * 7) % 400] if i % 3 == 0 else [i % 400]) for i in range(2000)]
    train, validation, corpus, dropped = split(rows, tools, seed=1)
    held = {p for row in validation for p in row.positives}
    assert held, "the split held out some tools"
    assert not any(p in held for row in train for p in row.positives), "a held-out tool is a training positive"
    assert held <= set(corpus), "every validation gold tool is in the validation corpus"
    assert len(train) + len(validation) + dropped <= len(rows)
    assert dropped > 0  # rows mixing held-out and other gold tools belong to neither side
    # The split is a function of the seed only.
    assert split(rows, tools, seed=1)[1] == validation


def test_write_eval_config(tmp_path):
    cfg = {"name": "x", "description": "a test run",
           "model": {"base": "BAAI/bge-small-en-v1.5", "pooling": "cls", "eval_max_length": 512,
                     "query_template": "Represent this: {query}"},
           "data": {"text_mode": "name", "hard_negatives": False}}
    write_eval_config(cfg, "train-x-s3", 3, tmp_path / "model", tmp_path)
    parsed = tomllib.loads((tmp_path / "eval.toml").read_text())
    assert parsed["name"] == "train-x-s3"
    assert parsed["train"] == {"config": "x", "seed": 3, "base": "BAAI/bge-small-en-v1.5",
                               "text_mode": "name", "hard_negatives": False}
    r = parsed["retriever"]
    assert r["type"] == "dense" and r["text_mode"] == "name" and r["max_length"] == 512
    assert r["query_template"] == "Represent this: {query}"
    assert r["query_template_inst"] == "Represent this: {instruction} {query}"
    assert r["model"].endswith("/model")
