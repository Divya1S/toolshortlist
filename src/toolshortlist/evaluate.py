"""Evaluate a retriever on the test sets and write results/<name>.json.

Run: uv run python -m toolshortlist.evaluate configs/<name>.toml
"""

import argparse
import json
import platform
import tomllib
from collections import defaultdict
from datetime import date
from importlib.metadata import version
from pathlib import Path

from .data import LOADERS, REVISIONS
from .metrics import METRICS, bootstrap_ci, mean, score_query
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, inference_device

DEPTH = 100  # results fetched per query; the metrics use the top 10


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_retriever(spec: dict):
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "bm25":
        return BM25Retriever(**spec)
    if kind == "dense":
        return DenseRetriever(**spec)
    if kind == "hybrid":
        parts = [build_retriever(load_config(path)["retriever"]) for path in spec.pop("parts")]
        return HybridRetriever(parts, **spec)
    raise ValueError(f"unknown retriever type: {kind}")


def ranked_keys(positions, tool_keys):
    """Label keys of the retrieved tools in rank order, each key once."""
    return list(dict.fromkeys(tool_keys[position] for position in positions))


def summarise(scores: list[dict], groups: list[str], with_ci: bool) -> dict:
    """Average per-query scores two ways: over queries, and over groups of queries."""
    by_group = defaultdict(list)
    for score, group in zip(scores, groups):
        by_group[group].append(score)
    summary = {
        "queries": len(scores),
        "groups": len(by_group),
        "query_mean": {m: mean([s[m] for s in scores]) for m in METRICS},
        "group_mean": {m: mean([mean([s[m] for s in rows]) for rows in by_group.values()])
                       for m in METRICS},
    }
    if with_ci:
        summary["ci95"] = {m: bootstrap_ci([s[m] for s in scores]) for m in METRICS}
    return summary


def evaluate(retriever, name: str, benchmarks: list, config: dict | None = None,
             out_dir: str = "results") -> dict:
    """Score `retriever` on each benchmark and write the result to <out_dir>/<name>.json.

    A retriever is anything with index(docs) and search(queries, k, instructions).
    """
    result = {
        "name": name,
        "date": date.today().isoformat(),
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            **{package: version(package) for package in ("torch", "transformers", "bm25s", "numpy")},
            "embedding_device": inference_device()[0],
            "embedding_precision": inference_device()[1],
        },
        "dataset_revisions": REVISIONS,
        "benchmarks": {},
    }
    for bench in benchmarks:
        retriever.index(bench.tool_texts)
        texts = [query.text for query in bench.queries]
        groups = [query.group for query in bench.queries]
        settings = {"query": None}
        if all(query.instruction for query in bench.queries):
            settings["instruction"] = [query.instruction for query in bench.queries]

        entry = {"tools": len(bench.tool_ids), "queries": len(bench.queries),
                 "notes": bench.notes, "settings": {}}
        for setting, instructions in settings.items():
            positions = retriever.search(texts, DEPTH, instructions)
            scores = [score_query(ranked_keys(row, bench.tool_keys), query.gold)
                      for query, row in zip(bench.queries, positions)]
            summary = {"all": summarise(scores, groups, bench.report_ci), "splits": {}, "groups": {}}
            for split, members in bench.splits.items():
                keep = [i for i, group in enumerate(groups) if group in members]
                summary["splits"][split] = summarise(
                    [scores[i] for i in keep], [groups[i] for i in keep], bench.report_ci)
            for group in sorted(set(groups)):
                rows = [score for score, g in zip(scores, groups) if g == group]
                summary["groups"][group] = {
                    "queries": len(rows), **{m: mean([s[m] for s in rows]) for m in METRICS}}
            entry["settings"][setting] = summary
        result["benchmarks"][bench.name] = entry

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to a retriever config, for example configs/bm25.toml")
    parser.add_argument("--benchmarks", default="toolret,livemcpbench,handwritten")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    config = load_config(args.config)
    benchmarks = []
    for name in args.benchmarks.split(","):
        bench = LOADERS[name]()
        if bench is None:
            print(f"skipping {name}: not written yet")
            continue
        if name == "toolret":
            audit_file = Path(args.out_dir) / "data_audit.json"
            if not audit_file.exists():
                raise SystemExit("Run `python -m toolshortlist.audit` first: it decides which "
                                 "ToolRet tasks are seen-source and which are unseen-source.")
            audit = json.loads(audit_file.read_text())
            bench.splits["unseen_source"] = audit["unseen_source_tasks"]
            bench.splits["seen_source"] = audit["seen_source_tasks"]
        benchmarks.append(bench)

    result = evaluate(build_retriever(config["retriever"]), config["name"], benchmarks, config,
                      args.out_dir)
    for bench_name, entry in result["benchmarks"].items():
        for setting, summary in entry["settings"].items():
            line = f"{config['name']} | {bench_name} | {setting} | all: " + ", ".join(
                f"{m} {100 * v:.2f}" for m, v in summary["all"]["query_mean"].items())
            print(line)


if __name__ == "__main__":
    main()
