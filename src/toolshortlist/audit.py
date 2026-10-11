"""Overlap audit: how much of ToolRet's evaluation set also appears in its training set.

Writes two files:
  results/data_audit.json         counts, and which tasks are seen-source or unseen-source
  results/train_exclude_ids.json  training rows to remove before any training

Run: uv run python -m toolshortlist.audit
"""

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from .data import REVISIONS, iter_toolret_train, load_toolret
from .retrievers import BM25Retriever

# All thresholds were fixed before the near-duplicate counts were seen.
QUERY_JACCARD = 0.9  # two queries are near-duplicates if their word sets overlap at least this much
DESCRIPTION_RATIO = 0.9  # two descriptions are near-identical at or above this difflib ratio
MIN_DESCRIPTION_CHARS = 40  # an identical description counts on its own only if at least this long
TASK_THRESHOLD = 0.01  # a task is seen-source if at least this share of its queries overlap
TRAIN_SOURCES = ["apigen", "toolace", "toolbench"]  # tasks whose source the training set was built from
CANDIDATES = 20  # training queries compared with each evaluation query

RULE = (
    "A task is seen-source if its source dataset is one the training set was built from, "
    "or at least 1% of its queries are exact or near-duplicates of a training query, "
    "or at least 1% of its queries have a gold tool that matches a training positive. "
    "Every other task is unseen-source."
)


def normalise(text: str) -> str:
    """Lower-case and keep only letters and digits, separated by single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def parse_tool(documentation: str) -> tuple[str, str, str]:
    """(canonical documentation, normalised name, normalised description).

    Name and description are empty when the documentation has no such field.
    """
    try:
        doc = json.loads(documentation)
    except ValueError:
        return documentation, "", ""
    if not isinstance(doc, dict):
        return documentation, "", ""
    return (json.dumps(doc, sort_keys=True, ensure_ascii=False),
            normalise(str(doc.get("name", ""))), normalise(str(doc.get("description", ""))))


class ToolIndex:
    """The training positives, organised so an evaluation tool can be matched against them."""

    def __init__(self):
        self.docs = set()
        self.descriptions_by_name = defaultdict(set)
        self.long_descriptions = set()

    def add(self, tool):
        doc, name, description = tool
        self.docs.add(doc)
        if name:
            self.descriptions_by_name[name].add(description)
        if len(description) >= MIN_DESCRIPTION_CHARS:
            self.long_descriptions.add(description)

    def match(self, tool) -> str | None:
        """How the tool matches a training positive, or None. The strictest criterion wins."""
        doc, name, description = tool
        if doc in self.docs:
            return "identical documentation"
        candidates = self.descriptions_by_name.get(name, ()) if name else ()
        if description and description in candidates:
            return "same name and description"
        if description and any(SequenceMatcher(None, description, other).ratio() >= DESCRIPTION_RATIO
                               for other in candidates if other):
            return "same name, near-identical description"
        if description in self.long_descriptions:
            return "same description under another name"
        return None


def main(out_dir: str = "results"):
    bench = load_toolret()
    corpus = dict(zip(bench.tool_ids, bench.tool_texts))

    # Read the training set once.
    parsed = {}
    def tool(documentation):
        if documentation not in parsed:
            parsed[documentation] = parse_tool(documentation)
        return parsed[documentation]

    positives, positive_or_negative_docs = ToolIndex(), set()
    rows_by_query = defaultdict(list)  # normalised query -> ids of the training rows with it
    n_rows = 0
    for row_id, query, row_positives, row_negatives in iter_toolret_train():
        n_rows += 1
        rows_by_query[normalise(query)].append(row_id)
        for documentation in row_positives:
            positives.add(tool(documentation))
            positive_or_negative_docs.add(tool(documentation)[0])
        for documentation in row_negatives:
            positive_or_negative_docs.add(tool(documentation)[0])

    # For each evaluation query, fetch the most similar training queries with BM25,
    # then compare word sets.
    train_queries = list(rows_by_query)
    train_words = [set(query.split()) for query in train_queries]
    finder = BM25Retriever()
    finder.index(train_queries)
    eval_queries = [normalise(query.text) for query in bench.queries]
    candidates = finder.search(eval_queries, CANDIDATES)

    exclude = set()
    tasks = defaultdict(Counter)
    tool_match = {}
    for query, text, candidate_rows in zip(bench.queries, eval_queries, candidates):
        counts = tasks[query.group]
        counts["queries"] += 1
        words = set(text.split())
        duplicates = {train_queries[c] for c in candidate_rows
                      if jaccard(words, train_words[c]) >= QUERY_JACCARD}
        if text in rows_by_query:
            counts["queries_identical_to_a_training_query"] += 1
            duplicates.add(text)
        counts["queries_duplicating_a_training_query"] += bool(duplicates)
        for duplicate in duplicates:
            exclude.update(rows_by_query[duplicate])

        matches = []
        for tool_id in query.gold:
            if tool_id not in tool_match:
                tool_match[tool_id] = positives.match(tool(corpus[tool_id]))
            matches.append(tool_match[tool_id])
        counts["gold_tools"] += len(matches)
        counts["gold_tools_matching_a_training_positive"] += sum(m is not None for m in matches)
        counts["queries_with_a_gold_tool_in_training"] += any(m is not None for m in matches)
        counts["queries_with_every_gold_tool_in_training"] += all(m is not None for m in matches)
        for match in matches:
            if match:
                counts[f"gold tools: {match}"] += 1

    report, seen, unseen = {}, [], []
    for task in sorted(tasks):
        counts = dict(tasks[task])
        n = counts["queries"]
        reasons = []
        if task in TRAIN_SOURCES:
            reasons.append("source dataset is in the training set")
        if counts.get("queries_duplicating_a_training_query", 0) / n >= TASK_THRESHOLD:
            reasons.append("queries duplicate training queries")
        if counts.get("queries_with_a_gold_tool_in_training", 0) / n >= TASK_THRESHOLD:
            reasons.append("gold tools are training positives")
        (seen if reasons else unseen).append(task)
        report[task] = {**counts, "seen_source": bool(reasons), "reasons": reasons}

    totals = Counter()
    for counts in tasks.values():
        totals.update(counts)
    corpus_docs = [tool(documentation)[0] for documentation in bench.tool_texts]
    audit = {
        "revisions": {name: REVISIONS[name] for name in REVISIONS if "ToolRet" in name},
        "thresholds": {"query_jaccard": QUERY_JACCARD, "description_ratio": DESCRIPTION_RATIO,
                       "min_description_chars": MIN_DESCRIPTION_CHARS, "task_threshold": TASK_THRESHOLD,
                       "candidates_per_query": CANDIDATES},
        "rule": RULE,
        "train": {"rows": n_rows, "distinct_queries": len(train_queries),
                  "distinct_positive_tools": len(positives.docs),
                  "rows_to_exclude": len(exclude)},
        "corpus": {"tools": len(corpus_docs),
                   "identical_to_a_training_positive": sum(doc in positives.docs for doc in corpus_docs),
                   "identical_to_a_training_positive_or_negative":
                       sum(doc in positive_or_negative_docs for doc in corpus_docs)},
        "totals": dict(totals),
        "seen_source_tasks": seen,
        "unseen_source_tasks": unseen,
        "tasks": report,
    }
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "data_audit.json").write_text(json.dumps(audit, indent=1) + "\n")
    (out / "train_exclude_ids.json").write_text(json.dumps(sorted(exclude)) + "\n")

    print(f"training rows: {n_rows}, to exclude: {len(exclude)}")
    print(f"seen-source tasks ({len(seen)}): {', '.join(seen)}")
    print(f"unseen-source tasks: {len(unseen)}")
    for task in seen + [t for t in unseen if report[t].get("queries_duplicating_a_training_query")
                        or report[t].get("queries_with_a_gold_tool_in_training")]:
        r = report[task]
        print(f"  {task:22} queries {r['queries']:5}  duplicating train {r.get('queries_duplicating_a_training_query', 0):5}"
              f"  with a gold tool in train {r.get('queries_with_a_gold_tool_in_training', 0):5}"
              f"  {'SEEN' if r['seen_source'] else 'unseen'}")


if __name__ == "__main__":
    main()
