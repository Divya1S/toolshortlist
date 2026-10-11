"""Loaders for the test sets and for ToolRet's training set.

Public data is downloaded from the Hugging Face Hub at run time, at pinned
revisions, into the Hub cache. It is never stored in this repository. Formats
and licences are described in DATA.md.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

# Dataset revisions, pinned on 6 Oct 2026.
REVISIONS = {
    "mangopy/ToolRet-Queries": "b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445",
    "mangopy/ToolRet-Tools": "e06c38c75612b6536bd959e08cdd345894aba6a7",
    "mangopy/ToolRet-Training-20w": "fdf5a317455b1e60785de7ba587496aa6cc878e4",
    "ICIP/LiveMCPBench": "ddea2d24196638bc4026c4cb891f679d0357bfd0",
}

TOOLRET_CATEGORIES = ["web", "code", "customized"]


@dataclass
class Query:
    id: str
    text: str
    gold: dict[str, int]  # key of each relevant tool -> relevance grade (> 0)
    group: str  # the ToolRet task, or the LiveMCPBench category
    instruction: str | None = None


@dataclass
class Benchmark:
    name: str
    tool_ids: list[str]
    tool_texts: list[str]  # what a retriever indexes
    tool_keys: list[str]  # what gold labels refer to; the tool id unless stated otherwise
    queries: list[Query]
    splits: dict[str, list[str]] = field(default_factory=dict)  # split name -> groups in it
    report_ci: bool = False  # report bootstrap confidence intervals (small test sets)
    notes: dict = field(default_factory=dict)  # facts about how the set was built


def _snapshot(repo_id: str) -> Path:
    return Path(snapshot_download(
        repo_id, repo_type="dataset", revision=REVISIONS[repo_id], allow_patterns=["*.parquet"],
    ))


def load_toolret() -> Benchmark:
    """ToolRet's evaluation set: every query is searched against the whole tool corpus."""
    tools_dir = _snapshot("mangopy/ToolRet-Tools")
    tool_ids, tool_texts = [], []
    for category in TOOLRET_CATEGORIES:
        for row in pq.read_table(next((tools_dir / category).glob("*.parquet"))).to_pylist():
            tool_ids.append(row["id"])
            tool_texts.append(row["documentation"])

    queries_dir = _snapshot("mangopy/ToolRet-Queries")
    known = set(tool_ids)
    queries, splits = [], {category: [] for category in TOOLRET_CATEGORIES}
    for path in sorted(queries_dir.glob("*/*.parquet")):
        task = path.parent.name
        for row in pq.read_table(path).to_pylist():
            # Match gold tools by id. The `doc` embedded in a label is unreliable (DATA.md).
            gold = {str(label["id"]): int(label["relevance"])
                    for label in json.loads(row["labels"]) if int(label["relevance"]) > 0}
            assert gold and set(gold) <= known, f"bad labels in {row['id']}"
            queries.append(Query(row["id"], row["query"], gold, task, row["instruction"]))
            if task not in splits[row["category"]]:
                splits[row["category"]].append(task)
    return Benchmark("toolret", tool_ids, tool_texts, list(tool_ids), queries, splits)


def iter_toolret_train():
    """Yield (id, query, positives, negatives) for each row of ToolRet's training set.

    Positives and negatives are lists of tool documentation strings (JSON).
    """
    train_dir = _snapshot("mangopy/ToolRet-Training-20w")
    for path in sorted(train_dir.glob("**/*.parquet")):
        table = pq.read_table(path, columns=["id", "query", "positive", "negative"])
        yield from zip(*(table.column(name).to_pylist() for name in table.column_names))


def mcp_tool_text(server: str, name: str, description: str, input_schema: dict | None) -> str:
    """One MCP tool as the text a retriever indexes: server, name, description, parameters."""
    return json.dumps({
        "server": server,
        "name": name,
        "description": description or "",
        "parameters": (input_schema or {}).get("properties", {}),
    }, ensure_ascii=False)


def load_livemcpbench() -> Benchmark:
    """LiveMCPBench as a retrieval test, under this project's own protocol (DATA.md).

    The query is the task text. The gold tools are the tool names its annotator
    listed. Annotations give a name without a server, and some names exist on
    several servers, so labels are matched by tool name: a gold name counts as
    retrieved when any tool with that name is retrieved.
    """
    repo = "ICIP/LiveMCPBench"
    def load(filename):
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=REVISIONS[repo])
        return json.loads(Path(path).read_text())

    tool_ids, tool_texts, tool_keys = [], [], []
    for server in load("tools/tools.json"):
        for server_name, entry in server["tools"].items():
            for tool in entry["tools"]:
                tool_ids.append(f"{server_name}/{tool['name']}")
                tool_keys.append(tool["name"])
                tool_texts.append(mcp_tool_text(
                    server_name, tool["name"], tool.get("description"), tool.get("inputSchema")))

    names = set(tool_keys)
    shared = {name for name in names if tool_keys.count(name) > 1}
    tasks = load("tasks/tasks.json")
    queries = []
    notes = {"tasks": len(tasks), "gold_mentions": 0, "gold_mentions_not_in_catalogue": 0,
             "gold_mentions_with_a_name_on_several_servers": 0, "tasks_dropped_without_gold": 0}
    for task in tasks:
        listed = [re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
                  for line in task["Annotator Metadata"]["Tools"].split("\n") if line.strip()]
        notes["gold_mentions"] += len(listed)
        notes["gold_mentions_not_in_catalogue"] += sum(name not in names for name in listed)
        notes["gold_mentions_with_a_name_on_several_servers"] += sum(name in shared for name in listed)
        gold = {name: 1 for name in listed if name in names}
        if not gold:
            notes["tasks_dropped_without_gold"] += 1
            continue
        queries.append(Query(task["task_id"], task["Question"], gold, task["category"]))
    return Benchmark("livemcpbench", tool_ids, tool_texts, tool_keys, queries,
                     report_ci=True, notes=notes)


def load_handwritten(directory: str = "testsets/handwritten") -> Benchmark | None:
    """The owner's hand-written test set, or None if it has not been written yet.

    Expected files (format documented in DATA.md):
      tools.json     list of {"server", "name", "description", "inputSchema"}
      queries.jsonl  one {"id", "query", "tools": ["server/name", ...]} per line
    """
    directory = Path(directory)
    if not (directory / "tools.json").exists() or not (directory / "queries.jsonl").exists():
        return None
    tool_ids, tool_texts = [], []
    for tool in json.loads((directory / "tools.json").read_text()):
        tool_ids.append(f"{tool['server']}/{tool['name']}")
        tool_texts.append(mcp_tool_text(
            tool["server"], tool["name"], tool.get("description"), tool.get("inputSchema")))
    assert len(set(tool_ids)) == len(tool_ids), "duplicate server/name in tools.json"
    queries = []
    for line in (directory / "queries.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            assert row["tools"] and set(row["tools"]) <= set(tool_ids), f"bad labels in {row['id']}"
            queries.append(Query(row["id"], row["query"], {tool: 1 for tool in row["tools"]},
                                 row["tools"][0].split("/")[0]))
    return Benchmark("handwritten", tool_ids, tool_texts, list(tool_ids), queries, report_ci=True)


LOADERS = {"toolret": load_toolret, "livemcpbench": load_livemcpbench, "handwritten": load_handwritten}
