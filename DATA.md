# Data

Datasets, licences, formats, splits and known problems. Everything here was read on 6 Oct 2026.

Each fact is tagged with where it came from:

- **[paper]**: the arXiv HTML of the paper, quoted or transcribed.
- **[hub]**: the Hugging Face Hub API, dataset viewer API or the files themselves.
- **[repo]**: the GitHub repository (API or raw files).
- **[probe]**: a throwaway Stage 0 script run over the downloaded files. Probe scripts are not in this repository. The overlap counts no longer come from a probe: since Stage 1 they are produced by the committed audit script.

## Summary

| Dataset | Role here | Hosted at | Licence | Size as released | Official split | Official metrics |
|---|---|---|---|---|---|---|
| ToolRet (evaluation) | In-benchmark test | [`mangopy/ToolRet-Queries`](https://huggingface.co/datasets/mangopy/ToolRet-Queries), [`mangopy/ToolRet-Tools`](https://huggingface.co/datasets/mangopy/ToolRet-Tools) | None declared on the dataset cards | 7,961 queries, 44,453 tools | Evaluation only | nDCG@10, P@10, R@10, C@10 |
| ToolRet-train | Training pairs, hard-negative mining, validation carve-out | [`mangopy/ToolRet-Training-20w`](https://huggingface.co/datasets/mangopy/ToolRet-Training-20w) | None declared on the dataset card | 208,826 rows, 920 MB of parquet | One `train` split, no validation split | None |
| LiveMCPBench | Out-of-distribution test on real MCP tools | [`ICIP/LiveMCPBench`](https://huggingface.co/datasets/ICIP/LiveMCPBench), [`icip-cas/LiveMCPBench`](https://github.com/icip-cas/LiveMCPBench) | Apache-2.0 | 95 tasks, 70 servers, 527 tools | One `test` split | Task success rate judged by an LLM. No retrieval metric |
| Owner's hand-written set | Out-of-distribution test | This repository, not yet written | Owner's choice | 100–150 queries planned | Test only | This project's metrics |
| HumanMCP | Dropped | No public release found | | | | |

## ToolRet (evaluation)

Paper: [Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](https://arxiv.org/abs/2503.01763), ACL 2025, v2 of 26 May 2025. Code: [`mangopy/tool-retrieval-benchmark`](https://github.com/mangopy/tool-retrieval-benchmark).

### Licence

- The code repository is Apache-2.0. [repo]
- None of the dataset cards (`ToolRet-Queries`, `ToolRet-Tools`, `ToolRet-Training-20w`, `ToolRet-before-sample`) declares a licence. [hub]
- The paper's ethics statement says: "The new benchmark is composed of various previous datasets and is therefore licensed under their respective data licenses." [paper]
- The licences of the 35 source datasets were not checked one by one in Stage 0.

Consequence for this project: ToolRet data is never copied into this repository or into any release. Scripts download it from the Hub at run time.

### Format

`ToolRet-Queries` has 35 configs, one per source task, each with a single split named `queries`. [hub]

| Field | Type | Content |
|---|---|---|
| `id` | string | For example `toolbench_query_808` |
| `query` | string | The user request |
| `instruction` | string | An LLM-written instruction (see "The instruction setting" below) |
| `labels` | string | JSON list of `{"id", "doc", "relevance"}`, the gold tools |
| `category` | string | `web`, `code` or `customized` |

`ToolRet-Tools` has 3 configs (`web`, `code`, `customized`), each with a single split named `tools`, and two fields: `id` and `documentation` (a JSON string whose keys differ by source dataset). [hub]

Gold tools must be matched by `id` against `ToolRet-Tools`, not by the `doc` embedded in the label. In the `ultratool` task all 1,024 labels carry one identical embedded `doc`, while their 287 distinct ids all resolve to different tools in the corpus. In at least ten other tasks the embedded docs also differ from the corpus text. [probe]

### Size: the release differs from the paper

| | Paper, Table 1 | Released files |
|---|---|---|
| Queries, total | 7,615 | 7,961 |
| Queries, web | 4,916 | 5,230 |
| Queries, code | 950 | 1,749 |
| Queries, customized | 1,749 | 982 |
| Tools, total | 43,215 | 44,453 |
| Tools, web | 36,978 | 37,292 |
| Tools, code | 3,794 | 3,794 |
| Tools, customized | 2,443 | 3,367 |

Paper column: [paper]. Released column: [hub], confirmed by [probe]. The reason for the difference is not stated anywhere I could find. Two later projects that evaluate on ToolRet both report 7,961 queries and 44,453 tools.

Other facts about the released files: [probe]

- Every query has at least one gold tool and at most eight. 4,402 queries have exactly one.
- The 44,453 tools have 44,401 distinct documentation strings, so there are a few exact duplicates inside the corpus.

### Official metrics and protocol

From the paper, section 5: [paper]

- NDCG@K, Recall@K, Precision@K.
- Completeness@K: "The C@K is 1 if all target tools are included in the top-k retrieved tools; otherwise, it is 0."
- Two settings. "w/o inst.": the query alone. "w/ inst.": the query concatenated with the instruction.
- The paper's "Average" column is the unweighted mean of the web, code and customized scores. Checked against the paper's Table 4: all 26 rows agree within 0.02 on nDCG@10. [probe]

From the released code (`toolret/eval.py`, `toolret/encode.py`): [repo]

- Metrics come from `pytrec_eval` at K = 5, 10, 20. Completeness is computed as `recall@K == 1` and is named `Comprehensiveness@K` in the code.
- Each query is searched against the whole corpus (all three categories together) with exact inner-product search (a FAISS flat index).
- Tools are embedded from the `documentation` string as it is. Maximum length is the model's limit, capped at 2,048 tokens.
- Models loaded through `transformers`, which includes every small model listed below, run in float16 and have their embeddings L2-normalised.
- In the "w/o inst." setting the code still passes an empty instruction, so queries are prefixed with `Instruct: \nQuery: ` (or `Instruct: \nquery: ` for e5 models).
- The only aggregation function in the code weights tasks by query count (a micro-average) and is commented out. It does not match the paper's category mean.

### The instruction setting

ToolRet's instructions were written by GPT-4o. The prompt shown in the paper's appendix B.6 includes the gold tools: "Below, I have provided the target tools (i.e., the labels for the query)... Your instruction should implicitly highlight the task requirements and the characteristics of the target tools relevant to the query." [paper]

So in the "w/ inst." setting the retriever's input contains text written with knowledge of the answer. A deployed retriever does not have that. Owner's decision, 6 Oct 2026: "w/o inst." is this project's primary setting, and "w/ inst." appears only as a labelled secondary column for comparison with published numbers.

### Published reference numbers

Average nDCG@10 from the paper, transcribed from the HTML tables. [paper]

| Model | w/o inst. (Table 4) | w/ inst. (Table 5) |
|---|---|---|
| BM25s | 22.32 | 36.46 |
| all-MiniLM-L6-v2 | 16.30 | 25.53 |
| e5-small-v2 | 19.99 | 31.14 |
| e5-base-v2 | 18.95 | 31.39 |
| bge-base-en-v1.5 | 22.09 | 34.77 |
| gte-base-en-v1.5 | 20.86 | 36.79 |
| NV-Embed-v1 (best embedding model) | 33.83 | 42.71 |

The paper also reports models trained on ToolRet-train (Tables 14 and 15), for example bge-base-en-v1.5 going from 22.06 to 29.35 average nDCG@10. It does not state which of the two settings those trained models were evaluated in. For the untrained rows, the columns labelled Web and Code in Table 14 carry the values that Table 4 lists under Code and Web. [paper]

## ToolRet-train

### Source and licence

The paper says the training set extends the collection "to include the training sets of three mainstream tool-use datasets: ToolACE, ToolBench and APIGen". [paper]

| Source | Hosted at | Licence |
|---|---|---|
| ToolACE | [`Team-ACE/ToolACE`](https://huggingface.co/datasets/Team-ACE/ToolACE) | apache-2.0 [hub] |
| ToolBench | [`OpenBMB/ToolBench`](https://github.com/OpenBMB/ToolBench) | Apache-2.0 [repo] |
| APIGen | [`Salesforce/xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | cc-by-4.0, and gated: access requires accepting the terms on the Hub [hub] |

Owner's decision, 6 Oct 2026: weights trained on this set may be published, with these three sources and their licences named in the model card. The data itself is never redistributed.

### Format

One config, one split (`train`), six parquet shards. [hub]

| Field | Type | Content |
|---|---|---|
| `id` | string | `train_0`, `train_1`, ... No field names the source dataset or the API provider |
| `query` | string | The user request |
| `prompt` | string | The LLM-written instruction |
| `positive` | list of strings | Gold tools, each a JSON string |
| `negative` | list of strings | Negative tools, each a JSON string |

### Size: the release differs from the paper

| | Paper | Released files |
|---|---|---|
| Rows | 205,826 (Table 7) | 208,826 [hub] |
| Negatives per row | 10 in the text of sections 1 and 7.2; 5 in Table 7 | Exactly 15 in every row [probe] |
| Positives per row | 2.31 on average (Table 7) | 1 to 10 [probe] |

The released training script (`script/train.sh`) uses a learning rate of 1e-6; the paper says 5e-5. [repo] [paper]

The negatives were mined with NV-Embed-v1 by the ToolRet authors. [paper] They are not this project's hard negatives: Stage 2 mines its own from a Stage 1 baseline.

### How this project uses it (Stage 2)

Implemented in `src/toolshortlist/train.py`; the counts for a run are in its `summary.json`.

- The rows listed in `results/train_exclude_ids.json` are dropped first.
- Validation is held out by tool, with a fixed seed shared by every run: 500 gold tools are held out; a row whose gold tools are all held out is a validation row (at most 1,000 are kept), a row with none of them is a training row, and a row with a mix is dropped. The validation corpus is the held-out tools plus 10,000 other training tools. Checkpoints are selected on validation nDCG@10 and nothing else.
- Hard negatives come from the training rows only: for each training query, untrained `bge-small-en-v1.5` ranks the training tools, and ranks 5 to 30 are kept after removing the query's positives and any tool with the same name and description as one (up to 8 per query).
- The released negatives in the `negative` field are not used.

## Train and evaluation overlap in ToolRet

ToolRet has no single pool split into train and test. The training set is a separate dataset built from three sources, and the same three sources (`toolace`, `toolbench`, `apigen`) are also three of the 35 evaluation tasks. The paper does not discuss overlap between the two. [paper]

The overlap is measured by `python -m toolshortlist.audit`, which writes `results/data_audit.json`. The counts, the rule and its thresholds are rendered in [results/data_audit.md](results/data_audit.md); no count is repeated here. What the audit found:

- Most `toolbench` evaluation queries are in the training set word for word, usually with the same gold tools. Example: evaluation query `toolbench_query_875` is training row `train_345`.
- Almost every `apigen` evaluation query, and a large share of `toolace` queries, has a gold tool that is also a training positive.
- `toollens`, `autotools-music` and `autotools-weather` use the same RapidAPI endpoints as ToolBench under different tool names. Exact matching misses this; matching on descriptions finds it.
- `toolalpaca` is classed as seen-source because of a single match on a generic description ("convert currency values from one currency to another"). This looks like a coincidence, not real overlap. The rule was fixed before the counts were seen, so it stands, and the task's own scores are in every results file for anyone who wants to regroup it.
- Every distinct training positive is a tool in the evaluation corpus.

Rules this project follows as a result (approved by the owner on 6 Oct 2026):

1. Remove from training every row whose query duplicates an evaluation query. The ids are in `results/train_exclude_ids.json`.
2. Report ToolRet in two groups: seen-source tasks and unseen-source tasks, as listed in `results/data_audit.json`. The unseen-source group is the headline in-benchmark number.

How much the overlap changes a published score is measured in [results/baselines.md](results/baselines.md), section 5.

## LiveMCPBench

Paper: [LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools?](https://arxiv.org/abs/2508.01780), v2 of 26 Feb 2026.

### Licence

Apache-2.0, on both the GitHub repository and the Hub dataset card. [repo] [hub] The tool descriptions were collected from 70 third-party MCP servers, each with its own repository. Those licences were not checked one by one.

### Format

| File | Content |
|---|---|
| `tasks/tasks.json` on the Hub (the `test` split), identical in size to `annotated_data/all_annotations.json` on GitHub | 95 tasks |
| `tools/tools.json` on the Hub | 70 servers, 527 tools |
| `tools/LiveMCPTool/tools.json` on GitHub | 69 servers, 525 tools. The repository describes this list as the tools "verified to work correctly" |

Task fields: `task_id`, `Question`, `answers`, `category`, `file_name`, and `Annotator Metadata` with `Steps`, `Number of steps`, `Tools`, `Number of tools`. [hub]

Server fields: `name`, `description`, `web`, `config`, `category`, `tools`. Each tool has `name`, `description`, `inputSchema`, `annotations`. [hub]

Categories: Office 31, Lifestyle 15, Leisure 14, Finance 14, Travel 12, Shopping 9. [probe]

### Official metric

Task success rate, judged by an LLM over the agent's trajectory (the paper's LiveMCPEval). The paper states why it avoids tool matching: "multiple tool invocation trajectories may lead to equally valid solutions. Such properties make traditional metrics, such as exact tool-matching accuracy, inadequate." [paper]

There is no official retrieval metric and no official retrieval label set. Any retrieval number this project reports on LiveMCPBench uses the project's own protocol and must be labelled that way.

### Using it as a retrieval test

The `Tools` field lists the tool names one annotator used to solve each task. This project's protocol, implemented in `load_livemcpbench` in `src/toolshortlist/data.py`:

- The query is the task text (`Question`).
- The gold tools are the annotated names that exist in the catalogue. Names that do not exist are dropped, and a task left with no gold tool is dropped.
- Annotations give a tool name without a server, and some names exist on several servers (`search`, `fetch`, `list_directory` and others). Labels are therefore matched by name: a gold name counts as retrieved when any tool with that name is retrieved.
- Each tool is indexed as JSON with its server name, tool name, description and parameters.

The resulting counts (tasks kept, labels dropped, labels with a shared name) are written to the `notes` of the LiveMCPBench entry in every results file. The Hub copy of the catalogue is used because it matches the paper; retrieval does not need the tools to run.

Other facts about the catalogue: 61 tool descriptions contain Chinese characters and 19 are empty. [probe]

Limits to state wherever these numbers appear:

- The labels are one valid solution path, not every acceptable tool.
- Matching by name is lenient where two servers offer different tools under one name.
- Queries are whole multi-step tasks, not single-tool requests.
- With fewer than 100 queries, one query is about one percentage point. Results carry bootstrap confidence intervals.

## Owner's hand-written set

Not yet written. Rules from PROJECT.md: 100–150 queries, aiming for 150, over tool schemas from open-source MCP servers, written and labelled by the owner, not generated, using servers that appear in no training data. Record each server's licence here when the set is built.

File format, read by `load_handwritten` in `src/toolshortlist/data.py`. Once both files exist, every evaluation run includes the set.

`testsets/handwritten/tools.json`: a list with one object per tool, copied from the server's own tool listing.

```json
[
  {"server": "filesystem", "name": "read_file", "description": "Read a file from disk.",
   "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}}
]
```

`testsets/handwritten/queries.jsonl`: one query per line. `tools` lists every tool that must be retrieved, as `server/name`.

```json
{"id": "h001", "query": "Show me what's in notes.txt", "tools": ["filesystem/read_file"]}
```

## HumanMCP

Paper: [arXiv 2602.23367](https://arxiv.org/abs/2602.23367). No public release was found. The paper has no data or code link, the Hub lists no repository linked to the paper, and searches of the Hub and GitHub for the name, the arXiv id and the authors found nothing. Dropped, as PROJECT.md allows.

## Candidate extra test set, not adopted

MCP-Atlas ([arXiv 2602.00933](https://arxiv.org/abs/2602.00933)) has a public split at [`ScaleAI/MCP-Atlas`](https://huggingface.co/datasets/ScaleAI/MCP-Atlas): 500 tasks, cc-by-4.0, not gated. [hub] Each row has a prompt, the list of enabled tool names and a reference trajectory with the tool calls. It contains tool names but no tool descriptions, so using it for retrieval would mean collecting the schemas from the 36 MCP servers first. Owner's decision, 6 Oct 2026: not adopted for now.

## Base models and reference models

Licence and parameter count as reported by the Hub. [hub]

| Model | Parameters | Licence | Role |
|---|---|---|---|
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22,713,728 | apache-2.0 | Baseline and candidate base |
| [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | 33,360,512 | mit | Baseline and candidate base |
| [`intfloat/e5-small-v2`](https://huggingface.co/intfloat/e5-small-v2) | 33,360,512 | mit | Baseline with a published ToolRet score |
| [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) | 109,482,752 | mit | Baseline and candidate base |
| [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) | 109,482,752 | mit | Baseline with a published ToolRet score |
| [`mangopy/ToolRet-trained-bge-base-en-v1.5`](https://huggingface.co/mangopy/ToolRet-trained-bge-base-en-v1.5) | 109,482,240 | None declared | The ToolRet authors' trained checkpoint. Evaluate only, never redistribute |
| [`mangopy/ToolRet-trained-e5-base-v2`](https://huggingface.co/mangopy/ToolRet-trained-e5-base-v2) | 109,482,240 | None declared | Same |
| [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 595,776,512 | apache-2.0 | LiveMCPBench's own retriever. Reference only, above this project's size limit |

## Open items

- Licences of ToolRet's 35 source datasets and LiveMCPBench's 70 servers, one by one.
- Why the released ToolRet files are larger than the paper states.

Resolved in Stage 1:

- The paper's "Average" is the mean over tasks within each category, then the mean of the three categories. With that aggregation BM25 reproduces the published numbers (see [results/baselines.md](results/baselines.md), section 4).
- The released category labels agree with the paper's result tables. The paper's Table 1 appears to have the code and customized query counts the wrong way round.
- Near-duplicate overlap, including `toollens`: see the section on overlap above.
