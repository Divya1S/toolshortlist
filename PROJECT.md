# ToolShortlist: project brief

A small, trained retrieval model that shortlists the right tools for an AI agent out of a large catalog, measured against public benchmarks and shipped as an installable package with a live demo.

The name `toolshortlist` was checked on 6 Oct 2026: nothing by that name on PyPI, npm or Hugging Face (models and datasets), and no GitHub repository with it in the name. It is the package name, the import name and the repository name. None of these is reserved until something is published, so check again at Stage 4 before the first release.

## 1. The problem

An agent connected to hundreds or thousands of tools cannot put every tool definition in its context, and when it tries, it picks badly. The usual fix is a retrieval step: given the user's request, return the top-k candidate tools and show only those to the LLM. That retrieval step is where much of the failure sits:

- LiveMCPBench (95 tasks, 70 servers, 527 tools) reports that "retrieval errors account for nearly half of all failures", with most models finishing 30–50% of tasks. [arXiv 2508.01780](https://arxiv.org/abs/2508.01780), read 6 Oct 2026.
- ToolRet (7.6k retrieval tasks, 43k tools) reports that the best off-the-shelf embedding model scores 33.83 nDCG@10, and that models trained on its training set improve. [arXiv 2503.01763](https://arxiv.org/pdf/2503.01763), confirmed against the paper on 6 Oct 2026. The 33.83 belongs to NV-Embed-v1, a 7B-parameter model, given the query alone; given ToolRet's instruction as well, it scores 42.71. The released files are larger than the paper states: 7,961 queries and 44,453 tools (see `DATA.md`).

## 2. The claim this project tests

> A small embedding model (under about 150M parameters), fine-tuned for tool retrieval at zero cost, retrieves better than BM25 and off-the-shelf small embedders on tool sources it did not see in training, and runs fast enough on a CPU to sit in front of any agent.

"Tool sources it did not see in training" means three test sets: the ToolRet tasks whose source dataset is not in the training set, LiveMCPBench, and the owner's hand-written set (section 6). Results on the ToolRet tasks that share a source with the training set are reported separately and do not count towards the claim.

The claim was narrowed on 6 Oct 2026. It used to say "on public benchmarks", but others have already shown that fine-tuning on ToolRet's training set raises ToolRet scores (section 9), and that training set overlaps the evaluation set (`DATA.md`).

The claim may turn out false. A hybrid of BM25 and embeddings is a strong baseline. If the trained model loses, the project reports that, and the failure analysis becomes the main result. No result is to be shaped to fit the claim.

## 3. Scope

In scope:
- Retrieval only: query in, ranked list of tools out.
- English queries.
- One end-to-end check that better retrieval makes a small local LLM pick the right tool more often.

Out of scope:
- Executing tools, multi-step agent planning, argument filling.
- Training or fine-tuning a generative LLM.
- Kubernetes, cloud deployments, paid infrastructure of any kind.
- A new benchmark presented as a research contribution. Several exist already (section 9).

## 4. Hard constraints

1. **Zero cost to build and to reproduce.** No paid API, no service that needs a credit card, no paid plan. If a step seems to need one, stop and say so. The single exception is the optional $3 OpenAI budget in section 5.1, which adds reference rows to the results and is never on the path needed to train, evaluate or reproduce the project.
2. **No invented numbers.** Every figure in the README or write-up is produced by a script and stored in `results/*.json`. Tables are rendered from those files, never typed by hand.
3. **No test leakage.** See section 6. Test data is never used for training, hard-negative mining, checkpoint selection or hyperparameter choice.
4. **Reproducible.** Fixed seeds, pinned dependencies, one command per stage. A stranger can rerun the evaluation from a clean clone.
5. **Licences respected.** Record each dataset's and base model's licence in `DATA.md` before using it. Do not redistribute data whose licence forbids it. ToolRet's dataset cards declare no licence, so ToolRet data is never copied into this repository or any release; scripts download it at run time. Owner's decision, 6 Oct 2026: weights trained on ToolRet's training set may be published, with the training sources and their licences named in the model card.
6. **Secrets stay out of the repo.** Keys come from environment variables only.

## 5. Environment

- Development machine: Apple Silicon Mac (arm64), 16 GB RAM, Python 3.13, `uv`, `git`. Checked 6 Oct 2026.
- Ollama is installed with `qwen2.5:7b`, `llama3.2:3b`, `llama3.2:1b` and `nomic-embed-text`. Use these for the end-to-end check in Stage 3.
- GPU training: Kaggle Notebooks free GPU. Kaggle's documentation gives the weekly quota as "30 hours or sometimes higher depending on demand and resources" (confirmed 6 Oct 2026). A session offers two T4 GPUs or one P100, runs for at most 12 hours and saves up to 20 GB of output. GPU and internet access need a phone-verified account, not a credit card. The agent cannot run Kaggle itself: it prepares a self-contained notebook or script, and the owner runs it and brings back the checkpoint and logs.
- Small runs (models around 30M parameters, a few thousand steps) can run locally on MPS for debugging.
- CI and hosting: GitHub Actions and GitHub Pages on a public repo. Both are free there (confirmed 6 Oct 2026). The standard Linux runner has 4 vCPUs and 16 GB of RAM. GitHub blocks files over 100 MiB, and Git LFS does not work with Pages.
- Model hosting: a public Hugging Face Hub model repo. Free for public repositories, with no per-repository size limit (confirmed 6 Oct 2026).

### 5.1 Optional paid budget: $3 of OpenAI credit

The owner has an OpenAI API key with $3 of prepaid credit. It is used only in Stage 3, only for reference comparisons, and in this order of priority:

1. **A hosted embedding baseline.** Run one OpenAI embedding model through the same evaluation harness, so the results table can say how a small free model compares with a paid hosted one. This is the most useful thing the money can buy.
2. **A stronger LLM in the end-to-end check.** Repeat the three retrieval conditions with one low-cost OpenAI chat model, to show whether the retriever's benefit holds beyond the small local model.

Not to be spent on: generating training data, anything in Stages 1 or 2, anything in CI, or anything a reader would need in order to reproduce the main results.

Spending rules:
- Before any call, confirm current model names and prices on OpenAI's pricing page. Do not rely on remembered prices.
- Before any batch, count the tokens locally with the tokenizer, print the estimated cost, and wait for the owner's approval.
- Keep a running total in `results/spend.json` and refuse to start a batch that would take the total above $2.50. The last $0.50 is a margin for estimation error.
- Cache every response on disk, keyed by model and input hash, so nothing is paid for twice and a crashed run resumes where it stopped.
- If the whole ToolRet corpus is too expensive to embed, use the LiveMCPBench tool set and the owner's hand-written set, and say in the table which test sets the hosted baseline covers.
- The key is read from the `OPENAI_API_KEY` environment variable. It never goes into a file, a notebook, a commit, a CI secret or a chat message.
- Check OpenAI's terms before committing cached embeddings or completions to the public repo. If redistribution is not allowed, commit only the metrics.
- Label these rows in every table as paid references, with the model name and the date run.

## 6. Data and splits

| Dataset | Use | Status |
|---|---|---|
| ToolRet training set (`mangopy/ToolRet-Training-20w`) | Training pairs, hard-negative mining and the validation carve-out | A separate dataset, not a split of the evaluation pool. Built from the training sets of ToolACE, ToolBench and APIGen. No licence declared on the dataset card |
| ToolRet evaluation set, unseen-source tasks | Main in-benchmark test and the headline number | The tasks that share no source with the training set. The list and the rule that decides it are in `results/data_audit.md`, produced by the Stage 1 audit |
| ToolRet evaluation set, seen-source tasks | Reported separately, never as the headline | The three training sources (`toolbench`, `toolace`, `apigen`) plus the tasks the audit found to use the same tools, `toollens` among them. Listed in `results/data_audit.md` |
| LiveMCPBench tool set and tasks | Out-of-distribution test on real MCP tools | Apache-2.0. 95 tasks, and no official retrieval labels or metric: the annotators' tool lists serve as labels under this project's own protocol |
| Owner's hand-written set | 100–150 queries over tool schemas from open-source MCP servers | Written and labelled by the owner, not generated. Aim for 150 (decided 6 Oct 2026): this set and LiveMCPBench are the only MCP test sets |
| ~~HumanMCP (2,800 tools, 308 servers)~~ | Dropped | No public release found on 6 Oct 2026 |
| MCP-Atlas public split (500 tasks) | Not adopted, by the owner's decision of 6 Oct 2026 | cc-by-4.0, but it holds tool names without descriptions. Revisit only if the two MCP test sets prove too small |

Where each dataset lives, its format and its licence are in `DATA.md`.

Leakage rules:
- ToolRet's training set overlaps its evaluation set (`DATA.md`). Remove from training every row whose query matches an evaluation query, exactly or as a near-duplicate.
- Report ToolRet in two groups, seen-source and unseen-source tasks, as in the table above.
- Stage 1 counts the exact and near-duplicate queries and tools (same name and near-identical description) shared between the training and evaluation sets with a committed script, and writes the counts to `results/`.
- Any split made in this project is by tool, never by query: no gold tool of a held-out row is a gold tool of a training row. Where the source is known (a ToolRet task, an MCP server), split by source. ToolRet's training rows carry no source or provider field, so the validation split there is by gold tool.
- The owner's hand-written set uses servers that appear in no training data.
- Keep a small validation split carved from training data for checkpoint selection.

## 7. Metrics

Retrieval quality:
- Recall@1, Recall@5, Recall@10
- nDCG@10
- Completeness@10 (all required tools appear in the top 10), as ToolRet defines it

Cost of use:
- Model size on disk, before and after quantization
- CPU latency per query at p50 and p95, for catalog sizes of 100, 1,000 and 10,000 tools, on the development Mac and on a GitHub Actions runner
- Index build time

Evaluation setting:
- The query alone is the primary setting (owner's decision, 6 Oct 2026). ToolRet's "with instruction" setting appears only as a labelled secondary column, for comparison with published numbers. Those instructions were written by GPT-4o with the gold tools in its prompt, so they give the retriever information a deployed system does not have.
- On LiveMCPBench and the hand-written set, report a bootstrap 95% confidence interval with every metric. With about 100 queries, one query is about one percentage point.

Report every trained-model number as mean and spread over 3 seeds where the GPU budget allows; say so where it does not.

## 8. Stages

Each stage ends with something that works and could be published as it stands. Do not start a stage before the previous one meets its exit criteria.

### Stage 0: ground truth (2 days)

- Read the ToolRet and LiveMCPBench papers and repos. Skim the prior work in section 9.
- Confirm where each dataset lives, its licence, its format and its official metrics. Write `DATA.md`.
- Confirm the free-tier terms in section 5 from the providers' own pages.
- Write one paragraph in `README.md` stating how this project differs from prior work.

Exit: `DATA.md` complete; every "to confirm" in this brief resolved or struck out; the owner has approved the angle.

### Stage 1: evaluation harness and baselines (week 1)

- Data loaders and a single `evaluate` entry point that takes any retriever and writes `results/<name>.json`.
- Baselines: BM25; two or three off-the-shelf small embedders (candidates: `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, one model near 110M parameters); a hybrid using reciprocal rank fusion.
- A reference baseline: the ToolRet authors' trained checkpoint `mangopy/ToolRet-trained-bge-base-en-v1.5`, scored per task. It shows how much of the published gain falls on the seen-source tasks. Evaluate it only; it has no declared licence, so never redistribute it.
- The overlap audit from section 6, written to `results/`.
- Reproduce at least one number published in the ToolRet paper for a model small enough to run locally. If it does not reproduce within a reasonable tolerance, find out why before going on. Candidates, both average nDCG@10 with the query alone: BM25s 22.32 and all-MiniLM-L6-v2 16.30. The released files hold more queries than the paper reports, so an exact match may not be possible.
- Unit tests for the metrics using tiny hand-computed cases.

Exit: a baseline table rendered from `results/`; the overlap counts in `results/`; one published number reproduced, or the gap explained.

### Stage 2: training (weeks 2–3)

- A training loop in plain PyTorch with Hugging Face `transformers`: a bi-encoder trained with a contrastive (InfoNCE) loss using in-batch negatives plus mined hard negatives. Keep it short and readable; the owner must be able to explain every line.
- Hard negatives mined from the training split only, using a Stage 1 baseline to find tools that look right and are wrong.
- Ablations, each as one config change:
  - tool text: name only, name and description, name and description and parameter schema
  - hard negatives on and off
  - base model size (about 22M, 33M, 110M)
- A Kaggle-ready notebook that clones the repo, runs one config and saves the checkpoint and logs.

Exit: a trained checkpoint evaluated on every test set; an ablation table; the claim in section 2 answered as supported, partly supported or not supported.

### Stage 3: failure analysis and end-to-end check (week 4)

- Slice results by catalog size, query vagueness and tool-description length. Sample at least 100 failures of the best model and sort them into named categories with counts.
- End-to-end: a local Ollama model picks one tool for each of about 200 test queries under three conditions: all tools in context (only where the catalog fits), BM25 top-k, and this retriever's top-k. Report tool-selection accuracy for each.

- Optional, under the rules in section 5.1: add the hosted embedding baseline to the retrieval table, then the hosted LLM rows to the end-to-end table.

Exit: a failure taxonomy with counts and examples; the end-to-end table. The paid reference rows are not required for exit.

### Stage 4: ship (week 5)

- Export to ONNX and quantize to int8. Re-run the full evaluation on the quantized model and report the accuracy change.
- Python package with a small API (`index(tools)`, `search(query, k)`) and a CLI.
- A minimal MCP server exposing one tool, `find_tools(query, k)`, over an indexed catalog.
- A static demo on GitHub Pages that runs the quantized model in the browser: type a request, see the ranked tools and the latency. The demo uses the 22M or 33M model. Its model file must stay under GitHub's 100 MiB limit or be fetched from the Hugging Face Hub.
- GitHub Actions: unit tests on every push, plus an evaluation gate on a fixed small subset that fails when Recall@5 drops below the committed reference by more than a set margin.
- Publish the model to the Hugging Face Hub with a model card that states training data, metrics, limits and licence.

Exit: `pip install` works from a clean environment; the demo link loads and answers in a fresh browser; CI is green and the gate is shown to fail on a deliberately worse model.

### Stage 5: write-up (owner)

The owner writes the README results section and a short post: the problem, the baselines, what was trained, what the numbers say, where it fails, what was learned. The agent may propose an outline and check every figure against `results/`, but does not write the prose.

## 9. Prior work to read before choosing the angle

All of these were read from their primary pages on 6 Oct 2026. The agent read ToolRet and LiveMCPBench in full, and `toolrank` and the StackOne post at source. Its sub-agents read the rest, and those readings were not re-checked except where noted. Numbers below are the sources' own claims.

Benchmarks and papers:

- ToolRet, "Retrieval Models Aren't Tool-Savvy": https://arxiv.org/pdf/2503.01763. The authors also publish checkpoints fine-tuned on their training set, including bge-base-en-v1.5 and e5-base-v2 at 109M parameters each: https://huggingface.co/mangopy. The paper reports bge-base-en-v1.5 rising from 22.06 to 29.35 average nDCG@10. The checkpoints declare no licence.
- LiveMCPBench: https://arxiv.org/abs/2508.01780
- MCP-Atlas (36 servers, 220 tools, 1,000 tasks): https://arxiv.org/pdf/2602.00933. An agent benchmark, not a retrieval one, and it trains nothing. 500 tasks are public.
- HumanMCP: https://arxiv.org/abs/2602.23367. No public release found. It reports that untuned all-MiniLM-L6-v2 reaches an 87.60% top-10 hit rate on a pool of 2,000 MCP tools with LLM-written queries, so MCP retrieval with such queries may leave little room to improve.
- "Tools are Under-Documented: Simple Document Expansion Boosts Tool Retrieval" (ICLR 2026, since renamed Tool-REX): https://arxiv.org/pdf/2510.22670. Expands ToolRet's tool documents with an LLM and fine-tunes Qwen3-Embedding at 0.6B and 4B parameters. No MCP evaluation.
- "Tool Retrieval Bridge": https://arxiv.org/pdf/2604.07816. A 3B LLM rewrites vague queries before retrieval; the retriever itself is not trained.
- "Enrich-Retrieve-Rank: Scaling Capability Discovery Beyond In-Context Routing": https://arxiv.org/pdf/2608.22695. Trains nothing. On a ToolRet-derived registry it finds that about 70% of misses happen at the retrieval stage, and that a BERT retriever fine-tuned on ToolBench does not beat general encoders.
- RAG-MCP: https://arxiv.org/abs/2505.03275. 13.62% accuracy with all tools in context against 43.13% with retrieval first, from 20 trials on the web-search subset of MCPBench. The pool size and the retriever are not specified, so this is weak evidence.
- ToolRet became an MTEB retrieval task (`ToolRetrieval`) on 11 Sep 2026, so leaderboard scores for small models may appear.

Models and packages:

- `toolrank`: https://github.com/yasinyaman/toolrank, on PyPI since 30 Sep 2026. The closest project in design, and a retriever, not a quality scorer. It trains Qwen3-Embedding-8B with a LoRA on 20,000 of ToolRet's training pairs, evaluates on ToolRet, on LiveMCPBench as a held-out set and on MCP-Zero, and ships a pip package with an MCP server. It needs a GPU with 16 GB of memory. On LiveMCPBench it reports 50.82 Recall@5 for the untrained 8B model and 52.06 after training. It does not mention the overlap between ToolRet's training and evaluation sets.
- StackOne v2 (12 May 2026): https://www.stackone.com/blog/autoresearch-charged-action-search/. A fine-tuned 109M BGE-base reported at 0.544 nDCG@10 on ToolRet. The training data and protocol are not stated and no weights were found.
- `nomos-v1-nano-g1`: https://huggingface.co/yafitzdev/nomos-v1-nano-g1. A 33M bge-small fine-tune for tool routing on a CPU, evaluated on a 60-query sample of ToolRet.
- ToolScout v0.1 (model card read 6 Oct 2026): https://huggingface.co/nuhmanpk/toolscout-bge-small. The closest existing project at the small end: a 33M `bge-small-en-v1.5` fine-tune for tool retrieval, trained on a Kaggle T4, published 27 Sep 2026. It is trained on 407 mostly templated examples and tested on a 36-tool catalog, where the untuned base model already reaches 95.7% top-1 on unseen tools. This project must differ in substance: public benchmarks, catalogs of thousands of tools, real MCP tool definitions, a failure analysis and an installable package. Do not reuse its name.
- `toolsift`: https://github.com/abdulmunimjemal/toolsift, a local MCP proxy aimed at the same problem. It uses BM25 and optional off-the-shelf embeddings, trains nothing, and is evaluated on 28 tools.

Vendor numbers and features:

- Vendor comparisons relayed in https://mcpproxy.app/blog/2026-03-15-beyond-bm25-tool-discovery/, corrected against the vendors' own posts. StackOne's 14% is BM25 top-1 on the 540-case test split of an internal 270-tool suite. Stacklok's 94% is an agent's tool-selection accuracy over 2,792 tools, not retrieval accuracy.
- Built-in tool search: Anthropic's tool search tool offers regex and BM25 variants and accepts a custom retriever; OpenAI's tool search does not state its method. ToolShortlist would plug in as the custom retriever, so it complements these features.

What nobody was found to have published: a model of 33M parameters or fewer trained on ToolRet's training set and scored on the full benchmark; a CPU-only int8 package with a browser demo; an evaluation that separates tool sources seen in training from unseen ones.

Angle, approved by the owner on 6 Oct 2026: small (22M to 110M parameters), CPU-only and installable; evaluated separately on tool sources seen and not seen in training, after auditing the overlap in ToolRet; and tested on real MCP tool catalogs that appear in no training data.

## 10. Repository layout

```
toolshortlist/
  PROJECT.md            this brief
  README.md             what it is, results table (rendered), how to reproduce
  DATA.md               datasets, licences, splits, overlap counts
  pyproject.toml
  src/toolshortlist/    data, retrievers, train, evaluate, export, server
  configs/              one file per experiment
  notebooks/            Kaggle training notebook
  results/              JSON written by scripts; source of every reported number
  tests/
  demo/                 static in-browser demo
  .github/workflows/
```

## 11. The owner's jobs

These stay with the owner because they are what an interviewer will probe:
- Approving the angle at the end of Stage 0.
- Writing and labelling the hand-written test set.
- Running the Kaggle training jobs.
- Reading the failure samples in Stage 3 and agreeing the categories.
- Writing the write-up in Stage 5.

## 12. Done means

- The claim in section 2 has a clear, evidenced answer.
- A reviewer can open one link, type a request and see ranked tools within seconds.
- A stranger can reproduce the headline table from a clean clone with documented commands.
- The README states what the model is bad at.
