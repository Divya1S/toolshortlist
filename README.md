# ToolShortlist

A small, trained retrieval model that shortlists the right tools for an AI agent out of a large catalog.

Status: Stage 2 (training) is in progress: the training code and the Kaggle notebook are ready, and the training runs are being done on Kaggle. The baseline results are in [results/baselines.md](results/baselines.md), the overlap audit of the benchmark in [results/data_audit.md](results/data_audit.md). See [PROJECT.md](PROJECT.md) for the plan and [DATA.md](DATA.md) for the datasets.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and about 2 GB of free disk for the datasets and models, which are downloaded from the Hugging Face Hub on first use.

```sh
uv sync --frozen        # install the pinned dependencies
sh scripts/stage1.sh    # tests, overlap audit, every baseline, tables
```

The script writes one JSON file per retriever to `results/` and renders the tables in [results/baselines.md](results/baselines.md) and [results/data_audit.md](results/data_audit.md) from them. The first run took about an hour on an M3 MacBook, almost all of it embedding the 44,453 ToolRet tools with each model on the GPU; embeddings are cached under `cache/`, so later runs take minutes. To run one retriever: `uv run python -m toolshortlist.evaluate configs/bm25.toml`.

## Train (Stage 2)

Training runs on a free Kaggle GPU, because the development machine has none. One run of [notebooks/train_kaggle.ipynb](notebooks/train_kaggle.ipynb) trains one config from `configs/train/` with one seed, evaluates the checkpoint exactly as the baselines were evaluated, and leaves a results file to copy into `results/`. The notebook clones this repository, so it must be on a public GitHub repository; set `REPO_URL` in its first code cell.

The training script itself is [src/toolshortlist/train.py](src/toolshortlist/train.py): plain PyTorch, one InfoNCE loss, in-batch negatives plus hard negatives mined by an untrained Stage 1 baseline from the training set only. A smoke test on a laptop:

```sh
uv run python -m toolshortlist.train configs/train/default.toml --limit 1500 --steps 60
uv run python -m toolshortlist.evaluate runs/train-default-s0/eval.toml --out-dir /tmp/smoke
```

## How this differs from prior work

Training a retriever for tool selection is not new. The ToolRet authors published checkpoints of 109M-parameter models fine-tuned on their training set ([paper](https://arxiv.org/abs/2503.01763), [checkpoints](https://huggingface.co/mangopy)); Tool-DE fine-tunes 0.6B and 4B embedders on ToolRet with expanded tool documentation ([paper](https://arxiv.org/abs/2510.22670)); [`toolrank`](https://github.com/yasinyaman/toolrank) trains an 8B embedder on ToolRet's training pairs, tests it on ToolRet and LiveMCPBench, and serves it over MCP from a GPU; and StackOne reports a fine-tuned 109M model on ToolRet without releasing weights ([post](https://www.stackone.com/blog/autoresearch-charged-action-search/)). At the small end, [ToolScout](https://huggingface.co/nuhmanpk/toolscout-bge-small) fine-tunes a 33M model on 407 examples and tests it on 36 tools. This project differs in three ways. First, it goes down to models of 22M and 33M parameters, trains them on ToolRet's training set, and ships the result as an int8 ONNX package that runs on a CPU and in a browser. Second, it audits the benchmark before trusting it: ToolRet's training set shares verbatim queries and tools with three of its 35 evaluation tasks, an overlap the ToolRet paper does not discuss, so results are reported separately for tool sources the model saw in training and sources it did not (counts in [DATA.md](DATA.md)). Third, it tests on real MCP tool catalogs that appear in no training data, and publishes a failure analysis of what the model still gets wrong there.
