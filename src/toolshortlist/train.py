"""Train a bi-encoder for tool retrieval with an InfoNCE loss.

One run is one config file in configs/train/. In order:

  1. load      ToolRet's training set, minus the rows whose query duplicates an
               evaluation query (results/train_exclude_ids.json).
  2. split     a validation set, held out by tool: a row is validation only if
               every one of its gold tools is held out, so no validation tool is
               ever a training positive.
  3. mine      hard negatives: for each training query, tools that an untrained
               Stage 1 baseline ranks highly but that are not its positives.
  4. train     one pass over the training rows. Each step embeds a batch of
               queries and tools with the same encoder, scores every query
               against every tool in the batch, and asks cross-entropy to put
               each query's own positive first.
  5. validate  nDCG@10 on the validation queries every `validate_every` steps;
               the best checkpoint is kept.
  6. save      the model, its tokenizer, a log, and an evaluation config that
               evaluate.py can run.

Run: uv run python -m toolshortlist.train configs/train/default.toml [--seed 0]
     Smoke test: add --limit 2000 --steps 60 to train on 2,000 rows for 60 steps.
"""

import argparse
import json
import platform
import random
import time
import tomllib
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .audit import parse_tool
from .data import REVISIONS, iter_toolret_train
from .metrics import score_query
from .retrievers import DenseRetriever, inference_device, pool
from .tooltext import render_tool

# Fixed for every run, so all configs share one validation set and one set of negatives.
SPLIT_SEED = 0
VALIDATION_TOOLS = 500  # gold tools held out of training
VALIDATION_QUERIES = 1000  # validation rows kept (the rest of the eligible rows are dropped)
VALIDATION_CORPUS_EXTRA = 10000  # other training tools added to the validation corpus as distractors
MINER = {"model": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
         "pooling": "cls", "max_length": 256,
         "query_template": "Represent this sentence for searching relevant passages: {query}"}
MINE_RANKS = (4, 30)  # hard negatives come from ranks 5 to 30 of the miner's ranking (0-based 4..29)
NEGATIVES_PER_QUERY = 8


@dataclass
class Row:
    id: str
    query: str
    positives: list[int]  # positions in the tool table
    hard_negatives: list[int] = field(default_factory=list)


def load_rows(exclude_file="results/train_exclude_ids.json", limit=None, seed=SPLIT_SEED):
    """Training rows and the table of distinct tools they mention."""
    exclude = set(json.loads(Path(exclude_file).read_text()))
    raw = [(row_id, query, positives, negatives)
           for row_id, query, positives, negatives in iter_toolret_train() if row_id not in exclude]
    if limit:
        random.Random(seed).shuffle(raw)
        raw = raw[:limit]
    tools, position = [], {}
    def tool_id(document):
        if document not in position:
            position[document] = len(tools)
            tools.append(document)
        return position[document]
    rows = []
    for row_id, query, positives, negatives in raw:
        for document in negatives:  # the released negatives join the tool table but are not used as such
            tool_id(document)
        rows.append(Row(row_id, query, sorted({tool_id(p) for p in positives})))
    return rows, tools


def split(rows, tools, seed=SPLIT_SEED):
    """Hold out tools, then rows: validation rows have only held-out gold tools, training rows none.

    Returns (train_rows, validation_rows, validation_corpus, dropped) where dropped counts the
    rows with a mix of held-out and other gold tools, which belong to neither side.
    """
    rng = random.Random(seed)
    gold = sorted({p for row in rows for p in row.positives})
    held = set(rng.sample(gold, min(VALIDATION_TOOLS, len(gold) // 10)))
    train, validation, dropped = [], [], 0
    for row in rows:
        inside = sum(p in held for p in row.positives)
        if inside == 0:
            train.append(row)
        elif inside == len(row.positives):
            validation.append(row)
        else:
            dropped += 1
    rng.shuffle(validation)
    validation = validation[:VALIDATION_QUERIES]
    others = [t for t in range(len(tools)) if t not in held]
    corpus = sorted(held) + sorted(rng.sample(others, min(VALIDATION_CORPUS_EXTRA, len(others))))
    return train, validation, corpus, dropped


def mine_hard_negatives(rows, tools, cache_dir="cache"):
    """Fill row.hard_negatives with tools the miner ranks highly that are not the row's positives.

    Tools with the same name and description as a positive are skipped too: ToolRet's corpus
    holds duplicates under different ids, and they would be false negatives.
    """
    miner = DenseRetriever(cache_dir=cache_dir, batch_size=64, **MINER)
    miner.index(tools)
    ranked = miner.search([row.query for row in rows], MINE_RANKS[1])
    keys = [parse_tool(document)[1:] for document in tools]  # (name, description), normalised
    for row, candidates in zip(rows, ranked):
        row.hard_negatives = pick_negatives(candidates, row.positives, keys)


def pick_negatives(candidates, positives, keys):
    """Up to NEGATIVES_PER_QUERY tools from the miner's ranks MINE_RANKS that are not positives.

    `keys[t]` is the (name, description) of tool t; a candidate that shares its key with a
    positive is skipped as a probable duplicate.
    """
    taken = set(positives) | {keys[p] for p in positives}
    return [int(t) for t in candidates[MINE_RANKS[0]:MINE_RANKS[1]]
            if t not in positives and keys[t] not in taken][:NEGATIVES_PER_QUERY]


def embed(model, tokenizer, texts, max_length, pooling, device):
    """L2-normalised embeddings with gradients, for a batch of texts."""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                       return_tensors="pt").to(device)
    hidden = model(**inputs).last_hidden_state
    return torch.nn.functional.normalize(pool(hidden, inputs["attention_mask"], pooling).float(), dim=1)


def infonce_loss(query_embeddings, doc_embeddings, positive_mask, temperature):
    """Cross-entropy over the batch's tools: query i must score tool i highest.

    positive_mask[i, j] is True where tool j is a gold tool of query i. Tool i is
    query i's own positive; any other positive of query i is masked out so it is
    not penalised as a negative.
    """
    scores = query_embeddings @ doc_embeddings.T / temperature
    targets = torch.arange(len(query_embeddings), device=scores.device)
    false_negatives = positive_mask.clone()
    false_negatives[targets, targets] = False
    scores = scores.masked_fill(false_negatives, float("-inf"))
    return torch.nn.functional.cross_entropy(scores, targets)


@torch.no_grad()
def validate(model, tokenizer, cfg, rows, tools, corpus, device):
    """Mean nDCG@10 and Recall@10 of the current model on the validation queries."""
    model.eval()
    texts = [render_tool(tools[t], cfg["data"]["text_mode"]) for t in corpus]
    doc_embeddings = torch.cat([
        embed(model, tokenizer, texts[i:i + 128], cfg["training"]["doc_max_length"], cfg["model"]["pooling"], device)
        for i in range(0, len(texts), 128)])
    queries = [cfg["model"]["query_template"].format(query=row.query) for row in rows]
    query_embeddings = torch.cat([
        embed(model, tokenizer, queries[i:i + 256], cfg["training"]["query_max_length"], cfg["model"]["pooling"], device)
        for i in range(0, len(queries), 256)])
    top = (query_embeddings @ doc_embeddings.T).topk(10, dim=1).indices.cpu().numpy()
    scores = [score_query([str(corpus[t]) for t in ranked], {str(p): 1 for p in row.positives})
              for row, ranked in zip(rows, top)]
    model.train()
    return {metric: float(np.mean([s[metric] for s in scores])) for metric in ("ndcg@10", "recall@10")}


def train(cfg, rows, validation, tools, corpus, seed, out_dir, max_steps=None):
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    device = inference_device()[0]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base"], revision=cfg["model"].get("revision"))
    model = AutoModel.from_pretrained(cfg["model"]["base"], revision=cfg["model"].get("revision")).to(device)
    if cfg["training"].get("gradient_checkpointing"):  # trades compute for memory; the maths is unchanged
        model.gradient_checkpointing_enable()
    model.train()

    tr = cfg["training"]
    rng = random.Random(seed)
    order = list(range(len(rows)))
    batches = []
    for _ in range(tr["epochs"]):
        rng.shuffle(order)
        batches += [order[i:i + tr["batch_size"]] for i in range(0, len(order), tr["batch_size"])]
    batches = [b for b in batches if len(b) == tr["batch_size"]]
    if max_steps:
        batches = batches[:max_steps]
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(tr["warmup"] * len(batches)), len(batches))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")  # mixed precision on CUDA only
    use_hard = cfg["data"]["hard_negatives"]

    log, best, best_state = [], None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for step, batch in enumerate(batches, start=1):
        batch_rows = [rows[i] for i in batch]
        positives = [rng.choice(row.positives) for row in batch_rows]
        docs = list(positives)
        if use_hard:
            docs += [rng.choice(row.hard_negatives) if row.hard_negatives else rng.choice(positives)
                     for row in batch_rows]
        queries = [cfg["model"]["query_template"].format(query=row.query) for row in batch_rows]
        texts = [render_tool(tools[t], cfg["data"]["text_mode"]) for t in docs]
        positive_mask = torch.tensor([[t in row.positives for t in docs] for row in batch_rows], device=device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            q = embed(model, tokenizer, queries, tr["query_max_length"], cfg["model"]["pooling"], device)
            d = embed(model, tokenizer, texts, tr["doc_max_length"], cfg["model"]["pooling"], device)
            loss = infonce_loss(q, d, positive_mask, tr["temperature"])
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        entry = {"step": step, "loss": round(loss.item(), 4), "lr": scheduler.get_last_lr()[0],
                 "seconds": round(time.time() - started, 1)}
        if step % tr["validate_every"] == 0 or step == len(batches):
            entry["validation"] = validate(model, tokenizer, cfg, validation, tools, corpus, device)
            if best is None or entry["validation"]["ndcg@10"] > best["validation"]["ndcg@10"]:
                best = entry
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"step {step}/{len(batches)} loss {entry['loss']} validation {entry['validation']}", flush=True)
        elif step % 50 == 0:
            print(f"step {step}/{len(batches)} loss {entry['loss']}", flush=True)
        log.append(entry)
    model.load_state_dict(best_state)
    (out_dir / "log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
    return model, tokenizer, best, len(batches)


def write_eval_config(cfg, run_name, seed, model_dir: Path, out_dir: Path):
    """An evaluate.py config that scores this checkpoint the same way the baselines were scored."""
    template = cfg["model"]["query_template"]
    lines = [
        f'name = "{run_name}"',
        f'description = "Trained in Stage 2: {cfg["description"]}"',
        "", "[train]", f'config = "{cfg["name"]}"', f"seed = {seed}", f'base = "{cfg["model"]["base"]}"',
        f'text_mode = "{cfg["data"]["text_mode"]}"', f'hard_negatives = {str(cfg["data"]["hard_negatives"]).lower()}',
        "", "[retriever]", 'type = "dense"', f'model = "{model_dir.resolve()}"',
        f'pooling = "{cfg["model"]["pooling"]}"', f'max_length = {cfg["model"]["eval_max_length"]}',
        f'query_template = {json.dumps(template)}',
        f'query_template_inst = {json.dumps(template.replace("{query}", "{instruction} {query}"))}',
        f'text_mode = "{cfg["data"]["text_mode"]}"', "",
    ]
    (out_dir / "eval.toml").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs", help="directory that receives runs/<name>/")
    parser.add_argument("--limit", type=int, help="train on this many rows only (smoke tests)")
    parser.add_argument("--steps", type=int, help="stop after this many steps (smoke tests)")
    parser.add_argument("--batch-size", type=int, help="override the config's batch size (smoke tests)")
    args = parser.parse_args()
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    run_name = f"train-{cfg['name']}-s{args.seed}"
    out_dir = Path(args.out) / run_name
    started = time.time()

    rows, tools = load_rows(limit=args.limit)
    train_rows, validation, corpus, dropped = split(rows, tools)
    print(f"{len(rows)} rows after exclusion: {len(train_rows)} train, {len(validation)} validation, "
          f"{dropped} dropped; {len(tools)} distinct tools; validation corpus {len(corpus)}", flush=True)
    if cfg["data"]["hard_negatives"]:
        mine_hard_negatives(train_rows, tools)
        print(f"hard negatives mined with {MINER['model']}: "
              f"{sum(bool(r.hard_negatives) for r in train_rows)} of {len(train_rows)} rows have some", flush=True)

    model, tokenizer, best, steps = train(cfg, train_rows, validation, tools, corpus, args.seed, out_dir, args.steps)
    model_dir = out_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    write_eval_config(cfg, run_name, args.seed, model_dir, out_dir)
    summary = {
        "run": run_name, "config": cfg, "seed": args.seed, "steps": steps,
        "data": {"rows_after_exclusion": len(rows), "train_rows": len(train_rows),
                 "validation_rows": len(validation), "dropped_rows": dropped, "tools": len(tools),
                 "validation_corpus": len(corpus), "limit": args.limit,
                 "miner": MINER["model"] if cfg["data"]["hard_negatives"] else None,
                 "mine_ranks": MINE_RANKS, "negatives_per_query": NEGATIVES_PER_QUERY},
        "best": best, "minutes": round((time.time() - started) / 60, 1),
        "environment": {"python": platform.python_version(), "machine": platform.machine(),
                        "device": inference_device()[0],
                        **{p: version(p) for p in ("torch", "transformers", "numpy")}},
        "dataset_revisions": REVISIONS,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"saved {model_dir}; best validation at step {best['step']}: {best['validation']}; "
          f"evaluate with: python -m toolshortlist.evaluate {out_dir / 'eval.toml'}", flush=True)


if __name__ == "__main__":
    main()
