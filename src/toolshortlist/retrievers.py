"""Retrievers. Each one has two methods:

    index(docs)                            docs: the tool texts, in a fixed order
    search(queries, k, instructions=None)  returns an (n_queries, k) array of doc
                                           positions, best first

A retriever formats its own input. `query_template` is used when the query is
given alone and `query_template_inst` when an instruction comes with it.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .tooltext import MODES, render_tool


def format_queries(queries, instructions, template, template_inst):
    if instructions is None:
        return [template.format(query=query) for query in queries]
    return [template_inst.format(instruction=instruction, query=query)
            for query, instruction in zip(queries, instructions)]


class BM25Retriever:
    """Lexical baseline: bm25s with English stop words and no stemming, as in the ToolRet paper."""

    def __init__(self, query_template="{query}", query_template_inst="{instruction} {query}"):
        self.query_template = query_template
        self.query_template_inst = query_template_inst

    def index(self, docs):
        import bm25s
        self.bm25 = bm25s.BM25()
        self.bm25.index(bm25s.tokenize(docs, stopwords="en", show_progress=False), show_progress=False)
        self.n_docs = len(docs)

    def search(self, queries, k, instructions=None):
        import bm25s
        texts = format_queries(queries, instructions, self.query_template, self.query_template_inst)
        tokens = bm25s.tokenize(texts, stopwords="en", show_progress=False)
        positions, _ = self.bm25.retrieve(tokens, k=min(k, self.n_docs), show_progress=False)
        return np.asarray(positions)


def pool(hidden, attention_mask, pooling: str):
    """One vector per text from the encoder's last hidden states: the [CLS] token, or the mean over tokens."""
    if pooling == "cls":
        return hidden[:, 0]
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)


def inference_device() -> tuple[str, str]:
    """(device, precision) used for encoding.

    On a GPU the model runs in half precision, as in the ToolRet evaluation code.
    It is about 1.5 times faster on an Apple GPU and changes the embeddings only
    in the sixth decimal. On a CPU it runs in full precision.
    """
    import torch
    if torch.backends.mps.is_available():
        return "mps", "fp16"
    if torch.cuda.is_available():
        return "cuda", "fp16"
    return "cpu", "fp32"


class DenseRetriever:
    """A bi-encoder: embed tools and queries with one model, rank by cosine similarity."""

    def __init__(self, model, revision=None, pooling="cls", max_length=512,
                 query_template="{query}", query_template_inst="{instruction} {query}",
                 doc_template="{doc}", text_mode="full", batch_size=32, cache_dir="cache"):
        assert pooling in ("cls", "mean")
        assert text_mode in MODES
        self.model_name, self.revision = model, revision
        self.pooling, self.max_length = pooling, max_length
        self.query_template, self.query_template_inst = query_template, query_template_inst
        self.doc_template = doc_template
        self.text_mode = text_mode  # how much of each tool document is embedded (tooltext.py)
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.model = None
        self.device, self.precision = inference_device()

    def _load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self.model = AutoModel.from_pretrained(self.model_name, revision=self.revision)
        self.model.to(self.device).eval()
        if self.precision == "fp16":
            self.model.half()

    def encode(self, texts):
        """L2-normalised embeddings, one row per text. Results are cached on disk."""
        digest = hashlib.sha256(json.dumps(
            [self.model_name, self.revision, self.pooling, self.max_length, self.precision, texts]).encode()).hexdigest()
        cache_file = self.cache_dir / f"{digest[:32]}.npy"
        if cache_file.exists():
            return np.load(cache_file)

        import torch
        if self.model is None:
            self._load()
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # similar lengths per batch
        embeddings = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        for start in range(0, len(texts), self.batch_size):
            batch = order[start:start + self.batch_size]
            inputs = self.tokenizer([texts[i] for i in batch], padding=True, truncation=True,
                                    max_length=self.max_length, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state.float()
                pooled = torch.nn.functional.normalize(
                    pool(hidden, inputs["attention_mask"], self.pooling), dim=1)
            embeddings[batch] = pooled.cpu().numpy()
            done = start + len(batch)
            if len(texts) >= 5000 and (done % (100 * self.batch_size) == 0 or done == len(texts)):
                print(f"  {self.model_name}: encoded {done}/{len(texts)}", file=sys.stderr, flush=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, embeddings)
        return embeddings

    def index(self, docs):
        self.doc_embeddings = self.encode(
            [self.doc_template.format(doc=render_tool(doc, self.text_mode)) for doc in docs])

    def search(self, queries, k, instructions=None):
        texts = format_queries(queries, instructions, self.query_template, self.query_template_inst)
        query_embeddings = self.encode(texts)
        k = min(k, len(self.doc_embeddings))
        positions = np.empty((len(texts), k), dtype=np.int64)
        for start in range(0, len(texts), 256):
            scores = query_embeddings[start:start + 256] @ self.doc_embeddings.T
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            top_scores = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_scores, axis=1, kind="stable")
            positions[start:start + 256] = np.take_along_axis(top, order, axis=1)
        return positions


def rrf_fuse(rankings, k, rrf_k=60):
    """Reciprocal rank fusion of several rankings of the same queries.

    A doc at rank r (counting from 1) in one ranking scores 1 / (rrf_k + r); its
    scores are added over the rankings. Ties go to the lower doc position.
    """
    n_queries = len(rankings[0])
    fused = []
    for q in range(n_queries):
        scores = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking[q], start=1):
                scores[int(doc)] = scores.get(int(doc), 0.0) + 1.0 / (rrf_k + rank)
        fused.append(sorted(scores, key=lambda doc: (-scores[doc], doc))[:k])
    return np.asarray(fused)


class HybridRetriever:
    """Fuses the top `depth` results of several retrievers with reciprocal rank fusion."""

    def __init__(self, retrievers, rrf_k=60, depth=100):
        self.retrievers, self.rrf_k, self.depth = retrievers, rrf_k, depth

    def index(self, docs):
        for retriever in self.retrievers:
            retriever.index(docs)

    def search(self, queries, k, instructions=None):
        assert k <= self.depth
        rankings = [retriever.search(queries, self.depth, instructions) for retriever in self.retrievers]
        return rrf_fuse(rankings, min(k, rankings[0].shape[1]), self.rrf_k)
