#!/bin/sh
# Stage 1 from start to finish: tests, overlap audit, every baseline, tables.
# Run from the repository root: sh scripts/stage1.sh
set -eu

uv run pytest -q
uv run python -m toolshortlist.audit

for run in \
    bm25 \
    all-MiniLM-L6-v2 bge-small-en-v1.5 bge-base-en-v1.5 \
    rrf-bm25-all-MiniLM-L6-v2 rrf-bm25-bge-small-en-v1.5 rrf-bm25-bge-base-en-v1.5 \
    paper-all-MiniLM-L6-v2 paper-codeprefix-all-MiniLM-L6-v2 \
    paper-bge-base-en-v1.5 paper-codeprefix-bge-base-en-v1.5 \
    paper-toolret-trained-bge-base paper-codeprefix-toolret-trained-bge-base
do
    uv run python -m toolshortlist.evaluate "configs/$run.toml"
done

uv run python -m toolshortlist.report
