"""Load the pre-tokenized reasoning corpus into training examples.

The corpus is a directory of per-problem segment files. Each segment holds a
token sequence and a parallel loss mask (1 = supervised token, 0 = prompt /
ignored). Training order is taken from an index file so runs are reproducible.

Expected layout under ``corpus_path``::

    <corpus_path>/<problem_id>/synthetic.json   # {"tokens": [...], "mask": [...]}
    <train_order_path>                           # JSONL, one {"problem_id", "epoch"} per line
"""

from __future__ import annotations

import json
import os


def load_training_order(train_order_path: str) -> list[str]:
    """Return de-duplicated problem ids for the first epoch, in file order."""
    ordered_ids: list[str] = []
    seen: set[str] = set()
    with open(train_order_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("epoch", 0) != 0:
                continue
            pid = rec["problem_id"]
            if pid in seen:
                continue
            seen.add(pid)
            ordered_ids.append(pid)
    return ordered_ids


def load_examples(corpus_path: str, ordered_ids: list[str], max_seq_len: int) -> list[dict]:
    """Build next-token-prediction examples from the tokenized corpus.

    Each example exposes ``tokens`` (inputs), ``targets`` (shifted by one) and
    ``weights`` (the shifted loss mask). Sequences longer than ``max_seq_len``
    are truncated; fully-masked or empty sequences are dropped.
    """
    examples: list[dict] = []
    for sid in ordered_ids:
        seg_path = os.path.join(corpus_path, sid, "synthetic.json")
        assert os.path.isfile(seg_path), f"missing corpus segment for {sid}"
        with open(seg_path) as f:
            rec = json.load(f)
        tokens = rec["tokens"]
        mask = rec["mask"]
        if not tokens:
            continue
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
            mask = mask[:max_seq_len]
        if not any(mask):
            continue
        examples.append({
            "problem_id": sid,
            "tokens": tokens[:-1],
            "targets": tokens[1:],
            "weights": [float(m) for m in mask[1:]],
        })
    return examples
