"""Convert ``corpus.py`` output into the flat token layout the trainer reads.

``corpus.py`` writes ``corpus/<pid>/synthetic.jsonl`` as interleaved
masked/unmasked segments plus a ``corpus.jsonl`` index. The trainer
(``nemotron_lora``) expects, per included problem::

    tokens/<pid>/synthetic.json   # {"tokens": [...], "mask": [...]}
    index.jsonl                   # {"problem_id": "...", "epoch": 0}

Run from this directory after ``corpus.py``::

    python export_tokens.py            # reads ./corpus + ./corpus.jsonl, writes ./tokens + ./index.jsonl
"""

from __future__ import annotations

import json
import os

CORPUS_DIR = "corpus"
CORPUS_INDEX = "corpus.jsonl"
TOKENS_DIR = "tokens"
INDEX_OUT = "index.jsonl"


def flatten_segments(segment_path: str) -> tuple[list[int], list[int]]:
    """Rebuild flat ``(tokens, mask)`` from a segment file (mask 1 = unmasked)."""
    tokens: list[int] = []
    mask: list[int] = []
    with open(segment_path) as f:
        for line in f:
            seg = json.loads(line)
            seg_tokens = seg["tokens"]
            tokens.extend(seg_tokens)
            mask.extend([1 if seg["type"] == "unmasked" else 0] * len(seg_tokens))
    return tokens, mask


def main() -> None:
    os.makedirs(TOKENS_DIR, exist_ok=True)
    n = 0
    with open(CORPUS_INDEX) as idx, open(INDEX_OUT, "w") as out:
        for line in idx:
            entry = json.loads(line)
            if not entry.get("included", False):
                continue
            pid = entry["problem_id"]
            seg_path = os.path.join(CORPUS_DIR, pid, "synthetic.jsonl")
            if not os.path.isfile(seg_path):
                continue
            tokens, mask = flatten_segments(seg_path)
            out_dir = os.path.join(TOKENS_DIR, pid)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "synthetic.json"), "w") as f:
                json.dump({"tokens": tokens, "mask": mask}, f)
            out.write(json.dumps({"problem_id": pid, "epoch": 0}) + "\n")
            n += 1
    print(f"Exported {n} problems to {TOKENS_DIR}/ and {INDEX_OUT}")


if __name__ == "__main__":
    main()
