"""Self-check for corpus loading. Run: python tests/test_data.py"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nemotron_lora.data import load_examples, load_training_order


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        order_path = os.path.join(d, "index.jsonl")
        with open(order_path, "w") as f:
            f.write(json.dumps({"problem_id": "a", "epoch": 0}) + "\n")
            f.write(json.dumps({"problem_id": "a", "epoch": 0}) + "\n")  # dup dropped
            f.write(json.dumps({"problem_id": "b", "epoch": 0}) + "\n")
            f.write(json.dumps({"problem_id": "c", "epoch": 1}) + "\n")  # epoch != 0 dropped

        order = load_training_order(order_path)
        assert order == ["a", "b"], order

        corpus = os.path.join(d, "tokens")
        _write(os.path.join(corpus, "a", "synthetic.json"), {"tokens": [1, 2, 3], "mask": [0, 1, 1]})
        _write(os.path.join(corpus, "b", "synthetic.json"), {"tokens": [4, 5], "mask": [0, 0]})  # all-masked dropped

        ex = load_examples(corpus, order, max_seq_len=8192)
        assert len(ex) == 1, ex
        assert ex[0]["tokens"] == [1, 2] and ex[0]["targets"] == [2, 3], ex[0]
        assert ex[0]["weights"] == [1.0, 1.0], ex[0]

        long = load_examples(corpus, ["a"], max_seq_len=2)  # truncation
        assert long[0]["tokens"] == [1], long[0]

    print("ok")


if __name__ == "__main__":
    main()
