"""Self-check for segment flattening. Run: python tests/test_export_tokens.py"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from export_tokens import flatten_segments


def main() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"type": "masked", "pos": 0, "tokens": [10, 11]}) + "\n")
        f.write(json.dumps({"type": "unmasked", "pos": 2, "tokens": [12, 13, 14]}) + "\n")
        path = f.name

    tokens, mask = flatten_segments(path)
    os.unlink(path)
    assert tokens == [10, 11, 12, 13, 14], tokens
    assert mask == [0, 0, 1, 1, 1], mask
    print("ok")


if __name__ == "__main__":
    main()
