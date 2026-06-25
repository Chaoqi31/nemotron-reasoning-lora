# Attribution

The data-generation code in this directory is authored by **Huikang Tong** and
copied from [github.com/tonghuikang/nemotron](https://github.com/tonghuikang/nemotron).
Each file carries an author header pointing back to that repository.

Files from the original repository (unmodified except for the author header):

- `reasoning.py`, `corpus.py`, `augmentation.py`
- `reasoners/` — `bit_manipulation.py`, `cipher.py`, `cryptarithm.py`,
  `equation_numeric.py`, `gravity.py`, `numeral.py`, `unit_conversion.py`,
  `store_types.py`, `wonderland.txt`
- `augmenters/` — `__init__.py`, `concatenation.py`, `lstrip.py`, `matching.py`,
  `spelling.py`, `splitting.py`
- `investigators/priority_problem_ids.txt`

Added by this repository:

- `export_tokens.py` — converts the `corpus.py` segment output into the flat
  `tokens/<pid>/synthetic.json` + `index.jsonl` layout the trainer consumes.

The `nemotron_lora` training code under `src/` is original to this repository.
