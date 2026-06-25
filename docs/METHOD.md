# Method

A detailed walk-through of how this project turns the
[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge)
into a fine-tune that scores ~0.86 on the private leaderboard. For a quick overview see the
[README](../README.md); this document is the full story.

## 1. The problem

Each task is a **few-shot rule-induction puzzle**. The model is shown a handful of worked
examples that demonstrate some hidden transformation, then asked to apply that same rule to
a held-out query. The families:

| Family | Hidden rule |
|--------|-------------|
| `cipher` | a letter substitution map |
| `cryptarithm` | a digit↔symbol assignment plus an arithmetic operation |
| `bit_manipulation` | a per-bit boolean function over 8-bit inputs |
| `gravity` | a quadratic law `d = k · t²` |
| `unit_conversion` | a linear law `output = factor · input` |
| `numeral` | Arabic → Roman numeral conversion |
| `equation_numeric` | the operator hidden between two numbers |

The `_deduce` variants have a unique recoverable rule; the `_guess` variants are
under-determined (the examples don't pin down a single rule).

## 2. Pipeline overview

The core idea: don't hope the base model guesses the rule — **solve each problem with a
deterministic program, have that program narrate its reasoning, keep only the traces that
are provably correct, and fine-tune on them.**

![Method overview](method.svg)

## 3. Symbolic solvers recover the rule

[`data_pipeline/reasoners/`](../data_pipeline/reasoners/) has one solver per family. Each
searches the space of rules its family allows until it finds one consistent with **every**
provided example, then applies it to the query.

- **`cipher`** — substitution ciphers are cracked by *letter-pattern matching*. Each
  ciphertext word is reduced to a repetition pattern (e.g. `abba`), candidate plaintext
  words with the same pattern are pulled from a fixed word list (`wonderland.txt`), and the
  cipher→plain mapping is grown greedily, keeping it bijective and consistent across words.
- **`cryptarithm`** — two phases. First a fast check for concatenation / reverse-concat
  rules. Otherwise a search over arithmetic operations (`add`, `sub`, `mul`, `absdiff`,
  `cat`, `revcat`) and digit↔symbol assignments: brute force for ≤8 symbols, backtracking
  beyond, all under a per-problem timeout. A candidate must reproduce every example.
- **`bit_manipulation`** — the output is 8 bits, and each output column is explained
  independently by one boolean family (`I`, `NOT`, `0`, `1`, `AND`, `OR`, `XOR`, `AND-NOT`,
  `OR-NOT`, `XOR-NOT`) over the input columns. The solver finds, per column, the family that
  matches across all examples.
- **`gravity`, `unit_conversion`** — fit the single constant (`k` or `factor`) from the
  examples, then compute the query answer with explicit long multiplication / long division
  steps, truncated to three decimal places.
- **`numeral`** — apply the standard subtractive-notation value table (`M, CM, D, CD, …`).
- **`equation_numeric`** — recover the operator sitting between the two operands by testing
  the candidates that reproduce the worked examples.

## 4. Reasoning trace and loss masking

A solver doesn't just emit the answer — it writes a natural-language chain-of-thought that
mirrors *how* it found the rule, ending in a boxed answer. The completion is:

```
<reasoning that mirrors the solver>
</think>
\boxed{ANSWER}<|im_end|>
```

The prompt is the problem rendered through the model's chat template with
`add_generation_prompt=True`, which already emits the opening `<think>\n`. The training
example is `prompt_tokens + completion_tokens` with a loss mask of `0` over the prompt and
`1` over the completion:

```
tokens:  [ prompt .............. ][ reasoning ... </think> \boxed{ans} <|im_end|> ]
mask:    [ 0 0 0 0 0 0 0 0 0 0 0 ][ 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ]
```

So the model is never scored on the question — only on producing the derivation and the
answer. It learns to *reproduce the procedure*, not to memorise answers.

## 5. Verification and data selection

This is the quality gate. After a solver produces a trace, the pipeline extracts the
`\boxed{}` value and compares it to the ground-truth answer:

- **match → `rule_found`**: the trace is a correct derivation and is written to the corpus.
- **no match / solver gave up → `rule_unknown`**: dropped.
- problems with a partial investigation but no verified rule → `hypothesis_formed` (also not
  used as supervised data).

Of ~9,500 problems, about **8,400 reach `rule_found`**. The `_guess` categories are kept
even without a unique rule, so the model learns to attempt a plausible answer rather than
stall on the under-determined cases.

This generate-and-verify loop is what makes the corpus trustworthy: every supervised token
sequence ends in a provably correct answer.

## 6. Sub-skill augmentation

[`data_pipeline/augmenters/`](../data_pipeline/augmenters/) synthesises extra examples that
drill the smaller string-manipulation skills the puzzles are built from. These are pure
format transforms (hundreds of problems each, ~100 rows apiece):

| Augmenter | Skill drilled |
|-----------|---------------|
| `spelling` | break text into individual characters (`–s–e–x–v–e–x–`) |
| `splitting` | split one bracket into per-symbol brackets (`【]}@]】` → `【]】【}】【@】【]】`) |
| `concatenation` | the inverse — merge per-symbol brackets into one |
| `lstrip` | strip leading whitespace inside a bracket (`【   $%^】` → `【$%^】`) |
| `matching` | bit-column matching: align input/output columns to operation sections |

These teach the tokenizer-level operations (splitting, merging, per-character handling) that
the cipher and bit tasks rely on, so the model is fluent in the mechanics before it has to
chain them.

## 7. Corpus tokenization

![Data pipeline](pipeline.svg)

[`corpus.py`](../data_pipeline/corpus.py) assembles each verified trace into the final
training example:

1. Tokenize the prompt through the chat template and the completion separately.
2. Concatenate, build the `0/1` loss mask, and truncate to `TOKEN_LIMIT = 8192`.
3. Write the tokens + mask as interleaved masked/unmasked segments to
   `corpus/<problem_id>/synthetic.jsonl`, with a row in `corpus.jsonl`.

[`export_tokens.py`](../data_pipeline/export_tokens.py) then flattens those segments back
into the layout the trainer reads:

```
tokens/<problem_id>/synthetic.json   # {"tokens": [...], "mask": [...]}
index.jsonl                          # {"problem_id": "...", "epoch": 0}
```

`index.jsonl` fixes the training order; the trainer de-duplicates it and shuffles once with
a fixed seed, so a run is reproducible.

## 8. Training

The trainer ([`src/nemotron_lora/train.py`](../src/nemotron_lora/train.py)) is a plain
manual loop — no `Trainer` abstraction — so every step is explicit.

![Forward and loss](training.svg)

**Base model & LoRA.** `Nemotron-3-Nano-30B-A3B` is loaded in bf16 (no quantisation). LoRA
(rank 32, α 32, dropout 0.05) is attached to attention (`q/k/v/o_proj`), the Mamba
projections (`in_proj/out_proj`), the MoE expert projections (`up/down_proj`) and `lm_head`.

**MoE expert tying.** The model has 128 experts. Training a distinct adapter per expert is
both expensive and unstable for a single epoch, so the expert LoRA factors are *tied*:

- at init, each tied factor is set to the **mean** across the expert dimension, and
- before every optimizer step, their gradients are replaced by the **sum** across experts.

The whole expert bank therefore moves as one shared low-rank update.

**Cut Cross-Entropy forward.** A custom forward runs the backbone, then computes the loss
with a fused `linear_cross_entropy(hidden, lm_weight, labels)` — where `lm_weight` folds in
the `lm_head` LoRA (`base + scaling · Bᵀ A`). Fusing the projection and the softmax-CE means
the `[batch · seq · vocab]` logit tensor is **never materialised**. The kernel returns
per-token CE, which the loop multiplies by the loss mask and averages
(`Σ(ce · w) / Σ w`).

**`lm_head` LoRA fix.** Unsloth omits the `lm_head` adapter for MoE models, so it is added
by hand. On save, the key prefix `base_model.model.lm_head.` is rewritten to
`base_model.model.backbone.lm_head.` to match the inference-time module tree.

**Precision.** LoRA factors fp32, base weights bf16, MoE router fp32. The Mamba CUDA fast
path (`is_fast_path_available`) is force-enabled for the state-space layers.

**Optimizer & schedule.** AdamW (`betas=(0.9, 0.95)`, `eps=1e-8`, `weight_decay=0.01`),
effective batch 32 via micro-batches of 4 with gradient accumulation. A cosine schedule
decays the LR from `2e-4` to `2e-5` across the run; gradients are tied (summed across
experts) before each step.

**One epoch on purpose.** The corpus is near-deterministic. A second pass drives train loss
toward zero and the leaderboard score *down* — the model starts memorising specific traces.
A single epoch (~250 steps over the corpus) with the mild dropout / weight-decay above lands
at the best score.

## 9. Why it fits on one GPU

Two costs would normally break a 30B fine-tune on a single 96 GB card, and each is removed:

- **Optimizer & adapter footprint** — LoRA only (base frozen), and the 128 experts share one
  tied adapter, so trainable parameters and their AdamW states stay small.
- **Logit memory** — Cut Cross-Entropy never builds the full-vocabulary logits, which is the
  spike that would otherwise dominate at this vocabulary size and sequence length.

The reference run used one **RTX PRO 6000 Blackwell (96 GB)** and finished in **~4 hours**.

## 10. Adapter export

After training, the adapter is saved, its `lm_head` key prefix is rewritten, and the
`adapter_*` files are zipped into `submission.zip` — the artefact the competition scores.
No base weights are shipped; the adapter is a few hundred MB.

## 11. Reproducibility

- **From the public corpus snapshot** — feeding the trainer the exact token/index files the
  winning run used reproduces the result, up to GPU-kernel non-determinism, provided the
  CUDA kernels are pinned (`mamba-ssm==2.3.1`, `causal-conv1d==1.6.1`).
- **From a fresh `data_pipeline/` rebuild** — regenerates an *equivalent* corpus, but example
  ordering and inclusion may not be byte-identical, so the resulting adapter can differ
  slightly. This reproduces the **method**, not necessarily the exact adapter.
