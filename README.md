# Nemotron Reasoning LoRA

Fine-tune **Nemotron-3-Nano-30B-A3B** — a 30B hybrid Mamba-2 + Mixture-of-Experts model —
with LoRA, on a single GPU, in one epoch. The run produces a small adapter packaged as
`submission.zip` for the
[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge),
where this recipe scores around **0.86** on the private leaderboard.

## How it works

The tasks are few-shot rule-induction puzzles — ciphers, cryptarithms, bit manipulation,
unit conversions and more. Instead of hoping the base model guesses each hidden rule, a
deterministic solver cracks every problem, narrates *how* it did it as a chain-of-thought,
and only the traces whose answer is provably correct become training data. The model is
fine-tuned to reproduce that reasoning.

![Method overview](docs/method.svg)

Fitting a 30B hybrid MoE model on one GPU takes a little engineering:

- **One tied LoRA for 128 experts** — rather than a separate adapter per expert, the expert
  LoRA factors are shared (mean-pooled init, summed gradients). That keeps the trainable
  parameter count small and the single epoch stable.
- **Cut Cross-Entropy** — the `lm_head` projection and the cross-entropy are fused into one
  kernel, so the full-vocabulary logits (the dominant memory spike at this vocab size) are
  never materialised.

Together they keep peak memory inside a 96 GB card.

> [!TIP]
> The full write-up — per-solver algorithms, the verify-and-filter loop, tokenization, the
> training loop, and the single-GPU memory work — is in **[docs/METHOD.md](docs/METHOD.md)**.

## Quickstart

Build the corpus (see [docs/METHOD.md](docs/METHOD.md#7-corpus-tokenization) for inputs):

```bash
cd data_pipeline
python reasoning.py && python augmentation.py && python corpus.py && python export_tokens.py
```

Then train — as a package, from the shell, or via the self-contained
[`notebooks/train.ipynb`](notebooks/train.ipynb):

```python
from nemotron_lora import train, TrainConfig

train(TrainConfig(corpus_path_override="data_pipeline/tokens",
                  train_order_path_override="data_pipeline/index.jsonl"))
```

The base model is pulled automatically via `kagglehub`; the run writes `submission.zip`.

> [!NOTE]
> You'll need a single GPU with ~90 GB of VRAM. The reference run used one RTX PRO 6000
> Blackwell (96 GB) and finished in about 4 hours. For a faithful reproduction, pin the
> CUDA kernels to `mamba-ssm==2.3.1` and `causal-conv1d==1.6.1`.

## Project layout

```
src/nemotron_lora/    # the trainer: config, data loading, training loop, adapter export
data_pipeline/        # corpus generation: solvers, augmenters, tokenization
notebooks/train.ipynb # the whole thing inline, Kaggle-ready
docs/METHOD.md        # the detailed method write-up
tests/                # self-checks for the data loaders
```
