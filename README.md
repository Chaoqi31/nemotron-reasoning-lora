# Nemotron Reasoning LoRA

Fine-tune **Nemotron-3-Nano-30B-A3B** — a 30B hybrid Mamba-2 + Mixture-of-Experts model —
with LoRA, on a single GPU, in one epoch. The run produces a small adapter packaged as
`submission.zip` for the
[NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge),
where this recipe scores around **0.86** on the private leaderboard.

## How it fits on one GPU

A 30B hybrid MoE model doesn't train on a single card by default. A few things make it work
(all in [`src/nemotron_lora/train.py`](src/nemotron_lora/train.py)):

- **One LoRA shared across all 128 experts.** The expert adapters are tied — mean-pooled
  init, summed gradients — instead of training 128 separate ones.
- **Cut Cross-Entropy.** The `lm_head` projection and the loss are fused, so the full-vocab
  logits never hit memory. This is what keeps the model in VRAM.
- **A hand-attached `lm_head` LoRA**, since Unsloth drops it for MoE models, plus the Mamba
  CUDA fast path and per-component precision (LoRA fp32, weights bf16, router fp32).
- **A gentle schedule** — cosine LR `2e-4 → 2e-5`, dropout `0.05`, weight decay `0.01` — so
  a single epoch learns the corpus without memorising it.

## Get started

The base model is pulled automatically via `kagglehub`. Run it as a package:

```python
from nemotron_lora import train, TrainConfig

train(TrainConfig())   # writes submission.zip
```

or from the shell with `python -m nemotron_lora`, or open
[`notebooks/train.ipynb`](notebooks/train.ipynb) — the same code inline, ready to run on
Kaggle with no repo imports.

> [!NOTE]
> You'll need a single GPU with ~90 GB of VRAM. The reference run used one RTX PRO 6000
> Blackwell (96 GB) and finished in about 4 hours. For a faithful reproduction, pin the
> CUDA kernels to `mamba-ssm==2.3.1` and `causal-conv1d==1.6.1`.

## The data

Training reads a pre-tokenized corpus — one file per problem holding a token sequence and a
loss mask (`1` = supervised, `0` = prompt) — plus an index that fixes the order:

```
tokens/<problem_id>/synthetic.json   # {"tokens": [...], "mask": [...]}
index.jsonl                          # {"problem_id": "...", "epoch": 0}
```

[`data_pipeline/`](data_pipeline/) builds this corpus end to end: deterministic solvers
turn each problem into a chain-of-thought trace, a set of augmenters expand the set, and
everything is tokenized with a loss mask. It expects the competition `train.csv`, the
base-model `tokenizer.json`, and a `problems.jsonl` rule index in the folder, then:

```bash
cd data_pipeline
python reasoning.py        # solver traces      -> reasoning/*.txt
python augmentation.py     # augmented examples -> augmentations/*.txt
python corpus.py           # tokenize + mask    -> corpus/<pid>/synthetic.jsonl
python export_tokens.py    #                    -> tokens/ + index.jsonl
```

Point the trainer at the output:

```python
TrainConfig(corpus_path_override="data_pipeline/tokens",
            train_order_path_override="data_pipeline/index.jsonl")
```

## Project layout

```
src/nemotron_lora/    # the trainer: config, data loading, training loop, adapter export
data_pipeline/        # corpus generation: solvers, augmenters, tokenization
notebooks/train.ipynb # the whole thing inline, Kaggle-ready
tests/                # self-checks for the data loaders
```
