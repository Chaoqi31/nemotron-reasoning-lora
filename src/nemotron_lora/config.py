"""Training configuration for the Nemotron-H reasoning LoRA fine-tune."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Hyper-parameters for a single-epoch LoRA fine-tune of Nemotron-3-Nano-30B-A3B.

    Defaults reproduce the configuration used for the strongest submission.
    """

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj",
        "in_proj", "out_proj",
        "lm_head",
    )
    # Tie the LoRA factors across all 128 MoE experts so every expert shares one
    # low-rank update. Cuts trainable params and stabilises the single-epoch run.
    moe_tie_weights: bool = True

    # Optimisation
    max_seq_len: int = 8192
    num_steps: int = 1000          # clamped to one epoch over the corpus
    batch_size: int = 32
    micro_batch_size: int = 4
    learning_rate: float = 2e-4
    lr_end: float = 2e-5           # cosine decay target
    weight_decay: float = 0.01
    seed: int = 42

    # Data / model locations (Kaggle layout by default; override for local runs)
    repo_base: str = (
        "/kaggle/input/datasets/huikang/huikang-nemotron-repository-snapshot/nemotron-master"
    )
    model_handle: str = "metric/nemotron-3-nano-30b-a3b-bf16/transformers/default"
    output_dir: str = "."

    # Set these to point at a locally built corpus (see data_pipeline/). When
    # None, the Kaggle snapshot layout under ``repo_base`` is used.
    corpus_path_override: str | None = None
    train_order_path_override: str | None = None

    @property
    def corpus_path(self) -> str:
        return self.corpus_path_override or f"{self.repo_base}/training/sft/04-08-16-14/tokens"

    @property
    def train_order_path(self) -> str:
        return self.train_order_path_override or f"{self.repo_base}/training/sft/04-08-16-14/logprobs/index.jsonl"
