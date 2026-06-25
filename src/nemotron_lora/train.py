"""Single-epoch LoRA fine-tune of Nemotron-3-Nano-30B-A3B (hybrid Mamba + MoE).

Pipeline: load the tokenized corpus, attach LoRA (including the MoE experts and
``lm_head``), patch the forward pass with Cut Cross-Entropy, and run a manual
gradient-accumulation loop with a cosine learning-rate schedule. The result is a
LoRA adapter packaged as ``submission.zip``.
"""

from __future__ import annotations

import gc
import math
import os
import random
import sys
import time

from .config import TrainConfig
from .data import load_examples, load_training_order


def _build_model(cfg: TrainConfig, model_path: str):
    import torch
    from unsloth import FastLanguageModel
    from peft import LoraConfig
    from peft.tuners.lora import Linear as LoraLinear

    model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg.max_seq_len,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        trust_remote_code=True,
        unsloth_force_compile=True,
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        target_modules=list(cfg.target_modules),
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    FastLanguageModel.for_training(model)

    # Enable the Mamba CUDA fast path (disabled by default under training).
    nemotron_mod = None
    for _name, _m in sys.modules.items():
        if "modeling_nemotron_h" in _name and hasattr(_m, "is_fast_path_available"):
            nemotron_mod = _m
            break
    assert nemotron_mod is not None, "modeling_nemotron_h not found"
    nemotron_mod.is_fast_path_available = True  # type: ignore[attr-defined]

    # Unsloth skips lm_head LoRA for MoE models; add it back by hand.
    causal_lm = model
    while hasattr(causal_lm, "model"):
        causal_lm = causal_lm.model
    lm_head = causal_lm.lm_head
    if not isinstance(lm_head, LoraLinear):
        lcfg = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout)
        model.base_model._create_and_replace(
            lcfg, "default", target=lm_head, target_name="lm_head", parent=causal_lm,
        )

    # LoRA factors in fp32; base weights stay bf16 (MoE router is already fp32).
    for name, param in model.named_parameters():
        if ".lora_" in name:
            param.data = param.data.to(torch.float32)
    for name, param in model.named_parameters():
        if ".lora_" in name:
            assert param.dtype == torch.float32, f"LoRA {name} expected fp32"
        elif ".mixer.gate." in name:
            assert param.dtype == torch.float32, f"router {name} expected fp32"
        else:
            assert param.dtype == torch.bfloat16, f"{name} expected bf16"
    return model


def _patch_forward_cce(model):
    """Replace the causal-LM forward with a Cut Cross-Entropy path.

    CCE fuses the lm_head projection and cross-entropy so the full-vocab logits
    are never materialised, which is what keeps the 30B model inside VRAM. The
    per-token loss is cached so the caller can apply the loss mask.
    """
    import torch
    from cut_cross_entropy import linear_cross_entropy

    base = model
    while hasattr(base, "model"):
        base = base.model

    def _forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
        backbone_out = base.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            **{k: v for k, v in kwargs.items() if k in ("position_ids", "past_key_values", "use_cache")},
        )
        hidden_states = backbone_out[0]
        lm_head = base.lm_head
        lm_weight = lm_head.base_layer.weight + lm_head.scaling["default"] * (
            lm_head.lora_B["default"].weight @ lm_head.lora_A["default"].weight
        )
        if labels is not None:
            model._cached_per_token_ce = linear_cross_entropy(  # type: ignore[attr-defined]
                hidden_states, lm_weight, labels, reduction="none",
            )
            return model._cached_per_token_ce.mean()
        model._cached_per_token_ce = None  # type: ignore[attr-defined]
        return None

    base.forward = _forward


def _moe_tied_params(model) -> list:
    """Collect the MoE expert LoRA factors that should share one update."""
    import torch

    w1_names = ("gate_up_proj", "up_proj", "gate_proj", ".w1.")
    w2_names = ("down_proj", ".w2.")
    tied: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad or ".experts." not in name or ".lora_" not in name:
            continue
        is_w1 = any(p in name for p in w1_names)
        is_w2 = any(p in name for p in w2_names)
        should_tie = (is_w1 and ".lora_A." in name) or (is_w2 and ".lora_B." in name)
        if not should_tie or param.dim() < 2 or param.shape[0] <= 1:
            continue
        tied.append(param)
    return tied


def train(cfg: TrainConfig | None = None) -> str:
    """Run the fine-tune and return the path to the written ``submission.zip``."""
    import kagglehub
    import torch

    cfg = cfg or TrainConfig()
    random.seed(cfg.seed)
    model_path = kagglehub.model_download(cfg.model_handle)

    ordered_ids = load_training_order(cfg.train_order_path)
    examples = load_examples(cfg.corpus_path, ordered_ids, cfg.max_seq_len)
    rng = random.Random(0)
    rng.shuffle(examples)
    total_tokens = sum(len(e["tokens"]) for e in examples)
    print(f"{len(examples)} examples, {total_tokens:,} tokens")

    gc.collect()
    torch.cuda.empty_cache()
    model = _build_model(cfg, model_path)
    _patch_forward_cce(model)

    tied = _moe_tied_params(model) if cfg.moe_tie_weights else []
    if tied:
        with torch.no_grad():
            for p in tied:
                p.data.copy_(p.data.mean(dim=0, keepdim=True).expand_as(p.data))
    print(f"MoE tied params: {len(tied)}")

    def _tie_grads() -> None:
        with torch.no_grad():
            for p in tied:
                if p.grad is not None:
                    p.grad.copy_(p.grad.sum(dim=0, keepdim=True).expand_as(p.grad))

    device = next(model.parameters()).device
    num_steps = min(cfg.num_steps, len(examples) // cfg.batch_size)
    print(f"Training {num_steps} steps (batch={cfg.batch_size}, micro={cfg.micro_batch_size})")

    optimizer: torch.optim.AdamW | None = None
    step = 0
    for batch_start in range(0, len(examples), cfg.batch_size):
        if step >= num_steps:
            break
        batch = examples[batch_start: batch_start + cfg.batch_size]
        n = len(batch)
        n_accum = math.ceil(n / cfg.micro_batch_size)

        for mb_start in range(0, n, cfg.micro_batch_size):
            mb = batch[mb_start: mb_start + cfg.micro_batch_size]
            n_micro = len(mb)
            max_len = max(len(e["tokens"]) for e in mb)

            inp = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
            tgt = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
            wts = torch.zeros(n_micro, max_len, dtype=torch.float32, device=device)
            attn = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
            for i, e in enumerate(mb):
                sl = len(e["tokens"])
                inp[i, :sl] = torch.tensor(e["tokens"], dtype=torch.long)
                tgt[i, :sl] = torch.tensor(e["targets"], dtype=torch.long)
                wts[i, :sl] = torch.tensor(e["weights"], dtype=torch.float32)
                attn[i, :sl] = 1

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=inp, attention_mask=attn, labels=tgt, use_cache=False)
                per_token_ce = model._cached_per_token_ce  # type: ignore[attr-defined]
                weight_sum = wts.sum()
                loss = (per_token_ce * wts).sum() / weight_sum if weight_sum > 0 else per_token_ce.sum() * 0.0
            (loss / n_accum).backward()
            del loss, per_token_ce

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.weight_decay,
            )
        progress = step / max(num_steps - 1, 1)
        lr = cfg.lr_end + (cfg.learning_rate - cfg.lr_end) * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        _tie_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1e9,
        )
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        print(f"  step {step}/{num_steps}: grad_norm={grad_norm:.4f}, lr={lr:.2e}", flush=True)

    print(f"Done. Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    return _save_adapter(model, cfg.output_dir)


def _save_adapter(model, output_dir: str) -> str:
    """Save the adapter, fix the lm_head key prefix, and zip it for submission."""
    import zipfile
    from safetensors.torch import load_file, save_file

    for f in os.listdir(output_dir):
        if f.startswith("adapter"):
            os.remove(os.path.join(output_dir, f))
    model.save_pretrained(output_dir)

    # The runtime model nests lm_head under ``backbone``; rename so keys match.
    st_path = os.path.join(output_dir, "adapter_model.safetensors")
    tensors = load_file(st_path)
    save_file(
        {k.replace("base_model.model.lm_head.", "base_model.model.backbone.lm_head."): v
         for k, v in tensors.items()},
        st_path,
    )

    adapter_files = [f for f in os.listdir(output_dir) if f.startswith("adapter")]
    zip_path = os.path.join(output_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in adapter_files:
            zf.write(os.path.join(output_dir, fname), fname)
    for fname in adapter_files:
        os.remove(os.path.join(output_dir, fname))
    print(f"Wrote {zip_path}")
    return zip_path
