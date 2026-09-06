"""
FINAL PHASE, step 2: fine-tune bge-m3 or Qwen3-Embedding on our triplets.

    python3 -m src.training.finetune_embedder \\
        --model BAAI/bge-m3 \\
        --train data/triplets/hf/triplets_27b_bm25_hf_train.parquet \\
                data/triplets/hf/triplets_27b_bm25_hf_eval.parquet \\
        --output models/bge-m3-bm25

Training data is `train` **plus** `eval`, as asked. That leaves nothing held
out, so there is no early stopping and no way to pick a checkpoint by
validation: the run is a fixed step budget and the last checkpoint is the one
evaluated. `--holdout` carves a slice back out if you want a curve, but the
default follows the brief.

`MultipleNegativesRankingLoss` is the loss: each row contributes its own hard
negative *and* every other row's positive in the batch as an in-batch negative.
That makes batch size a quality setting, not just a speed one - a bigger batch
is a harder contrastive problem - which is why `--batch-size` is pushed as far
as memory allows rather than tuned for throughput.

**LoRA for the 8B model is a constraint, not a preference.** A full fine-tune
checkpoint of Qwen3-Embedding-8B is ~16 GB, and the account's disk quota has
about 33 GB free; the second saved checkpoint would fill it. LoRA adapters are
a few hundred MB, and `evaluate_retrieval.py --adapter` loads one onto the base
model without materialising a merged copy. bge-m3 is 568M and trains fully.

The Qwen instruction prefix is applied to anchors here exactly as
`evaluate_retrieval.py` applies it at inference. Training without it and
evaluating with it would mean the model never sees at training time the input
distribution it is scored on.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]

from evaluate_retrieval import QWEN_QUERY_PROMPT  # noqa: E402


def load_rows(paths: List[str]) -> "datasets.Dataset":
    from datasets import Dataset, concatenate_datasets
    import pyarrow.parquet as pq
    parts = []
    for path in paths:
        if path.endswith(".parquet"):
            parts.append(Dataset(pq.read_table(path)))
        else:
            with open(path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            parts.append(Dataset.from_list(rows))
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]


def lora_targets(module) -> List[str]:
    """Which linear layers to adapt, read off the model instead of assumed.

    Hardcoding the decoder names (`q_proj`, `gate_proj`, …) works for Qwen3 and
    fails outright on bge-m3, which is XLM-RoBERTa and calls them
    `query`/`key`/`value`/`dense` - peft raises "Target modules not found"
    rather than adapting nothing quietly, but only after the model is loaded.
    Matching against the linear layers the model actually has covers both
    families and whatever comes next.
    """
    import torch.nn as nn
    names = {name.rsplit(".", 1)[-1]
             for name, mod in module.named_modules() if isinstance(mod, nn.Linear)}
    for group in (["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
                  ["query", "key", "value", "dense"]):
        hit = [n for n in group if n in names]
        if len(hit) >= 3:
            return hit
    return sorted(names)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune an embedding model on (anchor, positive, negative).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--train", required=True, nargs="+",
                   help="one or more parquet/jsonl files, concatenated")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=32,
                   help="also the number of in-batch negatives, so it is a "
                        "quality setting; raise it until memory says no")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="-1 runs the full epoch count; set it to fit a wall clock")
    p.add_argument("--lr", type=float, default=None,
                   help="default 2e-5 for a full fine-tune, 1e-4 for LoRA")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-seq-length", type=int, default=512,
                   help="passages are capped at 4000 chars upstream; 512 tokens "
                        "covers most of that and is what the eval encodes at")
    p.add_argument("--lora", action=argparse.BooleanOptionalAction, default=None,
                   help="default: on for models over ~1B, off otherwise")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--holdout", type=int, default=0,
                   help="rows kept out of training for a loss curve; 0 trains "
                        "on everything, which is what the brief asks")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction,
                   default=None, help="default: on when LoRA is on")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer,
                                       SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    is_qwen = "qwen" in args.model.lower()
    print(f"▸ {args.model}")
    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length

    n_params = sum(p.numel() for p in model.parameters())
    use_lora = args.lora if args.lora is not None else n_params > 1e9
    print(f"  {n_params / 1e9:.2f}B parameters, LoRA {'on' if use_lora else 'off'}")

    if use_lora:
        from peft import LoraConfig, get_peft_model
        inner = model[0].auto_model
        cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="FEATURE_EXTRACTION",
            target_modules=lora_targets(inner),
        )
        print(f"  LoRA targets: {', '.join(lora_targets(inner))}")
        # Gradient checkpointing recomputes each block's activations from its
        # inputs, and those inputs do not require grad when the base weights are
        # frozen - so the graph ends before it reaches the adapters and backward
        # dies with "element 0 of tensors does not require grad". Forcing the
        # embedding output to require grad reconnects it.
        if hasattr(inner, "enable_input_require_grads"):
            inner.enable_input_require_grads()
        model[0].auto_model = get_peft_model(inner, cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  trainable: {trainable / 1e6:.1f}M ({100 * trainable / n_params:.2f}%)")

    dataset = load_rows(args.train)
    keep = [c for c in ("anchor", "positive", "negative") if c in dataset.column_names]
    dataset = dataset.select_columns(keep)
    if is_qwen:
        prompt = QWEN_QUERY_PROMPT
        dataset = dataset.map(lambda b: {"anchor": [prompt + a for a in b["anchor"]]},
                              batched=True, desc="applying the Qwen query prompt")
    dataset = dataset.shuffle(seed=args.seed)

    holdout = None
    if args.holdout:
        holdout = dataset.select(range(args.holdout))
        dataset = dataset.select(range(args.holdout, len(dataset)))
    print(f"  {len(dataset)} training rows"
          + (f", {len(holdout)} held out" if holdout else ", nothing held out"))

    lr = args.lr if args.lr is not None else (1e-4 if use_lora else 2e-5)
    grad_ckpt = (args.grad_checkpointing
                 if args.grad_checkpointing is not None else use_lora)
    targs = SentenceTransformerTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=lr,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16 and torch.cuda.is_available(),
        gradient_checkpointing=grad_ckpt,
        # Non-reentrant checkpointing handles frozen-parameter graphs correctly;
        # the reentrant implementation is what raises on the LoRA setup above.
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        logging_steps=50,
        save_strategy="no",          # one checkpoint at the end; quota is tight
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=4,
    )
    print(f"  lr {lr}, batch {args.batch_size}, "
          f"{'bf16' if targs.bf16 else 'fp32'}, grad-checkpointing {grad_ckpt}")

    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=dataset,
        eval_dataset=holdout, loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()

    os.makedirs(args.output, exist_ok=True)
    if use_lora:
        # Just the adapter: the base weights are unchanged and already cached,
        # and a merged copy would be 16 GB against a 33 GB quota.
        model[0].auto_model.save_pretrained(args.output)
        print(f"✓ adapter saved to {args.output}")
    else:
        model.save(args.output)
        print(f"✓ model saved to {args.output}")
    with open(os.path.join(args.output, "training_args.json"), "w") as f:
        json.dump({**vars(args), "lora": use_lora, "lr": lr,
                   "rows": len(dataset)}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
