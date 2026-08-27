#!/usr/bin/env python
"""QLoRA fine-tune of the GoodFoods planner (Unsloth + TRL).

Trains LoRA adapters on a frozen 4-bit `unsloth/Qwen3-1.7B` so the small model
learns the 3-phase procedure its 71.1% eval score shows it gets wrong — chiefly
emitting create_reservation without a prior check_availability. The ML stack is
imported only inside run_training(), so --dry-run validates data + config on
any machine.

    python -m scripts.train_planner_qlora --dry-run
    python -m scripts.train_planner_qlora --data data/planner_train.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Qwen3 / ChatML turn markers. train_on_responses_only masks everything up to
# the assistant marker so loss is computed on the planner's JSON only.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QLoRA fine-tune the GoodFoods planner (Unsloth).",
    )
    p.add_argument("--data", default="data/planner_train.jsonl",
                   help="Training JSONL (chat format, one object per line).")
    p.add_argument("--val-data", default=None,
                   help="Validation JSONL. Default: <data-stem>_val.jsonl if present.")
    p.add_argument("--base-model", default="unsloth/Qwen3-1.7B",
                   help="Base model tag. Loaded in 4-bit (QLoRA).")
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES),
                   help="Comma-separated LoRA target modules.")
    p.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                   help="Load the base model in 16-bit instead of 4-bit (needs more VRAM).")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2,
                   help="Per-device train batch size.")
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch * accum).")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Cap training steps (-1 = use --epochs). Handy for smoke runs.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--packing", action="store_true",
                   help="Enable sequence packing (off by default — samples are short).")
    p.add_argument("--output-dir", default="adapters/planner_lora",
                   help="Where the LoRA adapter is saved.")
    p.add_argument("--checkpoint-dir", default="adapters/checkpoints",
                   help="Trainer working directory for intermediate checkpoints.")
    p.add_argument("--log-file", default="adapters/training_log.json")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate the dataset + resolved config, then exit 0. "
                        "Imports no ML libraries and needs no GPU.")
    return p.parse_args(argv)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a chat-format JSONL file, failing loudly on a malformed row."""
    if not path.exists():
        raise SystemExit(
            f"Dataset not found: {path}\n"
            "Generate it first:\n"
            "  python -m scripts.generate_training_data --n 3000 "
            "--out data/planner_train.jsonl"
        )
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: not valid JSON ({exc})")
            validate_row(row, f"{path}:{lineno}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path} is empty.")
    return rows


def validate_row(row: Dict[str, Any], where: str) -> None:
    """A row must be {"messages": [system, user, assistant]} with JSON content.

    `meta`, if present, is ignored by training — it exists for the manifest and
    the unit tests.
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise SystemExit(f"{where}: expected 3 messages, got {messages and len(messages)}")
    roles = [m.get("role") for m in messages]
    if roles != ["system", "user", "assistant"]:
        raise SystemExit(f"{where}: unexpected roles {roles}")
    for msg in messages:
        if not isinstance(msg.get("content"), str) or not msg["content"]:
            raise SystemExit(f"{where}: empty content in a {msg.get('role')} message")
    try:
        plan = json.loads(messages[2]["content"])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{where}: assistant content is not JSON ({exc})")
    if plan.get("plan") not in ("reply", "execute"):
        raise SystemExit(f"{where}: assistant plan is {plan.get('plan')!r}")


def dataset_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    plans: Dict[str, int] = {}
    actions: Dict[str, int] = {}
    for row in rows:
        plan = json.loads(row["messages"][2]["content"])
        plans[plan["plan"]] = plans.get(plan["plan"], 0) + 1
        key = plan.get("action") or "(reply)"
        actions[key] = actions.get(key, 0) + 1
    # Rough token budget check — chars/4 is the usual ballpark for English+JSON.
    longest = max(sum(len(m["content"]) for m in r["messages"]) for r in rows)
    return {
        "rows": len(rows),
        "plans": dict(sorted(plans.items())),
        "actions": dict(sorted(actions.items())),
        "longest_example_chars": longest,
        "longest_example_tokens_estimate": longest // 4,
    }


def length_warning(rows: List[Dict[str, Any]], max_seq_len: int) -> Optional[str]:
    """Flag samples whose estimated length exceeds the sequence budget.

    Truncation happens at the END of the sequence — which is where the
    assistant's JSON (the label) lives. A truncated sample is worse than a
    missing one, so this is a blocking concern, not cosmetic.

    Measured on the real prompts: the Phase-1 (discovery) system prompt alone is
    ~7.8 KB, putting discovery samples near ~2.1k tokens, while availability and
    booking sit around ~1.2-1.4k.
    """
    over = [r for r in rows
            if sum(len(m["content"]) for m in r["messages"]) // 4 > max_seq_len]
    if not over:
        return None
    phases: Dict[str, int] = {}
    for row in over:
        phase = (row.get("meta") or {}).get("phase", "unknown")
        phases[phase] = phases.get(phase, 0) + 1
    return (
        f"{len(over)}/{len(rows)} samples are estimated above --max-seq-len "
        f"({max_seq_len}) and would have their assistant JSON truncated: "
        f"{dict(sorted(phases.items()))}. Re-run with "
        f"`--max-seq-len 4096 --batch-size 1 --grad-accum 16`."
    )


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """The full resolved hyperparameter set — dumped in dry-run and at train time."""
    return {
        "base_model": args.base_model,
        "load_in_4bit": args.load_in_4bit,
        "max_seq_length": args.max_seq_len,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": [m.strip() for m in args.target_modules.split(",") if m.strip()],
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
        },
        "optim": {
            "learning_rate": args.lr,
            "lr_scheduler_type": args.lr_scheduler,
            "warmup_steps": args.warmup_steps,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "num_train_epochs": args.epochs,
            "max_steps": args.max_steps,
            "weight_decay": args.weight_decay,
            "optim": "adamw_8bit",
            "seed": args.seed,
        },
        "loss_masking": {
            "train_on_responses_only": True,
            "instruction_part": INSTRUCTION_PART,
            "response_part": RESPONSE_PART,
        },
        "packing": args.packing,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "output_dir": str(_resolve(args.output_dir)),
        "checkpoint_dir": str(_resolve(args.checkpoint_dir)),
        "log_file": str(_resolve(args.log_file)),
    }


def _filtered_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs `cls.__init__` accepts.

    TRL/transformers rename arguments between releases (`evaluation_strategy`
    -> `eval_strategy`, `tokenizer` -> `processing_class`, ...). Filtering keeps
    this script runnable on the GPU box months from now without an edit.
    """
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters)
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        print(f"  [compat] {cls.__name__} ignores: {', '.join(dropped)}")
    return {k: v for k, v in kwargs.items() if k in accepted}


def _prefer_kwarg(cls, kwargs: Dict[str, Any], preferred: str, legacy: str) -> None:
    """Keep only one spelling of a renamed argument, in place.

    Some transitional releases accept BOTH the new and the deprecated name and
    raise (or warn loudly) when given both, so pick the new one when it exists.
    """
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters)
    if preferred in accepted:
        kwargs.pop(legacy, None)
    elif legacy in accepted:
        kwargs[legacy] = kwargs.pop(preferred, kwargs.get(legacy))
    else:
        kwargs.pop(preferred, None)
        kwargs.pop(legacy, None)


def _render_messages(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Apply the model's chat template to one conversation.

    Qwen3 is a reasoning model whose template can inject a <think> block. The
    planner runs with reasoning OFF, so ask for the non-thinking rendering when
    the installed template supports it.
    """
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )


def make_loss_logger(log_path: Path):
    """A TrainerCallback that mirrors trainer logs into training_log.json."""
    from transformers import TrainerCallback

    class LossLogger(TrainerCallback):
        def __init__(self) -> None:
            self.records: List[Dict[str, Any]] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            entry: Dict[str, Any] = {"step": int(state.global_step)}
            if "loss" in logs:
                entry["train_loss"] = round(float(logs["loss"]), 6)
            if "eval_loss" in logs:
                entry["val_loss"] = round(float(logs["eval_loss"]), 6)
            if "learning_rate" in logs:
                entry["lr"] = float(logs["learning_rate"])
            if len(entry) == 1:  # step only — nothing worth recording
                return
            # Merge train/val entries logged at the same step.
            if self.records and self.records[-1]["step"] == entry["step"]:
                self.records[-1].update(entry)
            else:
                self.records.append(entry)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                json.dumps(self.records, indent=2) + "\n", encoding="utf-8",
            )

    return LossLogger()


def run_training(args: argparse.Namespace, config: Dict[str, Any],
                 train_rows: List[Dict[str, Any]],
                 val_rows: Optional[List[Dict[str, Any]]]) -> int:
    """Import the ML stack and train. Only called when --dry-run is absent."""
    # Unsloth must be imported before transformers/trl to install its patches.
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer

    try:  # TRL >= 0.12 puts the training args in SFTConfig
        from trl import SFTConfig as TrainArgs
    except ImportError:  # pragma: no cover - older TRL
        from transformers import TrainingArguments as TrainArgs

    print(f"\nLoading base model: {args.base_model} "
          f"({'4-bit' if args.load_in_4bit else '16-bit'})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,                      # let Unsloth pick (bf16/fp16 per GPU)
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        target_modules=config["lora"]["target_modules"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",   # the 4 GB-VRAM enabler
        random_state=args.seed,
    )

    def to_text(rows: List[Dict[str, Any]]) -> Dataset:
        return Dataset.from_list(
            [{"text": _render_messages(tokenizer, r["messages"])} for r in rows]
        )

    train_ds = to_text(train_rows)
    eval_ds = to_text(val_rows) if val_rows else None

    # Pre-flight: refuse to train on truncated labels. Cheap (seconds) compared
    # with discovering it in a flat loss curve two hours later.
    lengths = [len(tokenizer(t, add_special_tokens=False)["input_ids"])
               for t in train_ds["text"]]
    longest = max(lengths)
    over = sum(1 for n in lengths if n > args.max_seq_len)
    print(f"Token lengths: max={longest}, "
          f"mean={sum(lengths) // len(lengths)}, over-limit={over}")
    if over:
        print(
            f"\nABORT: {over}/{len(lengths)} samples exceed --max-seq-len "
            f"({args.max_seq_len}); the assistant JSON would be truncated away, "
            "which teaches the model nothing and quietly wrecks the run.\n"
            "Re-run with:\n"
            f"  python -m scripts.train_planner_qlora --max-seq-len 4096 "
            "--batch-size 1 --grad-accum 16"
        )
        return 2

    # Save the first rendered example so the templating can be eyeballed
    # without re-running the job.
    sample_path = _resolve(args.output_dir).parent / "sample_rendered.txt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(train_ds[0]["text"], encoding="utf-8")
    print(f"Rendered sample written to {sample_path}")

    train_kwargs = {
        "output_dir": config["checkpoint_dir"],
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.lr,
        "lr_scheduler_type": args.lr_scheduler,
        "logging_steps": args.logging_steps,
        "optim": "adamw_8bit",
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "report_to": "none",
        "save_strategy": "no",
        "max_seq_length": args.max_seq_len,
        "packing": args.packing,
        "dataset_text_field": "text",
    }
    if eval_ds is not None:
        # Renamed in transformers 4.46 (evaluation_strategy -> eval_strategy).
        train_kwargs["eval_strategy"] = "steps"
        train_kwargs["evaluation_strategy"] = "steps"
        train_kwargs["eval_steps"] = args.eval_steps
        train_kwargs["per_device_eval_batch_size"] = args.batch_size
        _prefer_kwarg(TrainArgs, train_kwargs, "eval_strategy", "evaluation_strategy")

    training_args = TrainArgs(**_filtered_kwargs(TrainArgs, train_kwargs))

    trainer_kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": training_args,
        "dataset_text_field": "text",
        "max_seq_length": args.max_seq_len,
        "packing": args.packing,
    }
    # Renamed in transformers 4.46 (tokenizer -> processing_class).
    _prefer_kwarg(SFTTrainer, trainer_kwargs, "processing_class", "tokenizer")
    trainer = SFTTrainer(**_filtered_kwargs(SFTTrainer, trainer_kwargs))

    # Mask the prompt: train only on the planner's JSON, never on the 4 KB
    # system prompt we already control.
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part=INSTRUCTION_PART,
            response_part=RESPONSE_PART,
        )
        print("Loss masked to assistant tokens (train_on_responses_only).")
    except Exception as exc:  # pragma: no cover - depends on unsloth version
        print(f"WARNING: train_on_responses_only unavailable ({exc}); "
              "training on full sequences instead.")

    log_path = _resolve(args.log_file)
    trainer.add_callback(make_loss_logger(log_path))

    print("\nTraining...\n")
    trainer.train()

    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    (out_dir.parent / "train_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8",
    )

    print(f"\nAdapter saved to  : {out_dir}")
    print(f"Training log      : {log_path}")
    print("\nNext step:")
    print("  python -m scripts.export_planner_gguf")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)

    train_path = _resolve(args.data)
    if args.val_data:
        val_path: Optional[Path] = _resolve(args.val_data)
    else:
        candidate = train_path.with_name(f"{train_path.stem}_val{train_path.suffix}")
        val_path = candidate if candidate.exists() else None

    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path) if val_path else None

    config = build_config(args)

    print("=" * 72)
    print(" GoodFoods planner QLoRA")
    print("=" * 72)
    print(f"train    : {train_path}")
    print(f"  {json.dumps(dataset_stats(train_rows))}")
    if val_rows:
        print(f"val      : {val_path}")
        print(f"  {json.dumps(dataset_stats(val_rows))}")
    else:
        print("val      : (none - no eval loss will be logged)")

    warning = length_warning(train_rows, config["max_seq_length"])
    if warning:
        print(f"\nWARNING: {warning}")

    if args.dry_run:
        print("\n--- resolved config (dry run) ---")
        print(json.dumps(config, indent=2))
        print("\nDry run OK - dataset valid, config resolved, no ML imports made.")
        return 0

    return run_training(args, config, train_rows, val_rows)


if __name__ == "__main__":
    sys.exit(main())
