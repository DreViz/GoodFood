#!/usr/bin/env python
"""Merge the planner LoRA into the base model and export GGUF Q4_K_M.

Takes the adapter from train_planner_qlora.py, merges it into
`unsloth/Qwen3-1.7B`, quantises to Q4_K_M (the project's quantization floor —
docs/model_specs.md) and writes a GGUF Ollama can serve, so the fine-tuned
planner drops into the runtime by changing one env var. ML imports are deferred
so --dry-run works anywhere.

    python -m scripts.export_planner_gguf [--dry-run]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

QUANTIZATION = "q4_k_m"
FINAL_NAME = "goodfoods-planner-q4km.gguf"
MODELFILE = "Modelfile.goodfoods-planner"
OLLAMA_MODEL_NAME = "goodfoods-planner"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge the planner LoRA and export a Q4_K_M GGUF for Ollama.",
    )
    p.add_argument("--adapter", default="adapters/planner_lora",
                   help="LoRA adapter directory from the training step.")
    p.add_argument("--out-dir", default="gguf",
                   help="Directory for the final .gguf.")
    p.add_argument("--work-dir", default="gguf/_merged",
                   help="Scratch directory Unsloth writes the merge + GGUF into.")
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="Sequence length used when reloading the adapter.")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Reload the base in 4-bit. Default is 16-bit, which "
                        "gives a cleaner merge when RAM allows (~3.4 GB for 1.7B).")
    p.add_argument("--keep-work-dir", action="store_true",
                   help="Keep the scratch merge directory (it is large).")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and print the plan; import no ML libs.")
    return p.parse_args(argv)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def find_gguf(root: Path) -> List[Path]:
    """All .gguf files under `root`, largest first."""
    return sorted(root.rglob("*.gguf"), key=lambda p: p.stat().st_size, reverse=True)


def _quant_tag(name: str) -> str:
    """Normalise a quantization label/filename for comparison ("Q4_K_M" -> "q4km")."""
    return name.lower().replace("_", "").replace("-", "").replace(".", "")


def select_quantised_gguf(candidates: List[Path], quantization: str) -> Path:
    """Pick the GGUF that actually carries the requested quantization.

    Selecting by size is WRONG here: Unsloth leaves its F16 intermediate
    (~3.4 GB for a 1.7B) beside the quantised output (~1.1 GB), so "largest
    file" publishes the UNQUANTISED model under a q4km filename — breaking the
    Q4_K_M floor in docs/model_specs.md and the 4 GB VRAM budget, with nothing
    downstream to catch it. Match on the name and refuse to guess.
    """
    tag = _quant_tag(quantization)
    matches = [p for p in candidates if tag in _quant_tag(p.name)]
    if not matches:
        raise SystemExit(
            f"No {quantization.upper()} GGUF found. Produced files: "
            + ", ".join(f"{p.name} ({p.stat().st_size / 1e9:.2f} GB)" for p in candidates)
            + "\nRefusing to publish an unverified quantization."
        )
    return matches[0]


def print_next_steps(final_path: Path) -> None:
    modelfile = REPO_ROOT / MODELFILE
    print("\n" + "=" * 72)
    print(f" GGUF ready: {final_path}")
    print("=" * 72)
    print("\nRegister it with Ollama (run from the repo root):\n")
    print(f"  ollama create {OLLAMA_MODEL_NAME} -f {MODELFILE}")
    print("\nThen point the agent at it and re-run the evals:\n")
    print(f"  python -m scripts.evaluate --model {OLLAMA_MODEL_NAME}")
    print(f"  python -m scripts.evaluate --model {OLLAMA_MODEL_NAME} "
          "--fixture tests/eval/heldout_conversations.yaml")
    if not modelfile.exists():  # pragma: no cover - the file is committed
        print(f"\nWARNING: {modelfile} is missing - `ollama create` will fail.")


def export(args: argparse.Namespace) -> int:
    from unsloth import FastLanguageModel

    adapter = _resolve(args.adapter)
    work_dir = _resolve(args.work_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading adapter: {adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    print(f"Merging LoRA into the base and quantising to {QUANTIZATION.upper()}...")
    work_dir.mkdir(parents=True, exist_ok=True)

    # Unsloth renamed this kwarg between releases, so try the known spellings.
    # There is deliberately NO bare-call fallback: calling without a
    # quantization argument silently exports Unsloth's default (f16/q8_0), and
    # an unnoticed wrong quantization costs a whole GPU session.
    last_error: Exception | None = None
    for kwargs in ({"quantization_method": QUANTIZATION},
                   {"quantization_type": QUANTIZATION}):
        try:
            model.save_pretrained_gguf(str(work_dir), tokenizer, **kwargs)
            last_error = None
            break
        except TypeError as exc:
            # Only a signature mismatch is worth retrying — re-running the whole
            # merge on an unrelated TypeError wastes many minutes.
            if "quantization" not in str(exc):
                raise
            last_error = exc
            continue
    if last_error is not None:
        raise SystemExit(
            f"save_pretrained_gguf rejected both quantization kwargs: {last_error}\n"
            "Check the installed unsloth version's signature before retrying."
        )

    candidates = find_gguf(work_dir)
    if not candidates:
        raise SystemExit(f"No .gguf produced under {work_dir}.")

    produced = select_quantised_gguf(candidates, QUANTIZATION)
    size_gb = produced.stat().st_size / 1e9
    final_path = out_dir / FINAL_NAME
    print(f"Selected {produced.name} ({size_gb:.2f} GB)")
    ignored = [p for p in candidates if p != produced]
    if ignored:
        print("  (intermediates ignored: "
              + ", ".join(f"{p.name} {p.stat().st_size / 1e9:.2f} GB" for p in ignored)
              + ")")
    if size_gb > 2.5:
        # A 1.7B at Q4_K_M lands near 1.1-1.4 GB. Anything much larger suggests
        # an unquantised file slipped through the name match.
        print(f"WARNING: {size_gb:.2f} GB is larger than expected for "
              f"{QUANTIZATION.upper()} at this model size - verify before serving it.")
    shutil.move(str(produced), str(final_path))

    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
        print(f"Removed scratch directory {work_dir}")

    print_next_steps(final_path)
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    adapter = _resolve(args.adapter)

    if args.dry_run:
        print("=" * 72)
        print(" GGUF export - dry run")
        print("=" * 72)
        print(f"adapter    : {adapter}  "
              f"({'found' if adapter.exists() else 'MISSING - train first'})")
        print(f"work dir   : {_resolve(args.work_dir)}")
        print(f"final gguf : {_resolve(args.out_dir) / FINAL_NAME}")
        print(f"quantise   : {QUANTIZATION.upper()} (project floor, docs/model_specs.md)")
        print(f"base dtype : {'4-bit' if args.load_in_4bit else '16-bit'}")
        modelfile = REPO_ROOT / MODELFILE
        print(f"modelfile  : {modelfile}  "
              f"({'found' if modelfile.exists() else 'MISSING'})")
        print(f"\n  ollama create {OLLAMA_MODEL_NAME} -f {MODELFILE}")
        print("\nDry run OK - no ML imports made.")
        return 0

    if not adapter.exists():
        raise SystemExit(
            f"Adapter not found: {adapter}\n"
            "Train it first: python -m scripts.train_planner_qlora"
        )
    return export(args)


if __name__ == "__main__":
    sys.exit(main())
