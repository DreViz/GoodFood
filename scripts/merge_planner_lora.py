#!/usr/bin/env python
"""Phase 8 — merge the planner LoRA into a 16-bit safetensors directory.

Compiler-free alternative to scripts/export_planner_gguf.py: instead of
compiling llama.cpp locally (needs CMake + MSVC on Windows), we produce a
merged 16-bit HuggingFace-format model and let Ollama's BUILT-IN converter
handle GGUF conversion + Q4_K_M quantization at `ollama create` time:

    ollama create goodfoods-planner --quantize q4_K_M -f Modelfile.goodfoods-planner

Same quantization floor (Q4_K_M, docs/model_specs.md), same runtime path,
same env-var swap. The GGUF script remains for machines with a C++ toolchain.

Merging from the 4-bit-loaded base is the standard Unsloth export flow: the
dequantize->merge->requantize round trip is what save_pretrained_gguf does
internally, and the final Q4_K_M quantization dominates any precision delta.

USAGE
-----
    .venv-train/Scripts/python.exe -m scripts.merge_planner_lora
    .venv-train/Scripts/python.exe -m scripts.merge_planner_lora --dry-run

OUTPUT
------
    gguf/_merged_fp16/   config.json + model safetensors + tokenizer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ADAPTER = "adapters/planner_lora"
DEFAULT_OUT = "gguf/_merged_fp16"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge the planner LoRA to 16-bit safetensors for Ollama.",
    )
    p.add_argument("--adapter", default=DEFAULT_ADAPTER,
                   help="LoRA adapter directory from train_planner_qlora.py.")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="Output directory for the merged 16-bit model.")
    p.add_argument("--dry-run", action="store_true",
                   help="Check inputs only; import no ML libraries.")
    return p.parse_args(argv)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv=None) -> int:
    args = parse_args(argv)
    adapter = _resolve(args.adapter)
    out_dir = _resolve(args.out)

    if args.dry_run:
        print(f"adapter : {adapter}  "
              f"({'found' if adapter.exists() else 'MISSING - train first'})")
        print(f"out dir : {out_dir}")
        print("\nDry run OK - no ML imports made.")
        return 0
    if not adapter.exists():
        raise SystemExit(f"Adapter not found: {adapter}\n"
                         "Train it first: python -m scripts.train_planner_qlora")

    # Unsloth before transformers/trl so its patches install first.
    from unsloth import FastLanguageModel

    print(f"Loading adapter: {adapter} (4-bit base, cached weights)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Merging LoRA and writing 16-bit model to {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(out_dir), tokenizer,
                                 save_method="merged_16bit")

    files = sorted(p.name for p in out_dir.iterdir())
    total_gb = sum(p.stat().st_size for p in out_dir.iterdir()) / 1e9
    print(f"Wrote {len(files)} files, {total_gb:.2f} GB total:")
    for name in files:
        print(f"  {name}")

    modelfile = REPO_ROOT / "Modelfile.goodfoods-planner"
    print("\nNext steps:")
    print(f"  ollama create goodfoods-planner --quantize q4_K_M "
          f"-f {modelfile.name}")
    print("  python -m scripts.evaluate --model goodfoods-planner")
    return 0


if __name__ == "__main__":
    sys.exit(main())
