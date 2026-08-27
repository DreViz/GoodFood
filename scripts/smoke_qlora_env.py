# Environment gate for QLoRA training on this machine: loads
# unsloth/Qwen3-1.7B-4bit, attaches LoRA adapters, and runs 10 training steps
# on toy data — CUDA torch, 4-bit load, forward/backward, optimizer, VRAM fit.
# Exits 0 = environment ready.
#
# Run with the TRAINING venv, not the app environment:
#   .venv-train/Scripts/python.exe -m scripts.smoke_qlora_env
#
# First run downloads ~1.2 GB of weights from Hugging Face.
import sys


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("FAIL: CUDA unavailable — install the CUDA torch build into .venv-train")
        return 2

    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

    from unsloth import FastLanguageModel
    from transformers import Trainer, TrainingArguments
    from datasets import Dataset
    from transformers import default_data_collator

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.2f}%)")

    # Toy corpus shaped like planner traffic: system/user/assistant chat turns.
    toy = [
        "<|im_start|>system\nYou are a JSON planner.<|im_end|>\n"
        "<|im_start|>user\ntomorrow at 7pm for 4<|im_end|>\n"
        "<|im_start|>assistant\n{\"plan\": \"execute\", \"action\": "
        "\"check_availability\"}<|im_end|>",
    ] * 8

    def tok(text: str):
        enc = tokenizer(text, truncation=True, max_length=512)
        # Causal LM needs labels; without them transformers 5.x returns logits
        # only and Trainer raises "did not return a loss".
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": list(enc["input_ids"])}

    ds = Dataset.from_dict({"text": toy})
    ds = ds.map(lambda row: tok(row["text"]), remove_columns=["text"])

    args = TrainingArguments(
        output_dir="adapters/_smoke",
        max_steps=10,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        fp16=False,
        bf16=True,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=default_data_collator)
    trainer.train()

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"PASS: 10 steps completed | peak VRAM {peak:.2f} GiB "
          f"of {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f} GiB")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
