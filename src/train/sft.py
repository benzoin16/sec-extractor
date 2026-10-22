"""QLoRA Supervised Fine-Tuning (SFT) script using Hugging Face TRL and PEFT.

Runs on rented cloud GPUs (RunPod, Modal, Colab).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
import yaml

app = typer.Typer(
    name="train",
    help="Train a QLoRA adapter using Hugging Face TRL and PEFT.",
    add_completion=False,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_dataset_as_conversations(file_path: Path) -> list[dict]:
    """Load JSONL dataset and extract messages list for chat template."""
    conversations = []
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                data = json.loads(line)
                if "messages" in data:
                    conversations.append({"messages": data["messages"]})
    return conversations


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/train_qlora.yaml"), "--config", "-c", help="Path to training config YAML."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and dataset loading without starting GPU training.",
    ),
) -> None:
    if not config.exists():
        logger.error("Configuration file '%s' does not exist.", config)
        raise typer.Exit(1)

    with open(config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    model_cfg = cfg.get("model", {})
    ds_cfg = cfg.get("dataset", {})
    qlora_cfg = cfg.get("qlora", {})
    train_cfg = cfg.get("training", {})
    chat_cfg = cfg.get("chat_template", {})

    train_path = Path(ds_cfg.get("train_file", "data/processed/train.jsonl"))
    val_path = Path(ds_cfg.get("val_file", "data/processed/val.jsonl"))

    logger.info("Verifying dataset files...")
    if not train_path.exists():
        logger.error("Train dataset '%s' not found.", train_path)
        raise typer.Exit(1)

    train_data = load_dataset_as_conversations(train_path)
    val_data = load_dataset_as_conversations(val_path) if val_path.exists() else []

    logger.info(
        "Loaded %d training examples and %d validation examples.", len(train_data), len(val_data)
    )

    if dry_run:
        logger.info("[Dry Run] Config and datasets validated successfully.")
        return

    # Check GPU availability and HF libraries
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    except ImportError as exc:
        logger.error(
            "Missing GPU training dependencies: %s\n"
            "This script is intended to run on a rented GPU with `pip install -e '.[train]'`.\n"
            "See docs/gpu_runbook.md for instructions.",
            exc,
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        logger.warning("CUDA is not available. QLoRA 4-bit training requires an NVIDIA GPU.")

    logger.info("Loading tokenizer for %s...", model_cfg["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = getattr(torch, qlora_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
    )

    logger.info("Loading base model in 4-bit QLoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=qlora_cfg.get("r", 16),
        lora_alpha=qlora_cfg.get("lora_alpha", 32),
        lora_dropout=qlora_cfg.get("lora_dropout", 0.05),
        bias=qlora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=qlora_cfg.get("target_modules"),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Data formatting and assistant loss collator
    response_template = chat_cfg.get("response_template", "<|im_start|>assistant\n")
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    def formatting_prompts_func(example: dict) -> list[str]:
        output_texts = []
        for msgs in example["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data) if val_data else None

    output_dir = Path(train_cfg.get("output_dir", "models/qwen2.5-7b-qlora"))
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg.get("eval_steps", 50),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        max_seq_length=ds_cfg.get("max_seq_length", 4096),
        seed=train_cfg.get("seed", 42),
        report_to=cfg.get("logging", {}).get("report_to", "none"),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        data_collator=collator,
        formatting_func=formatting_prompts_func,
        args=training_args,
    )

    logger.info("Starting QLoRA training...")
    trainer.train()

    logger.info("Saving final adapter checkpoint to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Training complete!")


if __name__ == "__main__":
    app()
