"""Fast single-GPU QLoRA training script using Unsloth.

Shares the exact same YAML schema as `configs/train_qlora.yaml`.
Intended for single-GPU RunPod or Colab instances where Unsloth is installed.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
import yaml

app = typer.Typer(
    name="train_unsloth",
    help="Fast single-GPU QLoRA fine-tuning using Unsloth.",
    add_completion=False,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_dataset_as_conversations(file_path: Path) -> list[dict]:
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
        False, "--dry-run", help="Validate config without starting training."
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

    train_path = Path(ds_cfg.get("train_file", "data/processed/train.jsonl"))
    if not train_path.exists():
        logger.error("Train dataset '%s' not found.", train_path)
        raise typer.Exit(1)

    train_data = load_dataset_as_conversations(train_path)
    logger.info("Loaded %d training examples for Unsloth.", len(train_data))

    if dry_run:
        logger.info("[Dry Run] Config validated for Unsloth.")
        return

    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        logger.error(
            "Unsloth is not installed in this environment: %s\n"
            "Install Unsloth on Linux/CUDA via: `pip install 'unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git'`",
            exc,
        )
        sys.exit(1)

    max_seq_length = ds_cfg.get("max_seq_length", 4096)
    logger.info("Initializing FastLanguageModel for %s...", model_cfg["name_or_path"])
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name_or_path"],
        max_seq_length=max_seq_length,
        load_in_4bit=qlora_cfg.get("load_in_4bit", True),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=qlora_cfg.get("r", 16),
        target_modules=qlora_cfg.get("target_modules"),
        lora_alpha=qlora_cfg.get("lora_alpha", 32),
        lora_dropout=qlora_cfg.get("lora_dropout", 0.05),
        bias=qlora_cfg.get("bias", "none"),
        use_gradient_checkpointing="unsloth",
        random_state=train_cfg.get("seed", 42),
    )

    train_ds = Dataset.from_list(train_data)
    output_dir = Path(train_cfg.get("output_dir", "models/qwen2.5-7b-qlora-unsloth"))
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        seed=train_cfg.get("seed", 42),
        report_to=cfg.get("logging", {}).get("report_to", "none"),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        dataset_text_field="messages",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    logger.info("Training with Unsloth...")
    trainer.train()

    logger.info("Saving adapter to %s...", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Unsloth training complete!")


if __name__ == "__main__":
    app()
