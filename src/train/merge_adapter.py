"""Merge a trained LoRA adapter into the base 16-bit model for standalone deployment."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="merge",
    help="Merge PEFT LoRA adapter with base model to export full weights.",
    add_completion=False,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@app.command()
def main(
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct",
        "--base-model",
        "-b",
        help="Hugging Face base model identifier or path.",
    ),
    adapter_dir: Path = typer.Option(
        Path("models/qwen2.5-7b-qlora"),
        "--adapter-dir",
        "-a",
        help="Path to trained LoRA adapter directory.",
    ),
    output_dir: Path = typer.Option(
        Path("models/qwen2.5-7b-merged"),
        "--output-dir",
        "-o",
        help="Target directory for merged full weights.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Check paths without running GPU merge."),
) -> None:
    if not adapter_dir.exists():
        logger.error("Adapter directory '%s' does not exist.", adapter_dir)
        raise typer.Exit(1)

    if dry_run:
        logger.info("[Dry Run] Adapter directory '%s' verified.", adapter_dir)
        return

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        logger.error("Missing dependencies for model merging: %s", exc)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loading base model %s in bfloat16...", base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    logger.info("Loading LoRA adapter from %s...", adapter_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    logger.info("Merging adapter into base model weights...")
    merged_model = model.merge_and_unload()

    logger.info("Saving merged model to %s...", output_dir)
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Model merge complete! Ready for serving or quantization.")


if __name__ == "__main__":
    app()
