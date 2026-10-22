"""AWQ 4-bit post-training quantization script using llm-compressor for vLLM deployment.

Note: AutoAWQ is deprecated. vLLM and Neural Magic officially maintain `llm-compressor`
for W4A16 AWQ and compressed-tensors quantization.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="quantize",
    help="Quantize merged model to 4-bit AWQ using llm-compressor for vLLM serving.",
    add_completion=False,
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_calibration_dataset(train_file: Path, max_samples: int = 128) -> list[dict]:
    """Extract sample text prompts from training data for activation calibration."""
    samples = []
    if not train_file.exists():
        return samples

    with open(train_file, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                item = json.loads(line)
                messages = item.get("messages", [])
                text = " ".join(
                    m.get("content", "") for m in messages if m.get("role") != "assistant"
                )
                if text:
                    samples.append({"text": text})
                if len(samples) >= max_samples:
                    break
    return samples


@app.command()
def main(
    model_path: Path = typer.Option(
        Path("models/qwen2.5-7b-merged"), "--model-path", "-m", help="Path to merged 16-bit model."
    ),
    output_dir: Path = typer.Option(
        Path("models/qwen2.5-7b-awq"),
        "--output-dir",
        "-o",
        help="Target directory for 4-bit AWQ model.",
    ),
    train_file: Path = typer.Option(
        Path("data/processed/train.jsonl"),
        "--train-file",
        help="Path to training data for calibration.",
    ),
    max_sequence_length: int = typer.Option(
        2048, "--max-seq-len", help="Calibration sequence length."
    ),
    num_calibration_samples: int = typer.Option(
        128, "--num-samples", help="Number of calibration samples."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Check input paths without running quantization."
    ),
) -> None:
    if not model_path.exists():
        logger.error("Model path '%s' does not exist.", model_path)
        raise typer.Exit(1)

    if dry_run:
        logger.info("[Dry Run] Paths verified for llm-compressor AWQ quantization.")
        return

    try:
        from datasets import Dataset
        from llmcompressor.modifiers.awq import AWQModifier
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.transformers import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        logger.error(
            "llm-compressor is not installed: %s\n"
            "Install in your GPU environment via: `pip install llm-compressor`\n"
            "See docs/gpu_runbook.md for instructions.",
            exc,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model for AWQ quantization from %s...", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    raw_calib = load_calibration_dataset(train_file, max_samples=num_calibration_samples)
    logger.info("Loaded %d calibration samples from %s", len(raw_calib), train_file)

    def preprocess(example: dict) -> dict:
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_sequence_length,
            padding=False,
        )

    ds = Dataset.from_list(raw_calib)
    calib_ds = ds.map(preprocess, remove_columns=["text"])

    # AWQ Recipe with W4A16 asymmetric quantization ignoring lm_head
    recipe = [
        AWQModifier(),
        QuantizationModifier(
            ignore=["lm_head"],
            scheme="W4A16_ASYM",
            targets=["Linear"],
        ),
    ]

    logger.info("Executing one-shot AWQ quantization...")
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=max_sequence_length,
        num_calibration_samples=len(raw_calib),
    )

    logger.info("Saving AWQ compressed model to %s...", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("AWQ quantization complete! Model is natively servable via vLLM.")


if __name__ == "__main__":
    app()
