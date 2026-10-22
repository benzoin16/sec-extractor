"""Async evaluation runner connecting to any OpenAI-compatible API endpoint."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from openai import AsyncOpenAI

from src.eval.metrics import evaluate_sample
from src.eval.models import SampleEvalResult

logger = logging.getLogger(__name__)


class EvalRunner:
    """Executes evaluation benchmarks against an OpenAI-compatible model endpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str = "EMPTY",
        max_concurrency: int = 8,
        timeout: float = 60.0,
        temperature: float = 0.0,
        seed: int = 42,
        rel_tol: float = 0.005,
        eps_abs_tol: float = 0.02,
    ) -> None:
        self.model_name = model_name
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.temperature = temperature
        self.seed = seed
        self.rel_tol = rel_tol
        self.eps_abs_tol = eps_abs_tol

    async def _evaluate_single(
        self,
        example: dict[str, Any],
        idx: int,
    ) -> SampleEvalResult:
        """Send a single inference request and evaluate the output."""
        messages = example.get("messages", [])
        # Ground truth is the assistant content of the training example
        truth_json = {}
        for msg in messages:
            if msg.get("role") == "assistant":
                try:
                    truth_json = json.loads(msg.get("content", "{}"))
                except Exception:
                    truth_json = {}

        # The input messages sent to the model omit the ground-truth assistant reply
        input_messages = [m for m in messages if m.get("role") in ("system", "user")]

        example_id = example.get("metadata", {}).get("accession_number", f"ex_{idx}")

        async with self.semaphore:
            start_t = time.perf_counter()
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=input_messages,
                    temperature=self.temperature,
                    seed=self.seed,
                )
                latency = time.perf_counter() - start_t
                choice = response.choices[0]
                completion_text = choice.message.content or ""
                usage = response.usage
                prompt_tok = usage.prompt_tokens if usage else len(str(input_messages)) // 4
                compl_tok = usage.completion_tokens if usage else len(completion_text) // 4
            except Exception as exc:
                latency = time.perf_counter() - start_t
                logger.error("Inference failed for %s: %s", example_id, exc)
                completion_text = f"ERROR: {exc}"
                prompt_tok = 0
                compl_tok = 0

        return evaluate_sample(
            completion_text=completion_text,
            truth_dict=truth_json,
            example_id=example_id,
            latency_seconds=latency,
            prompt_tokens=prompt_tok,
            completion_tokens=compl_tok,
            rel_tol=self.rel_tol,
            eps_abs_tol=self.eps_abs_tol,
        )

    async def run(
        self,
        test_examples: list[dict[str, Any]],
    ) -> list[SampleEvalResult]:
        """Run evaluation concurrently across all test examples."""
        tasks = [self._evaluate_single(ex, i) for i, ex in enumerate(test_examples)]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        await self.client.close()
