"""Inference client supporting vLLM schema-guided decoding and frontier fallback."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from openai import AsyncOpenAI

from src.dataset.builder import SYSTEM_PROMPT
from src.serve.schemas import ExtractedFinancialMetrics, ValidationDetail
from src.serve.validator import SanityValidator

logger = logging.getLogger(__name__)

# Default model identifiers and endpoints
DEFAULT_VLLM_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
DEFAULT_VLLM_MODEL = os.getenv("VLLM_MODEL_NAME", "qwen2.5-7b-instruct-sec")
DEFAULT_FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "gpt-4o-mini")


class ModelExtractionResult:
    """Encapsulates the raw model inference and validation outcome."""

    def __init__(
        self,
        metrics: ExtractedFinancialMetrics,
        validation: ValidationDetail,
        model_name: str,
        fallback_used: bool,
        tokens_prompt: int,
        tokens_completion: int,
        latency_ms: float,
        raw_output: str,
    ) -> None:
        self.metrics = metrics
        self.validation = validation
        self.model_name = model_name
        self.fallback_used = fallback_used
        self.tokens_prompt = tokens_prompt
        self.tokens_completion = tokens_completion
        self.latency_ms = latency_ms
        self.raw_output = raw_output


class ExtractionInferenceClient:
    """Manages inference against primary vLLM server with schema-guided decoding,

    falling back to a frontier model if validation fails or server is unavailable.
    Target vLLM version: 0.6.3+ (uses response_format json_schema with guided_json fallback).
    """

    def __init__(
        self,
        vllm_base_url: str = DEFAULT_VLLM_URL,
        vllm_model: str = DEFAULT_VLLM_MODEL,
        vllm_api_key: str = "EMPTY",
        frontier_model: str = DEFAULT_FRONTIER_MODEL,
        frontier_api_key: str | None = None,
        frontier_base_url: str | None = None,
        validator: SanityValidator | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.vllm_base_url = vllm_base_url
        self.vllm_model = vllm_model
        self.frontier_model = frontier_model
        self.validator = validator or SanityValidator()
        self.timeout = timeout

        # Primary vLLM client (OpenAI-compatible)
        self.primary_client = AsyncOpenAI(
            base_url=self.vllm_base_url,
            api_key=vllm_api_key,
            timeout=self.timeout,
        )

        # Fallback Frontier client
        fallback_key = frontier_api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        fallback_url = frontier_base_url or os.getenv("OPENAI_BASE_URL", None)
        self.fallback_client = AsyncOpenAI(
            base_url=fallback_url,
            api_key=fallback_key,
            timeout=self.timeout,
        )

        # Build JSON Schema for structured output
        self.json_schema = ExtractedFinancialMetrics.model_json_schema()

    async def _call_endpoint(
        self,
        client: AsyncOpenAI,
        model_name: str,
        prompt: str,
        use_vllm_guided: bool = False,
    ) -> tuple[dict[str, Any], int, int, str]:
        """Invoke an OpenAI-compatible endpoint with schema-guided decoding."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Extract the financial metrics from the following SEC filing text.\n\n{prompt}",
            },
        ]

        # Standard OpenAI Structured Output format (supported in vLLM >= 0.6.0)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "financial_extraction",
                "schema": self.json_schema,
                "strict": True,
            },
        }

        # Extra body for vLLM native outlines / xgrammar guided decoding
        extra_body: dict[str, Any] = {}
        if use_vllm_guided:
            extra_body["guided_json"] = self.json_schema

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format=response_format,  # type: ignore[arg-type]
            extra_body=extra_body if extra_body else None,
            temperature=0.0,
            max_tokens=512,
        )

        raw_content = response.choices[0].message.content or "{}"
        parsed = json.loads(raw_content)

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return parsed, prompt_tokens, completion_tokens, raw_content

    async def extract(
        self,
        text: str,
        allow_fallback: bool = True,
        strict_validation: bool = True,
    ) -> ModelExtractionResult:
        """Execute extraction pipeline with guided decoding and frontier fallback."""
        start_time = time.perf_counter()
        fallback_used = False
        selected_model = self.vllm_model
        validation_result: ValidationDetail | None = None
        metrics: ExtractedFinancialMetrics | None = None
        prompt_tokens = 0
        completion_tokens = 0
        raw_text_out = ""

        # Step 1: Attempt Primary Inference with vLLM
        try:
            parsed_json, p_tok, c_tok, raw_text_out = await self._call_endpoint(
                client=self.primary_client,
                model_name=self.vllm_model,
                prompt=text,
                use_vllm_guided=True,
            )
            prompt_tokens = p_tok
            completion_tokens = c_tok
            metrics = ExtractedFinancialMetrics.model_validate(parsed_json)
            validation_result = self.validator.validate(metrics)

            # Check if strict validation failed
            if strict_validation and not validation_result.passed:
                logger.warning(
                    "Primary model %s failed financial sanity checks: %s",
                    self.vllm_model,
                    validation_result.errors,
                )
                if not allow_fallback:
                    latency = (time.perf_counter() - start_time) * 1000.0
                    return ModelExtractionResult(
                        metrics=metrics,
                        validation=validation_result,
                        model_name=selected_model,
                        fallback_used=False,
                        tokens_prompt=prompt_tokens,
                        tokens_completion=completion_tokens,
                        latency_ms=latency,
                        raw_output=raw_text_out,
                    )
                # Mark for fallback
                metrics = None

        except Exception as e:
            logger.warning("Primary model %s call failed: %s", self.vllm_model, str(e))
            if not allow_fallback:
                raise

        # Step 2: Route to Fallback Frontier Model if needed
        if metrics is None or (
            validation_result and not validation_result.passed and allow_fallback
        ):
            logger.info("Routing request to fallback frontier model: %s", self.frontier_model)
            fallback_used = True
            selected_model = self.frontier_model

            fb_json, fb_ptok, fb_ctok, fb_raw = await self._call_endpoint(
                client=self.fallback_client,
                model_name=self.frontier_model,
                prompt=text,
                use_vllm_guided=False,
            )
            prompt_tokens += fb_ptok
            completion_tokens += fb_ctok
            raw_text_out = fb_raw

            metrics = ExtractedFinancialMetrics.model_validate(fb_json)
            validation_result = self.validator.validate(metrics)

        latency = (time.perf_counter() - start_time) * 1000.0

        return ModelExtractionResult(
            metrics=metrics,
            validation=validation_result or ValidationDetail(passed=True),
            model_name=selected_model,
            fallback_used=fallback_used,
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            latency_ms=latency,
            raw_output=raw_text_out,
        )
