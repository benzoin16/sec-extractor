"""Langfuse tracing integration with configurable privacy / data redaction."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from src.serve.schemas import ValidationDetail

logger = logging.getLogger(__name__)

# Config flags
MASK_INPUT_OUTPUT = os.getenv("LANGFUSE_MASK_INPUT_OUTPUT", "false").lower() in ("true", "1", "yes")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


class LangfuseTracer:
    """Manages request-level tracing with Langfuse, supporting privacy redaction and no-op fallback."""

    def __init__(
        self,
        public_key: str | None = LANGFUSE_PUBLIC_KEY,
        secret_key: str | None = LANGFUSE_SECRET_KEY,
        host: str = LANGFUSE_HOST,
        mask_input_output: bool = MASK_INPUT_OUTPUT,
    ) -> None:
        self.mask_input_output = mask_input_output
        self.enabled = bool(public_key and secret_key)
        self._client: Any = None

        if self.enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                logger.info(
                    "Langfuse tracing initialized (host: %s, mask_io: %s)", host, mask_input_output
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize Langfuse client: %s. Running in no-op mode.", e
                )
                self.enabled = False
        else:
            logger.info("Langfuse keys not configured. Running tracer in no-op mode.")

    def trace_extraction(
        self,
        trace_id: str | None,
        input_text: str,
        output_data: Any,
        model_name: str,
        latency_ms: float,
        tokens_prompt: int,
        tokens_completion: int,
        validation: ValidationDetail,
        fallback_used: bool,
    ) -> str:
        """Log a complete extraction run to Langfuse, respecting redaction settings."""
        active_id = trace_id or f"trace_{uuid.uuid4().hex[:12]}"

        # Prepare payload respecting privacy / compliance settings
        if self.mask_input_output:
            traced_input = f"[REDACTED_INPUT: {len(input_text)} chars]"
            traced_output = {
                "status": "redacted",
                "fields_present": list(output_data.keys()) if isinstance(output_data, dict) else [],
            }
        else:
            traced_input = input_text
            traced_output = output_data

        metadata = {
            "validation_passed": validation.passed,
            "validation_errors": validation.errors,
            "validation_warnings": validation.warnings,
            "fallback_used": fallback_used,
            "latency_ms": latency_ms,
            "mask_io": self.mask_input_output,
        }

        if not self.enabled or self._client is None:
            return active_id

        try:
            trace = self._client.trace(
                id=active_id,
                name="sec_metric_extraction",
                metadata=metadata,
                tags=[
                    "fallback" if fallback_used else "primary",
                    "valid" if validation.passed else "invalid",
                ],
            )

            trace.generation(
                name="llm_generation",
                model=model_name,
                input=traced_input,
                output=traced_output,
                usage={
                    "prompt_tokens": tokens_prompt,
                    "completion_tokens": tokens_completion,
                    "total_tokens": tokens_prompt + tokens_completion,
                },
                metadata=metadata,
            )
            # Flush asynchronously in background
            self._client.flush()
        except Exception as e:
            logger.warning("Failed to record trace to Langfuse: %s", e)

        return active_id
