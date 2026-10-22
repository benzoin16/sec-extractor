"""FastAPI application gateway for SEC financial metric extraction."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

from src.serve.client import ExtractionInferenceClient
from src.serve.metrics import (
    FALLBACKS_TOTAL,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    TOKENS_TOTAL,
    VALIDATION_FAILURES_TOTAL,
    get_metrics_payload,
)
from src.serve.pipeline import AccessionPipeline
from src.serve.schemas import (
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
)
from src.serve.tracing import LangfuseTracer

logger = logging.getLogger("src.serve")
logging.basicConfig(level=logging.INFO)

# Global clients initialized on startup
inference_client: ExtractionInferenceClient | None = None
accession_pipeline: AccessionPipeline | None = None
tracer: LangfuseTracer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize clients and resources on server startup."""
    global inference_client, accession_pipeline, tracer
    logger.info("Initializing EdgarLENS Serving Gateway...")

    inference_client = ExtractionInferenceClient()
    accession_pipeline = AccessionPipeline()
    tracer = LangfuseTracer()

    yield

    logger.info("Shutting down EdgarLENS Serving Gateway...")


app = FastAPI(
    title="EdgarLENS Financial Extraction Gateway",
    description="Schema-guided fine-tuned LLM extraction with validation sanity checks and frontier fallback.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit / Gradio web UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return health status and backend reachability."""
    client = inference_client or ExtractionInferenceClient()
    vllm_ok = False
    try:
        # Check if primary model endpoint is responding
        models = await client.primary_client.models.list()
        vllm_ok = len(models.data) >= 0
    except Exception:
        vllm_ok = False

    has_fallback = bool(os.getenv("OPENAI_API_KEY"))

    status_str = "healthy" if vllm_ok else ("degraded" if has_fallback else "unhealthy")

    return HealthResponse(
        status=status_str,
        vllm_reachable=vllm_ok,
        model_name=client.vllm_model,
        fallback_available=has_fallback,
    )


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """Export Prometheus metrics."""
    data, content_type = get_metrics_payload()
    return Response(content=data, media_type=content_type)


@app.post("/extract", response_model=ExtractResponse)
async def extract_metrics(request: ExtractRequest) -> ExtractResponse:
    """Extract 7 core financial concepts from raw text or SEC accession number."""
    start_time = time.perf_counter()
    client = inference_client or ExtractionInferenceClient()
    pipeline = accession_pipeline or AccessionPipeline()
    active_tracer = tracer or LangfuseTracer()

    # 1. Resolve Text (either direct or via accession cache/fetch)
    input_text = request.text
    if not input_text and request.accession_number:
        try:
            input_text = await pipeline.get_text_for_accession(
                accession_number=request.accession_number,
                ticker=request.ticker,
            )
        except Exception as e:
            logger.error(
                "Failed to retrieve filing for accession %s: %s", request.accession_number, e
            )
            REQUESTS_TOTAL.labels(endpoint="/extract", model="none", status="error").inc()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not retrieve or parse filing for accession {request.accession_number}: {e}",
            ) from e

    if not input_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No text available for extraction.",
        )

    # 2. Run Model Inference with Sanity Checks & Fallback
    try:
        result = await client.extract(
            text=input_text,
            allow_fallback=request.allow_fallback,
            strict_validation=request.strict_validation,
        )
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        REQUESTS_TOTAL.labels(endpoint="/extract", model=client.vllm_model, status="error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {e}",
        ) from e

    status_tag = "success" if result.validation.passed else "validation_warning"

    # 3. Record Prometheus Metrics
    REQUEST_LATENCY.labels(endpoint="/extract", model=result.model_name, status=status_tag).observe(
        result.latency_ms / 1000.0
    )
    REQUESTS_TOTAL.labels(endpoint="/extract", model=result.model_name, status=status_tag).inc()
    TOKENS_TOTAL.labels(model=result.model_name, type="prompt").inc(result.tokens_prompt)
    TOKENS_TOTAL.labels(model=result.model_name, type="completion").inc(result.tokens_completion)

    if result.fallback_used:
        FALLBACKS_TOTAL.labels(reason="validation_failed_or_primary_offline").inc()

    for err in result.validation.errors:
        check_name = err.split(":")[0].replace(" ", "_")[:32]
        VALIDATION_FAILURES_TOTAL.labels(check_type=check_name).inc()

    # 4. Record Langfuse Trace
    trace_id = active_tracer.trace_extraction(
        trace_id=None,
        input_text=input_text,
        output_data=result.metrics.model_dump(),
        model_name=result.model_name,
        latency_ms=result.latency_ms,
        tokens_prompt=result.tokens_prompt,
        tokens_completion=result.tokens_completion,
        validation=result.validation,
        fallback_used=result.fallback_used,
    )

    total_latency_ms = (time.perf_counter() - start_time) * 1000.0

    return ExtractResponse(
        metrics=result.metrics,
        validation=result.validation,
        latency_ms=round(total_latency_ms, 2),
        model=result.model_name,
        fallback_used=result.fallback_used,
        trace_id=trace_id,
        tokens_prompt=result.tokens_prompt,
        tokens_completion=result.tokens_completion,
    )
