"""Lightweight OpenAI-compatible Mock Inference Server for CPU-only testing and demos."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

logger = logging.getLogger("src.serve.mock_server")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="EdgarLENS Mock vLLM Server")


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1700000000
    owned_by: str = "edgarlens"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


@app.get("/v1/models")
async def list_models() -> ModelList:
    return ModelList(
        data=[
            ModelCard(id="qwen2.5-7b-instruct-sec"),
            ModelCard(id="mock-vllm-7b"),
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, Any]:
    """Return mock schema-constrained financial extraction JSON."""
    payload = await request.json()
    messages = payload.get("messages", [])
    user_text = ""
    for m in messages:
        if m.get("role") == "user":
            user_text = m.get("content", "")

    # Check for test control flags
    if "TRIGGER_INVALID_JSON" in user_text:
        content_str = "Invalid JSON {{{ broken"
    elif "FAIL_SANITY_SCALE" in user_text:
        # Deliberately unscaled (dollar value under threshold)
        content_str = json.dumps(
            {
                "revenue": {"value": 383.28, "period_type": "annual"},
                "net_income": {"value": 96.99, "period_type": "annual"},
                "diluted_eps": {"value": 6.13, "period_type": "annual"},
                "total_assets": None,
                "operating_income": None,
                "total_liabilities": None,
                "operating_cash_flow": None,
            }
        )
    elif "FAIL_SANITY_SIGN" in user_text:
        # Sign mismatch: positive income but negative EPS
        content_str = json.dumps(
            {
                "revenue": {"value": 383285000000.0, "period_type": "annual"},
                "net_income": {"value": 96995000000.0, "period_type": "annual"},
                "diluted_eps": {"value": -6.13, "period_type": "annual"},
                "total_assets": None,
                "operating_income": None,
                "total_liabilities": None,
                "operating_cash_flow": None,
            }
        )
    else:
        # Parse or default realistic financial metrics
        rev_match = re.search(r"revenue[^\d]*(\d+(?:,\d{3})*(?:\.\d+)?)", user_text, re.IGNORECASE)
        rev_val = 383285000000.0
        if rev_match:
            try:
                raw_num = float(rev_match.group(1).replace(",", ""))
                if "billion" in user_text.lower():
                    raw_num *= 1_000_000_000
                elif "million" in user_text.lower() or raw_num < 10000:
                    raw_num *= 1_000_000
                rev_val = raw_num
            except Exception:
                pass

        content_str = json.dumps(
            {
                "revenue": {"value": rev_val, "period_type": "annual"},
                "net_income": {"value": round(rev_val * 0.25, 2), "period_type": "annual"},
                "diluted_eps": {"value": 6.13, "period_type": "annual"},
                "total_assets": {"value": round(rev_val * 0.95, 2), "period_type": "instant"},
                "operating_income": {"value": round(rev_val * 0.30, 2), "period_type": "annual"},
                "total_liabilities": {"value": round(rev_val * 0.70, 2), "period_type": "instant"},
                "operating_cash_flow": {"value": round(rev_val * 0.28, 2), "period_type": "annual"},
            }
        )

    prompt_toks = max(20, len(user_text) // 4)
    completion_toks = len(content_str) // 4

    return {
        "id": f"chatcmpl-mock-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", "mock-vllm-7b"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content_str},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": completion_toks,
            "total_tokens": prompt_toks + completion_toks,
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="EdgarLENS Mock vLLM Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
