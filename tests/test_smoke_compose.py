"""Smoke tests for Docker Compose configuration and mock inference execution."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from src.serve.mock_server import app as mock_app


def test_docker_compose_syntax_and_profiles() -> None:
    """Validate docker-compose.yml structure, service definitions, and profile separation."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists(), "docker-compose.yml must exist at repository root"

    with open(compose_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    services = config.get("services", {})
    assert "api" in services
    assert "mock-model" in services
    assert "vllm" in services
    assert "prometheus" in services
    assert "grafana" in services
    assert "ui" in services

    # CPU Profile check (Laptop runnable without GPU)
    mock_service = services["mock-model"]
    assert "cpu" in mock_service.get("profiles", [])

    # GPU Profile check
    vllm_service = services["vllm"]
    assert "gpu" in vllm_service.get("profiles", [])
    assert "deploy" in vllm_service or "runtime" in vllm_service

    # Check Prometheus configuration volume
    prom_service = services["prometheus"]
    assert any("prometheus.yml" in str(v) for v in prom_service.get("volumes", []))


def test_mock_model_server_generates_valid_schema() -> None:
    """Verify that the CPU mock model generates valid financial extraction completions."""
    client = TestClient(mock_app)

    # 1. Test /v1/models
    models_res = client.get("/v1/models")
    assert models_res.status_code == 200
    data = models_res.json()
    assert len(data["data"]) >= 1

    # 2. Test /v1/chat/completions realistic response
    payload = {
        "model": "qwen2.5-7b-instruct-sec",
        "messages": [
            {
                "role": "user",
                "content": "Total revenue: $ 383,285 million, net income: $ 96,995 million",
            }
        ],
    }
    chat_res = client.post("/v1/chat/completions", json=payload)
    assert chat_res.status_code == 200
    chat_data = chat_res.json()
    assert "choices" in chat_data
    content = chat_data["choices"][0]["message"]["content"]

    import json

    parsed = json.loads(content)
    assert "revenue" in parsed
    assert parsed["revenue"]["value"] == 383_285_000_000.0


def test_mock_model_server_test_flags() -> None:
    """Verify that mock model triggers deliberate sanity failure flags when requested."""
    client = TestClient(mock_app)

    # Unscaled test flag
    res_scale = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "FAIL_SANITY_SCALE test"}]},
    )
    assert res_scale.status_code == 200
    import json

    data_scale = json.loads(res_scale.json()["choices"][0]["message"]["content"])
    assert data_scale["revenue"]["value"] == 383.28

    # Sign mismatch flag
    res_sign = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "FAIL_SANITY_SIGN test"}]},
    )
    assert res_sign.status_code == 200
    data_sign = json.loads(res_sign.json()["choices"][0]["message"]["content"])
    assert data_sign["revenue"]["value"] > 0
    assert data_sign["diluted_eps"]["value"] < 0
