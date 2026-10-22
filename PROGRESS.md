# EdgarLENS: SEC Filing Extractor & LLMOps Pipeline

## Project Status & Roadmap

### Phase 1: Ingest + Parse (Completed)
- [x] **Repo Skeleton**: `pyproject.toml`, ruff lint configuration, `.gitignore`, `.env.example`.
- [x] **Configs**: `configs/ingest.yaml`, `configs/parse.yaml`.
- [x] **`src.ingest`**:
  - `CIKResolver`: CIK lookup and local caching (`company_tickers.json`).
  - `SlidingWindowRateLimiter`: Sliding-window throttle ensuring compliance with EDGAR 10 req/s fair access policy.
  - `FilingFetcher`: Multi-year submission discovery for 10-K and 10-Q filings with pagination support.
  - `Downloader`: Primary HTML downloader with idempotent local storage in `data/raw/{cik}/{accession_nodash}/primary.html`.
  - Ingest CLI: `python -m src.ingest --tickers AAPL MSFT ...`
- [x] **`src.parse`**:
  - `TableParser`: Converts complex financial tables with `colspan`, `rowspan`, inline XBRL tags, and detached currency symbols into clean GitHub-Flavored Markdown.
  - `Sectioner`: Detects 10-K and 10-Q Item boundaries (Item 1, 1A, 7, 8, etc.) while filtering out Table of Contents (TOC) links and anchor tables.
  - `Chunker`: Token-bounded chunker (2–4k tokens via `tiktoken`) with strict table atomicity (tables are never split across chunks).
  - Parse CLI: `python -m src.parse --input data/raw --output data/processed`
- [x] **Test Suite**: 13 unit tests across ingest, table parsing, sectioning, and chunking.

---

### Phase 2: XBRL Labels + Training Dataset (Completed)
- [x] **Ground Truth Fetcher**: `CompanyFactsFetcher` fetches and caches `companyfacts/CIK{cik}.json` from `data.sec.gov` with rate limiting and local caching.
- [x] **Concept Mapping Config**: `configs/xbrl_map.yaml` covering the 7 target concepts with tag fallback cascades and units:
  - `revenue`, `net_income`, `diluted_eps`, `total_assets`, `operating_income`, `total_liabilities`, `operating_cash_flow`.
- [x] **Restatement & Period Handling**:
  - Strict accession number filtering (`accn == meta.accession_number`) prevents lookahead bias from subsequent restatements.
  - Duration classification for 10-Qs into explicit `quarter` (3-month) and `ytd` (6/9-month) periods.
  - Explicit period types emitted alongside target values (`annual`, `quarter`, `ytd`, `instant`).
  - Liabilities fallback calculation: sums `LiabilitiesCurrent + LiabilitiesNoncurrent` when both exist, tagged with `is_derived=True`.
- [x] **Numeric Presence & Matcher**:
  - Multi-scale matching ($1\times, 10^3, 10^6, 10^9$), 2-decimal EPS float tolerance, and parenthesized negative values `(1,234)`.
  - Derived liabilities only emit a non-null target if the calculated sum itself appears in the chunk text.
  - Ambiguity rule: skips examples if identical values could be either quarter or YTD, tracking `skipped_ambiguous_count`.
- [x] **Negative Example Sampling**:
  - 15% negative ratio configurable in `configs/dataset.yaml`.
  - Balanced 50/50 mix of hard negatives (chunks with tables and unrelated numbers) and text negatives (narrative, Item 1A).
- [x] **Company + Time Splitting & Leak Assertions**:
  - `DatasetSplitter` holds out test companies (e.g. `JPM`, `AMZN`) across all years and holds out `val_fiscal_year` (2024) for training companies.
  - Writes `split_manifest.json` and executes `assert_no_leak` testing company, accession, and temporal isolation.
- [x] **JSONL Output & Stats Report**:
  - Formats training data in standard chat-message format: `train.jsonl`, `val.jsonl`, `test.jsonl`.
  - Generates comprehensive markdown stats report in `reports/dataset_stats.md`.
- [x] **CLIs**:
  - `python -m src.labels --tickers AAPL MSFT ...`
  - `python -m src.dataset --input data/processed --output data/processed`
- [x] **Test Suite**: 24 tests passing with 89% total code coverage and 0 lint errors.

---

### Phase 3: Baselines + Fine-Tuning Scripts + Eval Harness (Completed)
- [x] **Evaluation Harness (`src.eval`)**:
  - Deterministic evaluation runner (`temperature=0.0`, `seed=42`) with asynchronous concurrency control for any OpenAI-compatible API (Frontier APIs, vLLM, Ollama).
  - Strict numeric normalization with scale resolution ($1\times, 10^3, 10^6, 10^9$), 0.5% rounding tolerance, strict sign check, and $\pm 0.02$ EPS tolerance.
  - JSON validity rate, field-level exact match, overall exact match, null accuracy, and hallucination rate tracking.
  - Performance tracking: p50/p95 latency, tokens/sec, and cost per 1k examples.
  - Output results to `reports/eval_*.json` and comparative markdown summary in `reports/eval_summary.md`.
  - Prompt versioning schema: `configs/prompts/v1_zero_shot.yaml`.
  - CLI: `python -m src.eval --model <model> --base-url <url> --prompt-version v1.0`.
- [x] **QLoRA Fine-Tuning Scripts (`src.train`)**:
  - `src/train/collator.py`: `AssistantOnlyDataCollator` masking prompt tokens with `-100` so loss is computed exclusively on assistant targets.
  - `src/train/sft.py`: Primary SFT script with `trl.SFTTrainer` + `peft` + 4-bit `BitsAndBytesConfig`, W&B / MLflow logging, evaluation checkpoints.
  - `src/train/sft_unsloth.py`: Fast single-GPU launcher using `unsloth.FastLanguageModel` sharing the exact same YAML schema.
  - `src/train/merge_adapter.py`: Standalone 16-bit weight exporter merging LoRA adapter into base model.
  - `src/train/quantize.py`: 4-bit AWQ GEMM post-training quantization for production vLLM serving.
  - Configs: `configs/train_qlora.yaml` (default Qwen2.5-7B-Instruct) and `configs/train_qlora_llama.yaml` (alternate Llama-3.1-8B-Instruct).
- [x] **GPU Runbook (`docs/gpu_runbook.md`)**:
  - Exact pinned library versions (`torch==2.5.1`, `transformers==4.46.1`, `trl==0.12.0`, `peft==0.13.2`, `autoawq==0.2.7.post2`, `vllm==0.6.3.post1`).
  - RunPod, Modal, and Google Colab step-by-step recipes.
  - GGUF conversion instructions for local CPU inference.
  - AWQ exact-match delta evaluation verification procedure.
- [x] **Test Suite**: 32 unit tests passing with 0 lint errors.

---

### Phase 4: Serving + Observability + CI Gate (Completed)
- [x] **FastAPI Gateway (`src.serve.app`)**:
  - `POST /extract`: Accepts raw chunk text or accession number (triggers automatic fetch -> parse -> chunk with caching).
  - `GET /health`: Model status, vLLM backend reachability, and frontier fallback availability.
  - `GET /metrics`: Prometheus formatted metrics export.
  - Pydantic models: `ExtractRequest`, `ExtractResponse`, `ExtractedFinancialMetrics`, `ValidationDetail`.
- [x] **vLLM Schema-Guided Decoding (`src.serve.client`)**:
  - Targets `vllm >= 0.6.3` with OpenAI-compatible `response_format={"type": "json_schema", ...}` and `extra_body={"guided_json": ...}` backwards compatibility.
- [x] **Financial Post-Validation Sanity Engine (`src.serve.validator`)**:
  - Scale plausibility checks (flags unscaled dollar amounts < $10k and out-of-bound EPS).
  - Margin sanity (Net Income & Operating Income vs Revenue).
  - Sign consistency & implied share count checks ($\text{Net Income} / \text{EPS}$).
  - Balance sheet checks (Assets vs Liabilities sanity).
  - Automated fallback routing to frontier model (`gpt-4o-mini`) on validation failure or vLLM timeout.
- [x] **Langfuse Observability & Tracing (`src.serve.tracing`)**:
  - Traces input text length, output JSON, token counts, latency, validation status, fallback flag, model version.
  - Configurable privacy redaction flag (`LANGFUSE_MASK_INPUT_OUTPUT=true`) to keep sensitive financial texts out of third-party logs.
- [x] **Docker Compose Stack (`docker-compose.yml`)**:
  - Dual profiles:
    - `gpu` profile: `vllm` (NVIDIA container) + `api` + `prometheus` + `grafana` + `langfuse`.
    - `cpu` profile: `mock-model` (OpenAI-compatible server) + `api` + `prometheus` + `grafana` + `ui` (runs on standard laptops).
  - Pre-provisioned Prometheus (`configs/prometheus.yml`) and Grafana dashboard (`configs/grafana/dashboards/edgarlens_metrics.json`) tracking P50/P95 latency, error rate, fallback rate, and token throughput.
- [x] **GitHub Actions CI & Eval Quality Gate (`.github/workflows/ci.yml`)**:
  - Linting (`ruff check`) + unit tests (`pytest`).
  - Automated evaluation gate running against a fixed regression dataset (`tests/fixtures/eval_regression_fixture.jsonl`).
  - Strict quality gates in `configs/eval_thresholds.yaml` (minimum JSON validity, exact match, maximum hallucination rate, and P95 latency).
  - Cached dependencies and masked environment secrets.
- [x] **Interactive Streamlit Demo UI (`src.serve.demo`)**:
  - Paste raw filing text or enter accession number with ticker.
  - Displays formatted metric tables, latency gauges, validation badges, fallback flags, and raw JSON.
- [x] **Test Suite**: 47 unit and integration tests passing with 72% coverage and 0 lint errors (`test_validator.py`, `test_api.py`, `test_smoke_compose.py`).

---

### Phase 5: Failure Mining & Retraining Loop (Upcoming)
- [ ] Failure mining from production traces and negative validation logs.
- [ ] Documented retraining iteration showing accuracy improvement.

