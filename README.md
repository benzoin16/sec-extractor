# sec-extractor (EdgarLENS)

SEC EDGAR filing extractor with a complete LLMOps pipeline (QLoRA fine-tuning, evaluation, serving, tracing, and active retraining).

## Architecture

```
ingest ──> parse & section ──> XBRL label join ──> training set ──> QLoRA fine-tune
                                                                          │
retrain <── failure mining <── CI eval gate <── Langfuse trace <── vLLM + FastAPI serve
```

## Setup

```bash
# Clone and install dependencies
pip install -e ".[dev]"

# Configure environment variables
cp .env.example .env
# Edit .env and supply your SEC_USER_AGENT ("YourName user@domain.com")
```

## Usage

### 1. Ingest Filings
Fetch 10-K and 10-Q filings from the EDGAR submissions API and cache raw documents to `data/raw`:

```bash
# CLI with tickers
python -m src.ingest --tickers AAPL MSFT GOOGL

# CLI with YAML configuration
python -m src.ingest --config configs/ingest.yaml

# Dry run inspection
python -m src.ingest --tickers AAPL --dry-run
```

### 2. Parse & Chunk Filings
Extract Item sections (Item 1, 1A, 7, 8, etc.), transform complex financial tables to clean GitHub-Flavored Markdown, and chunk into 2–4k token pieces without breaking table boundaries:

```bash
# Parse raw filings into processed chunks
python -m src.parse --input data/raw --output data/processed

# Or with configuration
python -m src.parse --config configs/parse.yaml
### 3. Extract XBRL Ground Truth Labels
Fetch `companyfacts` from EDGAR API and extract ground truth for the 7 concepts filtered strictly by accession number:

```bash
# Extract ground-truth labels for filings
python -m src.labels --tickers AAPL MSFT GOOGL

# Or with YAML configuration
python -m src.labels --config configs/ingest.yaml
```

### 4. Build Training Dataset & Split Manifest
Assemble chat-message instruction tuning examples, perform company and time holdout splits, verify zero-leakage, and generate dataset distribution reports:

```bash
python -m src.dataset --input data/processed --output data/processed
```

### 5. Evaluate Baseline Models
Run deterministic benchmarks against OpenAI-compatible model endpoints (Frontier API or base 7B models):

```bash
# Evaluate frontier baseline (e.g. GPT-4o-mini)
python -m src.eval --model gpt-4o-mini --base-url https://api.openai.com/v1

# Evaluate local or hosted base 7B model (vLLM / Ollama)
python -m src.eval --model Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8000/v1
```

### 6. Cloud GPU Fine-Tuning & Quantization
See [`docs/gpu_runbook.md`](docs/gpu_runbook.md) for full RunPod, Modal, and Colab recipes.

```bash
# 1. Train QLoRA adapter
python -m src.train.sft --config configs/train_qlora.yaml

# 2. Merge adapter into base weights
python -m src.train.merge_adapter --base-model Qwen/Qwen2.5-7B-Instruct --adapter-dir models/qwen2.5-7b-qlora --output-dir models/qwen2.5-7b-merged

# 3. 4-bit AWQ Quantization for vLLM
python -m src.train.quantize --model-path models/qwen2.5-7b-merged --output-dir models/qwen2.5-7b-awq
```

## Running Tests

```bash
pytest
```
