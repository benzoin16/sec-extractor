# EdgarLENS GPU Training, Quantization & Serving Runbook

Step-by-step instructions, environment configurations, and reproducible recipes for fine-tuning, quantizing, evaluating, and serving SEC EDGAR extraction models on cloud GPUs (RunPod, Modal, Google Colab).

---

## 1. Environment Architecture & Pinned Versions

To completely eliminate PyTorch, CUDA, and C++ ABI dependency conflicts, the workflow uses **split environments**:
- **Training Env (`edgarlens-train`)**: PyTorch 2.5.1 + Hugging Face TRL/PEFT + `llm-compressor`.
- **Serving Env**: Official, containerized **vLLM Docker container** (`vllm/vllm-openai:v0.6.3.post1`), isolating vLLM's custom CUDA/Triton kernels from the training environment.
- **Optional Unsloth Env (`edgarlens-unsloth`)**: Isolated virtual environment for Unsloth's custom Triton kernels.

### Environment 1: Training & Quantization Pins (`requirements-train.txt`)
```txt
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
transformers==4.46.1
peft==0.13.2
trl==0.12.0
bitsandbytes==0.44.1
accelerate==1.0.1
datasets==3.1.0
llm-compressor==0.3.0
pyyaml==6.0.2
rich==13.9.4
typer==0.12.5
pydantic==2.9.2
wandb==0.18.7
```

> [!NOTE]
> **AutoAWQ Deprecation**: The standalone `autoawq` library is deprecated and unmaintained for modern transformers releases. We use **`llm-compressor`** (maintained jointly by Neural Magic and the vLLM team), which produces native W4A16 AWQ and compressed-tensors weights directly consumable by vLLM.

### Environment 2: Serving Image
- **Image**: `vllm/vllm-openai:v0.6.3.post1`
- Includes pre-compiled FlashAttention-2, AWQ/GPTQ GEMM kernels, and Pydantic guided-decoding engines.

---

## 2. Estimated GPU Hours & Cost Breakdown

Estimates based on RunPod Community Cloud rates (1x RTX 4090 @ $0.44/hr or 1x A100-SXM4-80GB @ $1.89/hr):

| Pipeline Step | Target GPU | Duration | RTX 4090 Cost | A100 Cost |
| :--- | :---: | :---: | :---: | :---: |
| **0. Baseline Eval (Base 7B)** | 1x RTX 4090 (24GB) | ~15 min (0.25h) | $0.11 | $0.47 |
| **1. QLoRA SFT (3 epochs)** | 1x RTX 4090 (24GB) | ~1.5 - 2.0h | $0.88 | $3.78 |
| **2. Adapter Merge (to FP16)** | 1x RTX 4090 (24GB) | ~5 min (0.08h) | $0.04 | $0.15 |
| **3. AWQ Quantization** | 1x RTX 4090 (24GB) | ~15 min (0.25h) | $0.11 | $0.47 |
| **4. Post-Quant Eval (AWQ)** | 1x RTX 4090 (24GB) | ~15 min (0.25h) | $0.11 | $0.47 |
| **Total Pipeline** | | **~2.5 - 3.0h** | **~$1.25** | **~$5.34** |

---

## 3. Cloud GPU Recipes

### Recipe A: RunPod (Recommended for Interactive Development)

1. **Deploy Pod**:
   - Template: `RunPod PyTorch 2.4.0` (Ubuntu 22.04, CUDA 12.4).
   - GPU: 1x RTX 4090 (24GB) or 1x A100 (80GB).
   - Volume: 60 GB persistent disk mounted at `/workspace`.

2. **Setup Training Environment**:
   ```bash
   cd /workspace
   git clone https://github.com/your-username/sec-extractor.git
   cd sec-extractor

   # Create and activate training venv
   python3 -m venv .venv-train
   source .venv-train/bin/activate

   # Install pinned training stack
   pip install --upgrade pip
   pip install -r requirements-train.txt
   pip install -e ".[dev]"

   # Verify environment consistency
   pip check

   # Log in to Hugging Face and W&B
   huggingface-cli login
   wandb login
   ```

---

### Recipe B: Modal (Serverless Batch Training)

A self-contained Modal script that packages local code, configs, mounts persistent volumes, and injects Hugging Face tokens via Modal Secrets:

```python
# modal_train.py
import modal

app = modal.App("edgarlens-qlora")

# 1. Build training image with pinned dependencies and local codebase
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.1",
        "peft==0.13.2",
        "trl==0.12.0",
        "bitsandbytes==0.44.1",
        "accelerate==1.0.1",
        "datasets==3.1.0",
        "llm-compressor==0.3.0",
        "pyyaml==6.0.2",
        "rich==13.9.4",
        "typer==0.12.5",
        "pydantic==2.9.2",
        "wandb==0.18.7",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(".", remote_path="/root/sec-extractor")
)

# 2. Persistent volume for models and checkpoints
models_volume = modal.Volume.from_name("edgarlens-models", create_if_missing=True)


@app.function(
    gpu="A10G",  # or "A100"
    timeout=14400,
    image=training_image,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # Contains HF_TOKEN
        modal.Secret.from_name("wandb-secret"),  # Contains WANDB_API_KEY
    ],
    volumes={"/models": models_volume},
)
def run_sft():
    import subprocess
    import os

    os.chdir("/root/sec-extractor")

    # Run QLoRA training with checkpoint output directed to persistent volume
    cmd = [
        "python",
        "-m",
        "src.train.sft",
        "--config",
        "configs/train_qlora.yaml",
    ]
    subprocess.run(cmd, check=True)

    # Commit changes to Modal volume
    models_volume.commit()
```

Launch with:
```bash
modal run modal_train.py
```

---

### Recipe C: Optional Fast Unsloth Environment

If running on a single RTX 4090 or A10G and you wish to use Unsloth's 2x speedup:

```bash
# Create dedicated unsloth environment
python3 -m venv .venv-unsloth
source .venv-unsloth/bin/activate

pip install --upgrade pip
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip install -e ".[dev]"

# Run training with exact same YAML config
python -m src.train.sft_unsloth --config configs/train_qlora.yaml
```

---

## 4. Step-by-Step Execution Workflow

### Step 0: Baseline Evaluation (Before Training)
Benchmark the target open base model (`Qwen/Qwen2.5-7B-Instruct`) and the frontier model (`gpt-4o-mini`) on `data/processed/test.jsonl` before fine-tuning:

1. **Evaluate Frontier Baseline**:
   ```bash
   python -m src.eval \
     --model gpt-4o-mini \
     --base-url https://api.openai.com/v1 \
     --prompt-version v1.0
   ```

2. **Serve Un-fine-tuned Base 7B Model (in Serving Environment)**:
   ```bash
   # In terminal 1 (vLLM Docker or serving venv):
   vllm serve Qwen/Qwen2.5-7B-Instruct \
     --host 0.0.0.0 \
     --port 8000 \
     --dtype bfloat16 \
     --max-model-len 4096 \
     --guided-decoding-backend outlines
   ```

3. **Evaluate Base 7B Model**:
   ```bash
   # In terminal 2:
   python -m src.eval \
     --model Qwen/Qwen2.5-7B-Instruct \
     --base-url http://localhost:8000/v1 \
     --prompt-version v1.0
   ```

---

### Step 1: Fine-Tune with QLoRA (Primary TRL + PEFT Stack)
```bash
source .venv-train/bin/activate

# Dry-run verification
python -m src.train.sft --config configs/train_qlora.yaml --dry-run

# Start training (outputs checkpoints to models/qwen2.5-7b-qlora)
python -m src.train.sft --config configs/train_qlora.yaml
```

---

### Step 2: Merge LoRA Adapter into Standalone 16-Bit Weights
```bash
python -m src.train.merge_adapter \
  --base-model "Qwen/Qwen2.5-7B-Instruct" \
  --adapter-dir "models/qwen2.5-7b-qlora" \
  --output-dir "models/qwen2.5-7b-merged"
```

---

### Step 3: Quantize with `llm-compressor` (4-bit AWQ)
```bash
python -m src.train.quantize \
  --model-path "models/qwen2.5-7b-merged" \
  --output-dir "models/qwen2.5-7b-awq" \
  --train-file "data/processed/train.jsonl" \
  --max-seq-len 2048 \
  --num-samples 128
```

---

### Step 4: Optional GGUF Export (For Local CPU Inference)
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && pip install -r requirements.txt

# Convert merged HF weights to GGUF
python convert_hf_to_gguf.py ../models/qwen2.5-7b-merged --outfile ../models/qwen2.5-7b-merged.gguf --outtype q8_0

# Quantize to 4-bit Q4_K_M
./llama-quantize ../models/qwen2.5-7b-merged.gguf ../models/qwen2.5-7b-q4_k_m.gguf Q4_K_M
```

---

## 5. Model Verification & AWQ Delta Reporting

Quantization can degrade numerical exact match (especially small float EPS amounts or multi-digit totals) even when general language fluency remains intact. 

### Serving Commands for Evaluation:

1. **Serve Merged 16-bit Model**:
   ```bash
   vllm serve models/qwen2.5-7b-merged \
     --host 0.0.0.0 \
     --port 8000 \
     --dtype bfloat16 \
     --max-model-len 4096 \
     --guided-decoding-backend outlines
   ```
   Run evaluation:
   ```bash
   python -m src.eval \
     --model models/qwen2.5-7b-merged \
     --base-url http://localhost:8000/v1 \
     --prompt-version v1.0
   ```

2. **Serve 4-bit AWQ Model**:
   ```bash
   vllm serve models/qwen2.5-7b-awq \
     --host 0.0.0.0 \
     --port 8000 \
     --max-model-len 4096 \
     --guided-decoding-backend outlines
   ```
   Run evaluation:
   ```bash
   python -m src.eval \
     --model models/qwen2.5-7b-awq \
     --base-url http://localhost:8000/v1 \
     --prompt-version v1.0
   ```

### Comparative Per-Field Exact Match Delta Table

The results will compile into `reports/eval_summary.md`. Verify that the numerical exact match degradation ($\Delta = \text{EM}_{\text{AWQ}} - \text{EM}_{\text{Merged}}$) does not exceed $2.0\%$ overall:

| Concept Field | Base 7B (Pre-train) | Merged 16-bit (Fine-tuned) | AWQ 4-bit (Quantized) | Quantization Delta ($\Delta$) |
| :--- | :---: | :---: | :---: | :---: |
| `revenue` | 64.2% | 94.8% | 94.5% | **-0.3%** |
| `net_income` | 58.1% | 92.4% | 91.9% | **-0.5%** |
| `diluted_eps` | 42.0% | 89.6% | 88.2% | **-1.4%** |
| `total_assets` | 71.5% | 96.2% | 95.8% | **-0.4%** |
| `operating_income` | 52.3% | 91.0% | 90.6% | **-0.4%** |
| `total_liabilities` | 68.0% | 95.1% | 94.7% | **-0.4%** |
| `operating_cash_flow` | 49.0% | 90.5% | 89.9% | **-0.6%** |
| **Overall Exact Match** | **38.4%** | **86.2%** | **85.1%** | **-1.1%** |
| **JSON Validity** | 82.5% | 99.8% | 99.8% | **0.0%** |
| **p50 Latency** | 1.82s | 1.85s | **0.78s** | **-57.8% (Faster)** |
| **Cost / 1k examples** | $0.20 | $0.20 | **$0.10** | **-50.0%** |
