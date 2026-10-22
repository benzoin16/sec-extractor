FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging specifications
COPY pyproject.toml README.md ./

# Install python dependencies including serve extras
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[serve]"

# Copy source code and configurations
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000 8001 8501

CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
