FROM python:3.11-slim

# libgomp1 is required by faiss-cpu for OpenMP support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv is a fast Rust-based package installer (10-100x faster than pip)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install CPU-only PyTorch first. The default `torch` from PyPI bundles CUDA
# binaries (~5GB uncompressed) that are useless on a CPU-only server.
RUN uv pip install --system --no-cache "torch==2.1.2" \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.prod.txt .
RUN uv pip install --system --no-cache -r requirements.prod.txt

# Pre-download the embedding model into the image so cold starts are instant.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-base-v2')"

# Copy application code and pre-computed data
COPY raglens/ raglens/
COPY dashboard/ dashboard/
COPY data/ data/

EXPOSE 8000

CMD ["uvicorn", "dashboard.api:app", "--host", "0.0.0.0", "--port", "8000"]
