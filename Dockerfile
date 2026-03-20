# Visgate AI Inference Engine - Production Docker Image
# Using pre-built PyTorch image to save disk space
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=300 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev \
    ffmpeg libsm6 libxext6 libgl1 libsndfile1 git curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Install PyTorch from pre-built wheels (smaller download)
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 \
    torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118 \
    torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other ML dependencies
RUN pip3 install --no-cache-dir \
    transformers==4.36.0 \
    diffusers==0.25.0 \
    accelerate==0.25.0 \
    scipy==1.11.4 \
    huggingface_hub==0.20.2 \
    boto3==1.34.0 \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic==2.5.3 \
    python-multipart==0.0.6

# Copy app code
COPY app ./app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
