# Visgate AI Inference Engine - Production Docker Image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DOWNLOAD_TIMEOUT=300

# Sistem güncellemesi ve gereksinimler
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev \
    ffmpeg libsm6 libxext6 libgl1 libsndfile1 git wget curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Cache dizini
RUN pip3 install --no-cache-dir --upgrade pip wheel

WORKDIR /app

# Kütüphaneleri kur (önce heavy deps)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY app ./app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Varsayılan komut (override edilebilir)
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
