FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Sistem Gereksinimleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg libsm6 libxext6 libgl1 libsndfile1 git wget curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

WORKDIR /app

# Kütüphaneleri kur
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY app /app/app

# Çalıştırma senaryosuna göre komut (Serverless vb. için ezilebilir)
CMD ["bash", "-c", "cd app && uvicorn main:app --host 0.0.0.0 --port 8000"]
