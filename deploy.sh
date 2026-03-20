#!/bin/bash
# Inference Engine Deployment Script
# RunPod/Vast.ai için environment variable'ları ayarla

set -e

echo "🚀 Inference Engine Deployment"
echo "================================"

# Region bucket seçimi (sunucu lokasyonuna göre)
REGION="${REGION:-apac}"

case $REGION in
  apac)  R2_MODELS_BUCKET="visgate-deploy-api-models-apac" ;;
  eeur)  R2_MODELS_BUCKET="visgate-deploy-api-models-eeur" ;;
  enam)  R2_MODELS_BUCKET="visgate-deploy-api-models-enam" ;;
  oc)    R2_MODELS_BUCKET="visgate-deploy-api-models-oc" ;;
  weur)  R2_MODELS_BUCKET="visgate-deploy-api-models-weur" ;;
  wnam)  R2_MODELS_BUCKET="visgate-deploy-api-models-wnam" ;;
  *)     R2_MODELS_BUCKET="visgate-deploy-api-models-apac" ;;
esac

echo "📍 Region: $REGION"
echo "📦 Model Bucket: $R2_MODELS_BUCKET"

# Docker Compose
cat > docker-compose.yml << EOF
version: '3.8'

services:
  inference-engine:
    build: .
    image: visgate/inference-engine:latest
    container_name: visgate-inference
    ports:
      - "8000:8000"
    environment:
      - HF_HUB_TOKEN=\${HF_HUB_TOKEN}
      - R2_ENDPOINT_URL=\${R2_ENDPOINT_URL}
      - R2_ACCESS_KEY_ID=\${R2_ACCESS_KEY_ID}
      - R2_SECRET_ACCESS_KEY=\${R2_SECRET_ACCESS_KEY}
      - R2_MODELS_BUCKET=\${R2_MODELS_BUCKET}
      - R2_INPUT_BUCKET=visgate-deploy-api-inference-input
      - R2_OUTPUT_BUCKET=visgate-deploy-api-inference-output
    volumes:
      - ./models:/root/.cache/huggingface/hub
      - /tmp:/app/temp
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF

echo "✅ docker-compose.yml oluşturuldu"
echo ""
echo "📝 Kullanım:"
echo "   1. Environment değişkenlerini ayarla:"
echo "      export HF_HUB_TOKEN=..."
echo "      export R2_ENDPOINT_URL=https://..."
echo "      export R2_ACCESS_KEY_ID=..."
echo "      export R2_SECRET_ACCESS_KEY=..."
echo "      export REGION=apac  # apac, eeur, enam, oc, weur, wnam"
echo ""
echo "   2. Container'ı başlat:"
echo "      docker-compose up -d"
echo ""
echo "   3. API'yi test et:"
echo "      curl http://localhost:8000/health"
echo "      curl http://localhost:8000/status"
