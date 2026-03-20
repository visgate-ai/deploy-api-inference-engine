#!/bin/bash
# Vast.ai Serverless Deployment Script
# Bu script, API'yi Vast.ai GPU sunucusunda çalıştırır

set -e

# Renkli çıktı
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Vast.ai Deployment${NC}"
echo "================================"

# Configuration
IMAGE_NAME="visgate-deploy-api:${1:-latest}"
VAST_API_KEY="${VAST_API_KEY:-}"

# SSH key oluştur (eğer yoksa)
SSH_KEY_PATH="$HOME/.ssh/vastai_key"
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo -e "${YELLOW}🔑 SSH Key oluşturuluyor...${NC}"
    ssh-keygen -t ed25519 -f "$SSH_KEY_PATH" -N ""
fi

# GPU tekliflerini ara
echo -e "${YELLOW}🔍 GPU Teklifleri Araştırılıyor...${NC}"

if [ -n "$VAST_API_KEY" ]; then
    # Mevcut teklifleri listele
    curl -s -H "Authorization: Bearer $VAST_API_KEY" \
        "https://console.vast.ai/api/v0/bids?limit=10" | jq '.offers[] | {id: .id, gpu: .gpu_name, num_gpus: .num_gpus, price: .dph_total, inet_down: .inet_down, inet_up: .inet_up}'
    
    # Örnek: A100 veya RTX 4090 bul
    echo -e "${YELLOW}📋 Önerilen GPU'lar:${NC}"
    echo "  - A100 (40GB) - Büyük modeller için"
    echo "  - A6000 (48GB) - Orta-büyük modeller için"
    echo "  - RTX 4090 (24GB) - Küçük-orta modeller için"
fi

# Docker compose ile başlatma
echo -e "${YELLOW}🐳 Docker Compose ile Başlatma${NC}"

cat > docker-compose.vastai.yml << 'EOF'
version: '3.8'

services:
  ai-api:
    build: .
    image: visgate-deploy-api:latest
    container_name: visgate-ai-worker
    ports:
      - "8000:8000"
    environment:
      - HF_HUB_TOKEN=${HF_HUB_TOKEN}
      - R2_ENDPOINT_URL=${R2_ENDPOINT_URL}
      - R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID}
      - R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY}
      - R2_INPUT_BUCKET_NAME=${R2_INPUT_BUCKET_NAME}
      - R2_OUTPUT_BUCKET_NAME=${R2_OUTPUT_BUCKET_NAME}
      - R2_MODELS_BUCKET_NAME=${R2_MODELS_BUCKET_NAME}
    volumes:
      - ./models:/root/.cache/huggingface/hub
      - ./data:/app/data
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

echo -e "${GREEN}✅ Vast.ai Konfigürasyonu Hazır!${NC}"
echo ""
echo "Kullanım:"
echo "========="
echo "1. Vast.ai'de bir sunucu kiralayın:"
echo "   vastai create instance --gpu A100 --num-gpus 1 --image pytorch/pytorch:latest"
echo ""
echo "2. Sunucuya SSH ile bağlanın:"
echo "   ssh -i $SSH_KEY_PATH root@<server-ip>"
echo ""
echo "3. Bu projeyi sunucuya kopyalayın ve çalıştırın:"
echo "   git clone <repo-url>"
echo "   cd deploy-api-inference-engine"
echo "   export HF_HUB_TOKEN=..."
echo "   export R2_*=..."
echo "   docker-compose -f docker-compose.vastai.yml up -d"
echo ""
echo "4. API'yi test edin:"
echo "   curl http://<server-ip>:8000/health"

# Environment dosyası template
echo -e "${YELLOW}📝 Environment Template Oluşturuluyor...${NC}"
cat > .env.vastai << 'EOF'
# HuggingFace
HF_HUB_TOKEN=your_hf_token_here

# R2 Cloudflare
R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your_access_key
R2_SECRET_ACCESS_KEY=your_secret_key
R2_INPUT_BUCKET_NAME=visgate-deploy-api-inference-input
R2_OUTPUT_BUCKET_NAME=visgate-deploy-api-inference-output
R2_MODELS_BUCKET_NAME=visgate-deploy-api-models-apac
EOF

echo -e "${GREEN}✅ Tamamlandı!${NC}"
echo ""
echo "Önemli Notlar:"
echo "=============="
echo "- R2 bucket'larınızda 'models/' prefix'i ile model dosyaları saklayın"
echo "- İlk istekte model HuggingFace'ten inecek ve cache'lenecek"
echo "- Sonraki istekler için model GPU'da hazır kalacak"
echo "- VRAM dolduğunda otomatik temizlik yapılır"
