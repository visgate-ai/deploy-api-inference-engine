#!/bin/bash
# RunPod Serverless Deployment Script
# Bu script, API'yi RunPod serverless GPU endpoint olarak deploy eder

set -e

# Renkli çıktı
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 RunPod Serverless Deployment${NC}"
echo "========================================"

# Environment değişkenleri
export HF_HUB_TOKEN=${HF_HUB_TOKEN:-""}
export R2_ENDPOINT_URL=${R2_ENDPOINT_URL:-""}
export R2_ACCESS_KEY_ID=${R2_ACCESS_KEY_ID:-""}
export R2_SECRET_ACCESS_KEY=${R2_SECRET_ACCESS_KEY:-""}
export R2_INPUT_BUCKET_NAME=${R2_INPUT_BUCKET_NAME:-""}
export R2_OUTPUT_BUCKET_NAME=${R2_OUTPUT_BUCKET_NAME:-""}
export R2_MODELS_BUCKET_NAME=${R2_MODELS_BUCKET_NAME:-""}

# Docker image build
IMAGE_NAME="visgate-deploy-api:${1:-latest}"
REGISTRY="registry.runpod.io/your-username"

echo -e "${YELLOW}📦 Docker Image Build${NC}"
docker build -t $IMAGE_NAME .

echo -e "${YELLOW}🔄 Docker Image Tagging${NC}"
docker tag $IMAGE_NAME $REGISTRY/$IMAGE_NAME

echo -e "${YELLOW}⬆️ Docker Image Push${NC}"
docker push $REGISTRY/$IMAGE_NAME

# RunPod endpoint oluşturma
echo -e "${YELLOW}🚀 RunPod Endpoint Oluşturma${NC}"

# API key kontrolü
if [ -z "$RUNPOD_API_KEY" ]; then
    echo -e "${RED}❌ RUNPOD_API_KEY ayarlanmamış!${NC}"
    echo "export RUNPOD_API_KEY=your_api_key"
    exit 1
fi

# Endpoint JSON
cat > runpod_endpoint.json << 'EOF'
{
  "gpuCount": 1,
  "gpuTypeId": "ampere-a100",
  "imageName": "your-username/visgate-deploy-api:latest",
  "env": [
    {"key": "HF_HUB_TOKEN", "value": ""},
    {"key": "R2_ENDPOINT_URL", "value": ""},
    {"key": "R2_ACCESS_KEY_ID", "value": ""},
    {"key": "R2_SECRET_ACCESS_KEY", "value": ""},
    {"key": "R2_INPUT_BUCKET_NAME", "value": ""},
    {"key": "R2_OUTPUT_BUCKET_NAME", "value": ""},
    {"key": "R2_MODELS_BUCKET_NAME", "value": ""}
  ],
  "minMemoryGb": 80,
  "maxConcurrentRequests": 1,
  "idleTimeout": 60,
  "volumes": [],
  "ports": [{"privatePort": 8000, "publicPort": 8000}],
  "networkVolumeId": "your-network-volume-id"
}
EOF

echo -e "${GREEN}✅ Deployment hazır!${NC}"
echo ""
echo "RunPod console'dan endpoint oluşturun:"
echo "1. Go to https://www.runpod.io/console/serverless"
echo "2. Create new endpoint"
echo "3. Use the JSON config above"
echo ""
echo "veya API ile:"
echo "curl -X POST https://api.runpod.io/serverless/v1/endpoints \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer \$RUNPOD_API_KEY' \\
  -d @runpod_endpoint.json"
