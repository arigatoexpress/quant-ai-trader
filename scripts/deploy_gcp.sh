#!/bin/bash

# Google Cloud Deployment Script for Quant AI Trader
# This script deploys the application to Google Cloud Platform

set -e  # Exit on any error

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-quant-ai-trader}"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
SERVICE_NAME="quant-ai-trader"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Starting Google Cloud deployment..."
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Service: ${SERVICE_NAME}"

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it first."
    exit 1
fi

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "❌ Not authenticated with gcloud. Please run 'gcloud auth login'"
    exit 1
fi

# Set project
echo "🔧 Setting project..."
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable sql-component.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Build and push Docker image
echo "🏗️  Building Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} .

# Create Cloud SQL instance (if not exists)
echo "💾 Setting up Cloud SQL..."
if ! gcloud sql instances describe quantai-db --region=${REGION} &> /dev/null; then
    echo "Creating Cloud SQL instance..."
    gcloud sql instances create quantai-db \
        --database-version=POSTGRES_14 \
        --tier=db-f1-micro \
        --region=${REGION} \
        --root-password=secure_password_123
    
    # Create database
    gcloud sql databases create quantai --instance=quantai-db
    
    # Create user
    gcloud sql users create trader \
        --instance=quantai-db \
        --password=trader_password_123
else
    echo "Cloud SQL instance already exists"
fi

# Create secrets in Secret Manager
echo "🔐 Setting up secrets..."
if ! gcloud secrets describe app-secrets &> /dev/null; then
    echo "Creating app secrets..."
    
    # Create .env file for secrets
    cat > /tmp/app-secrets.env << EOF
GROK_API_KEY=your_grok_api_key_here
COINGECKO_API_KEY=your_coingecko_api_key
DEXSCREENER_API_KEY=your_dexscreener_api_key
DEFILLAMA_API_KEY=your_defillama_api_key
MASTER_PASSWORD=secure_master_password_123
DATABASE_URL=postgresql://trader:trader_password_123@/quantai?host=/cloudsql/${PROJECT_ID}:${REGION}:quantai-db
EOF
    
    gcloud secrets create app-secrets --data-file=/tmp/app-secrets.env
    rm /tmp/app-secrets.env
else
    echo "Secrets already exist"
fi

# Deploy to Cloud Run
echo "☁️  Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 3600 \
    --concurrency 80 \
    --set-env-vars "ENVIRONMENT=production,LOG_LEVEL=INFO,PAPER_TRADING_MODE=true" \
    --set-secrets="/app/secrets=app-secrets:latest" \
    --add-cloudsql-instances ${PROJECT_ID}:${REGION}:quantai-db

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')

echo "✅ Deployment complete!"
echo "🌐 Service URL: ${SERVICE_URL}"
echo "📊 Monitoring: https://console.cloud.google.com/monitoring"
echo "📝 Logs: https://console.cloud.google.com/logs"

# Set up monitoring alerts
echo "🔔 Setting up monitoring..."
./scripts/setup_monitoring.sh

echo "🎉 Deployment successful!"
echo ""
echo "Next steps:"
echo "1. Update secrets in Secret Manager with your actual API keys"
echo "2. Configure your domain and SSL certificate"
echo "3. Set up backup and disaster recovery"
echo "4. Review security settings and access controls" 