#!/bin/bash

# =============================================================================
# Quant AI Trader - Production Setup Script
# =============================================================================

set -e  # Exit on any error

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    🤖 QUANT AI TRADER - PRODUCTION SETUP                     ║"
echo "║                                                                              ║"
echo "║  This script will configure your system for production deployment           ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo

# Check if running as root (not recommended for production)
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  WARNING: Running as root is not recommended for production"
    echo "   Consider creating a dedicated user for the trading system"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
echo "🔍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Python 3.8+ required, found Python $PYTHON_VERSION"
    echo "   Please install Python 3.8+ and try again"
    exit 1
fi

echo "✅ Python version: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "🔄 Creating virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo "🔄 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
    echo "✅ Dependencies installed from requirements.txt"
else
    echo "⚠️  requirements.txt not found, installing core dependencies..."
    pip install pandas numpy yfinance ta scikit-learn torch fastapi uvicorn aiohttp pytz python-dotenv psutil > /dev/null 2>&1
    echo "✅ Core dependencies installed"
fi

# Create necessary directories
echo "🔄 Creating directories..."
mkdir -p logs
mkdir -p backups
mkdir -p data
mkdir -p config
echo "✅ Directories created"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📝 Setting up environment configuration..."
    
    if [ -f "env_template.txt" ]; then
        cp env_template.txt .env
        echo "✅ Environment template copied to .env"
        echo "⚠️  IMPORTANT: Edit .env file with your actual API keys and configuration"
        echo "   Required variables: GROK_API_KEY, MASTER_PASSWORD, MAX_TRADE_AMOUNT, RISK_TOLERANCE"
    else
        echo "❌ env_template.txt not found"
        echo "   Please create .env file manually with required environment variables"
    fi
else
    echo "✅ .env file exists"
fi

# Check for API keys in environment
echo "🔍 Checking configuration..."
source .env 2>/dev/null || true

MISSING_VARS=()
REQUIRED_VARS=("GROK_API_KEY" "MASTER_PASSWORD" "MAX_TRADE_AMOUNT" "RISK_TOLERANCE")

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -eq 0 ]; then
    echo "✅ All required environment variables are set"
else
    echo "⚠️  Missing required environment variables: ${MISSING_VARS[*]}"
    echo "   Please edit .env file and set these variables before running in production"
fi

# Make scripts executable
echo "🔄 Setting up executables..."
chmod +x src/production_launcher.py 2>/dev/null || true
chmod +x setup_production.sh 2>/dev/null || true
echo "✅ Scripts made executable"

# Run system validation
echo "🔍 Running system validation..."
cd src
python3 -c "
import sys
import os
sys.path.append('.')

try:
    # Test core imports
    from data_fetcher import DataFetcher
    from trading_agent import TradingAgent
    from technical_analyzer import TechnicalAnalyzer
    from web_app import WebApp
    print('✅ Core components import successfully')
    
    # Test data fetcher
    df = DataFetcher()
    price_data = df.fetch_price_and_market_cap('BTC')
    if price_data:
        print('✅ Data fetcher working')
    else:
        print('⚠️  Data fetcher returned no data (this is normal without API keys)')
    
    print('✅ System validation passed')
    
except Exception as e:
    print(f'❌ System validation failed: {e}')
    sys.exit(1)
"

cd ..

if [ $? -eq 0 ]; then
    echo "✅ System validation passed"
else
    echo "❌ System validation failed"
    echo "   Please check the error messages above and fix any issues"
    exit 1
fi

# Display production readiness checklist
echo
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         🎯 PRODUCTION READINESS CHECKLIST                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo
echo "✅ System Requirements:"
echo "   ✓ Python 3.8+ installed"
echo "   ✓ Virtual environment created"
echo "   ✓ Dependencies installed"
echo "   ✓ Directories created"
echo "   ✓ Core components validated"
echo
echo "📝 Configuration (REQUIRED):"
if [ ${#MISSING_VARS[@]} -eq 0 ]; then
    echo "   ✓ Environment variables configured"
else
    echo "   ❌ Edit .env file with: ${MISSING_VARS[*]}"
fi
echo
echo "🔐 Security Checklist:"
echo "   🔲 Change default MASTER_PASSWORD"
echo "   🔲 Set strong, unique API keys"
echo "   🔲 Review RISK_TOLERANCE and MAX_TRADE_AMOUNT"
echo "   🔲 Enable PAPER_TRADING_MODE for testing"
echo "   🔲 Configure wallet addresses"
echo "   🔲 Set up monitoring and alerting"
echo
echo "🚀 Deployment Options:"
echo "   • Test Mode:      PAPER_TRADING_MODE=true  (Recommended for testing)"
echo "   • Production:     PAPER_TRADING_MODE=false (Real money trading)"
echo "   • Autonomous:     ENABLE_AUTONOMOUS_TRADING=true"
echo
echo "🎯 Next Steps:"
echo "   1. Edit .env file with your configuration"
echo "   2. Test with paper trading: python3 src/production_launcher.py"
echo "   3. Monitor logs in ./logs/ directory"
echo "   4. Access dashboard at http://127.0.0.1:8000"
echo
echo "⚠️  IMPORTANT SAFETY NOTES:"
echo "   • Start with PAPER_TRADING_MODE=true to test the system"
echo "   • Use small amounts when switching to live trading"
echo "   • Monitor the system closely in the first 24 hours"
echo "   • Keep backups of your configuration and data"
echo "   • Never share your API keys or .env file"
echo
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                              🎉 SETUP COMPLETE!                              ║"
echo "║                                                                              ║"
echo "║  Your Quant AI Trader is ready for production deployment.                   ║"
echo "║  Please complete the configuration checklist above before running.          ║"
echo "║                                                                              ║"
echo "║  Start the system: python3 src/production_launcher.py                       ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝" 