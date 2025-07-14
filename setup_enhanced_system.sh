#!/bin/bash

# Enhanced Quant AI Trader Setup Script
# Sets up the complete system with Telegram bot integration and asymmetric trading scanner

set -e  # Exit on any error

echo "🚀 Enhanced Quant AI Trader Setup"
echo "=================================="
echo "Setting up:"
echo "• Multi-source data integration"
echo "• Asymmetric trading opportunity scanner"
echo "• Telegram bot alerts"
echo "• Production-ready environment"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Python 3.11+ is available
check_python() {
    echo "🐍 Checking Python version..."
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        print_status "Python 3.11 found"
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l) == 1 ]]; then
            PYTHON_CMD="python3"
            print_status "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.11+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3.11+ not found. Please install Python 3.11+"
        exit 1
    fi
}

# Setup virtual environment
setup_venv() {
    echo ""
    echo "📦 Setting up virtual environment..."
    
    if [ -d ".venv" ]; then
        print_warning "Virtual environment already exists, removing..."
        rm -rf .venv
    fi
    
    $PYTHON_CMD -m venv .venv
    print_status "Virtual environment created"
    
    # Activate virtual environment
    source .venv/bin/activate
    print_status "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip
    print_status "Pip upgraded"
}

# Install dependencies
install_dependencies() {
    echo ""
    echo "📚 Installing dependencies..."
    
    # Install requirements
    pip install -r requirements.txt
    print_status "Dependencies installed"
    
    # Install additional Telegram and async dependencies
    echo "📱 Installing Telegram bot dependencies..."
    pip install python-telegram-bot==20.7 aiohttp==3.9.1 websockets==12.0
    print_status "Telegram dependencies installed"
    
    # Install data analysis dependencies
    echo "📊 Installing data analysis dependencies..."
    pip install ta==0.10.2 dataclasses-json==0.6.3
    print_status "Data analysis dependencies installed"
}

# Setup environment configuration
setup_environment() {
    echo ""
    echo "⚙️  Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f "env_template.txt" ]; then
            cp env_template.txt .env
            print_status "Environment template copied to .env"
        elif [ -f ".env.template" ]; then
            cp .env.template .env
            print_status "Environment template copied to .env"
        else
            print_warning "No environment template found, creating basic .env"
            cat > .env << 'EOF'
# Basic environment configuration
GROK_API_KEY=your_grok_api_key_here
COINGECKO_API_KEY=your_coingecko_api_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
NEWS_API_KEY=your_news_api_key_here
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here

# Security
MASTER_PASSWORD=your_secure_master_password_here
SESSION_TIMEOUT=3600

# Trading Configuration
MAX_TRADE_AMOUNT=1000.0
RISK_TOLERANCE=0.02
CONFIDENCE_THRESHOLD=0.7
EOF
        fi
    else
        print_info ".env file already exists, skipping"
    fi
    
    print_warning "Please edit .env file with your actual API keys and configuration"
}

# Create required directories
create_directories() {
    echo ""
    echo "📁 Creating required directories..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p reports
    mkdir -p config
    
    print_status "Directories created"
}

# Test the installation
test_installation() {
    echo ""
    echo "🧪 Testing installation..."
    
    echo "Testing Python imports..."
    $PYTHON_CMD -c "
import sys
print(f'Python version: {sys.version}')

# Test core imports
try:
    import pandas as pd
    import numpy as np
    import requests
    import yaml
    print('✅ Core dependencies OK')
except ImportError as e:
    print(f'❌ Core dependency error: {e}')
    sys.exit(1)

# Test Telegram imports
try:
    import telegram
    print('✅ Telegram bot dependencies OK')
except ImportError as e:
    print(f'⚠️  Telegram dependency warning: {e}')

# Test async imports
try:
    import aiohttp
    import asyncio
    print('✅ Async dependencies OK')
except ImportError as e:
    print(f'⚠️  Async dependency warning: {e}')

# Test data analysis imports
try:
    import ta
    print('✅ Technical analysis dependencies OK')
except ImportError as e:
    print(f'⚠️  Technical analysis warning: {e}')

print('\\n🎉 Installation test completed!')
"
    
    if [ $? -eq 0 ]; then
        print_status "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Create launch scripts
create_launch_scripts() {
    echo ""
    echo "🚀 Creating launch scripts..."
    
    # Create launcher for asymmetric scanner
    cat > start_scanner.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
echo "🔍 Starting Asymmetric Trading Scanner..."
python src/asymmetric_scanner.py
EOF
    chmod +x start_scanner.sh
    
    # Create launcher for Telegram bot
    cat > start_telegram_bot.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
echo "📱 Starting Telegram Trading Bot..."
python src/telegram_bot.py
EOF
    chmod +x start_telegram_bot.sh
    
    # Create launcher for integrated demo
    cat > start_demo.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
echo "🎬 Starting Integrated Demo..."
python src/integrated_demo.py
EOF
    chmod +x start_demo.sh
    
    # Create launcher for production system
    cat > start_production.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
echo "🏭 Starting Enhanced Production System..."
python src/enhanced_production_launcher.py
EOF
    chmod +x start_production.sh
    
    # Create launcher for testing real data integrations
    cat > test_real_data.sh << 'EOF'
#!/bin/bash
source .venv/bin/activate
echo "🧪 Testing Real Data Integrations..."
python src/test_real_data_integrations.py
EOF
    chmod +x test_real_data.sh
    
    print_status "Launch scripts created"
}

# Display setup completion and next steps
display_completion() {
    echo ""
    echo "🎉 Enhanced Setup Complete!"
    echo "=========================="
    echo ""
    echo "📋 Next Steps:"
    echo ""
    echo "1. Configure your .env file with actual API keys:"
    echo "   nano .env"
    echo ""
    echo "2. Get Telegram Bot Token:"
    echo "   • Message @BotFather on Telegram"
    echo "   • Create new bot with /newbot"
    echo "   • Copy token to TELEGRAM_BOT_TOKEN in .env"
    echo "   • Get your chat ID and add to TELEGRAM_CHAT_ID"
    echo ""
    echo "3. Get API keys:"
    echo "   • CoinGecko: https://www.coingecko.com/en/api"
    echo "   • News API: https://newsapi.org/"
    echo "   • Grok AI: https://x.ai/"
    echo ""
    echo "4. Test the system:"
    echo "   ./test_real_data.sh          # Test real data integrations"
    echo "   ./start_demo.sh              # Run integrated demo"
    echo "   ./start_scanner.sh           # Run asymmetric scanner"
    echo "   ./start_telegram_bot.sh      # Start Telegram bot"
    echo "   ./start_production.sh        # Enhanced production system"
    echo ""
    echo "5. Available features:"
    echo "   ✅ REAL DATA INTEGRATIONS:"
    echo "      • DexScreener API (300 calls/min) - Live DEX pair data"
    echo "      • CoinGecko Pro API (500 calls/min) - Market data & trends"  
    echo "      • DeFi Llama API (120 calls/min) - TVL & yield farming data"
    echo "      • Sui Network API (120 calls/min) - Native blockchain data"
    echo "      • Noodles Finance API - Sui ecosystem DeFi data"
    echo "      • Jupiter Aggregator - Solana DEX data"
    echo "      • Birdeye - Trending tokens and analytics"
    echo "      • GeckoTerminal - Additional DEX pool data"
    echo "   ✅ Intelligent data aggregation with cross-validation"
    echo "   ✅ Asymmetric trading opportunity scanner"
    echo "   ✅ Real-time Telegram alerts and notifications"
    echo "   ✅ Automated monitoring and analysis"
    echo "   ✅ Risk-adjusted opportunity scoring"
    echo "   ✅ Production-ready deployment with health monitoring"
    echo ""
    echo "📚 Documentation:"
    echo "   • README.md - Complete system overview"
    echo "   • AI_ENHANCEMENTS_README.md - AI features guide"
    echo "   • TRADING_ANALYSIS_GUIDE.md - Trading strategies"
    echo ""
    echo "🔧 Telegram Bot Commands:"
    echo "   /start - Initialize bot"
    echo "   /opportunities - View latest opportunities"
    echo "   /add_alert BTC price 5 - Set price alert"
    echo "   /status - Check system status"
    echo "   /market SUI - Get market data"
    echo ""
}

# Main execution
main() {
    check_python
    setup_venv
    install_dependencies
    setup_environment
    create_directories
    test_installation
    create_launch_scripts
    display_completion
}

# Handle interruption
trap 'echo -e "\n${RED}❌ Setup interrupted${NC}"; exit 1' INT

# Run main function
main

echo ""
print_status "Setup completed successfully! 🚀"
echo ""
echo "To get started:"
echo "  source .venv/bin/activate"
echo "  ./start_demo.sh"
echo "" 