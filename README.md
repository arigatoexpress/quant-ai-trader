# Quant AI Trader 🚀

Advanced AI-powered cryptocurrency trading system with asymmetric opportunity scanning, real-time market analysis, and autonomous decision-making capabilities.

## 🎯 Overview

The Quant AI Trader is a sophisticated algorithmic trading system that combines artificial intelligence, real-time market data, and asymmetric trading strategies to identify and execute high-probability trading opportunities across multiple cryptocurrency exchanges and DeFi protocols.

### Key Features

- **🤖 AI-Powered Analysis**: Advanced ML models including LSTM, Transformer, and XGBoost for market prediction
- **💎 Asymmetric Trading**: Focus on maximum profit through high-conviction, limited-risk opportunities  
- **🔄 Real-Time Data**: Live market data from 8+ sources including DexScreener, CoinGecko Pro, DeFi Llama
- **🛡️ Risk Management**: AI-driven risk assessment with dynamic position sizing using Kelly Criterion
- **📊 Multi-Chain Support**: Trading across Ethereum, Solana, Sui, Base, and other major blockchains
- **🔐 Enterprise Security**: Advanced encryption, audit logging, and secure key management
- **📈 Performance Analytics**: Comprehensive backtesting and performance attribution
- **🌐 Web Dashboard**: Modern React-based interface for monitoring and control
- **☁️ Cloud Ready**: Google Cloud deployment with auto-scaling and monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 16+ (for web dashboard)
- Git
- API keys for trading platforms and data sources

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/quant-ai-trader.git
   cd quant-ai-trader
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.template .env
   # Edit .env with your API keys and configuration
   ```

5. **Initialize the system**
   ```bash
   python src/main.py --validate
   ```

6. **Start trading**
   ```bash
   python src/main.py
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Core Configuration
GROK_API_KEY=your_grok_api_key_here
INITIAL_CAPITAL=100000
PAPER_TRADING_MODE=true
ENABLE_AUTONOMOUS_TRADING=false

# Risk Management
MAX_POSITION_SIZE=0.25
RISK_TOLERANCE=0.15
KELLY_MULTIPLIER=0.5

# Data Sources
COINGECKO_API_KEY=your_coingecko_pro_key
DEXSCREENER_API_KEY=your_dexscreener_key
DEFILLAMA_API_KEY=your_defillama_key

# Security
MASTER_PASSWORD=your_strong_password
ENABLE_2FA=true
SESSION_TIMEOUT=3600

# Deployment
ENVIRONMENT=development
LOG_LEVEL=INFO
WEB_PORT=8080
```

### Trading Configuration

Edit `config/config.yaml` to customize:

- Asset preferences and blacklists
- Risk management parameters  
- ML model configurations
- Alert thresholds
- Execution parameters

## 🎯 Trading Strategies

### Asymmetric Opportunities

The system focuses on identifying trades with:
- **Limited downside risk** (typically 1-5% max loss)
- **Unlimited upside potential** (10x+ possible gains)
- **High probability of success** (>70% confidence)
- **Positive expected value** (>20% expected return)

### Strategy Types

1. **DeFi Yield Farming**: High-yield opportunities with risk assessment
2. **Options Trading**: Underpriced options with high convexity
3. **Arbitrage**: Cross-exchange and cross-chain price discrepancies
4. **Momentum Trading**: AI-detected trend breakouts and reversals
5. **News Trading**: Sentiment-driven opportunities from market events

## 📊 Performance Monitoring

### Web Dashboard

Access the web dashboard at `http://localhost:8080` to monitor:

- Real-time portfolio performance
- Active trading opportunities
- Risk metrics and exposure
- AI model predictions
- System health and alerts

### Key Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst-case loss scenario
- **Win Rate**: Percentage of profitable trades
- **Expected Value**: Average expected return per trade
- **Kelly Allocation**: Optimal position sizing

## 🛡️ Security Features

### Authentication & Authorization

- **Strong Password Requirements**: Minimum 12 characters with complexity
- **Two-Factor Authentication**: TOTP support via authenticator apps
- **Session Management**: Automatic timeout and secure sessions
- **API Key Encryption**: All sensitive data encrypted at rest

### Audit & Monitoring

- **Comprehensive Logging**: All trading actions logged and auditable
- **Real-time Alerts**: Immediate notification of security events
- **Health Monitoring**: Continuous system health checks
- **Backup & Recovery**: Automated backups of critical data

## ☁️ Google Cloud Deployment

### Prerequisites

1. Google Cloud Account with billing enabled
2. gcloud CLI installed and configured
3. Docker installed locally

### Deployment Steps

1. **Prepare deployment**
   ```bash
   ./scripts/prepare_deployment.sh
   ```

2. **Build and deploy**
   ```bash
   ./scripts/deploy_gcp.sh
   ```

3. **Configure monitoring**
   ```bash
   ./scripts/setup_monitoring.sh
   ```

### Cloud Services Used

- **Compute Engine**: Main application hosting
- **Cloud SQL**: Database for trading data
- **Cloud Storage**: Backup and data storage
- **Cloud Monitoring**: System and application monitoring
- **Cloud Logging**: Centralized log management
- **Cloud Scheduler**: Automated tasks and maintenance

## 🧪 Testing

### Run Tests

```bash
# Quick validation
python src/simple_test.py

# Comprehensive testing
python src/comprehensive_testing_framework.py

# Validate real data integrations
python src/test_real_data_integrations.py
```

### Test Coverage

- **Unit Tests**: Individual component testing
- **Integration Tests**: Data source and API testing
- **Performance Tests**: Latency and throughput validation
- **Security Tests**: Vulnerability scanning and audit
- **End-to-End Tests**: Complete trading workflow validation

## 📖 API Documentation

### Core APIs

- **Trading Agent**: Execute trades and manage positions
- **Data Fetcher**: Real-time market data collection
- **Risk Manager**: Position sizing and risk assessment
- **Portfolio Analyzer**: Performance tracking and analysis

### Example Usage

```python
from src.trading_agent import TradingAgent
from src.asymmetric_trading_framework import MaxProfitTradingFramework

# Initialize trading system
trader = TradingAgent()
framework = MaxProfitTradingFramework(config)

# Scan for opportunities
opportunities = await framework.scan_asymmetric_opportunities(market_data)

# Execute high-conviction trades
for opp in opportunities[:5]:
    if opp.confidence_score > 0.8:
        await trader.execute_trade(opp)
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always trade with money you can afford to lose.

- **No Financial Advice**: This is not financial advice
- **Use at Your Own Risk**: Authors are not responsible for any losses
- **Paper Trading Recommended**: Test thoroughly before live trading
- **Regulatory Compliance**: Ensure compliance with local regulations

## 🔗 Links

- **Documentation**: [docs.quantaitrader.com](https://docs.quantaitrader.com)
- **Community**: [Discord Server](https://discord.gg/quantaitrader)
- **Support**: [support@quantaitrader.com](mailto:support@quantaitrader.com)
- **Bug Reports**: [GitHub Issues](https://github.com/yourusername/quant-ai-trader/issues)

## 🙏 Acknowledgments

- OpenAI for GPT models and AI research
- xAI for Grok API access
- CoinGecko for market data
- DexScreener for DEX analytics
- The open-source community for tools and libraries

---

**Built with ❤️ for the crypto trading community**

