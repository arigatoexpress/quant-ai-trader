# Multi-Chain Portfolio Management System

## Overview

This comprehensive system analyzes your cryptocurrency portfolio across multiple blockchains and provides AI-powered trading recommendations with autonomous risk management.

## Supported Blockchains

- **SUI**: 11 wallets monitored
- **Solana**: 2 wallets monitored
- **Ethereum**: 1 wallet monitored
- **Base**: 1 wallet monitored
- **Sei**: 1 wallet monitored

## Key Features

### 🔗 Multi-Chain Analysis
- Real-time balance tracking across all supported chains
- Comprehensive token identification and valuation
- Cross-chain portfolio aggregation and analysis

### 🤖 AI-Powered Recommendations
- GROK-3 AI integration for intelligent analysis
- Confidence-based recommendation system
- Priority-based action suggestions (HIGH/MEDIUM/LOW)
- Market context-aware decision making

### ⚖️ Risk Management
- Automated risk level assessment (LOW/MEDIUM/HIGH)
- Diversification score calculation (0-100)
- Concentration risk monitoring
- Proactive alert system

### 📊 Portfolio Analytics
- Total portfolio value tracking
- Chain allocation analysis
- Token allocation breakdown
- Performance metrics and history

### 🔄 Autonomous Management
- Continuous 24/7 monitoring
- Automated rebalancing suggestions
- Real-time alert generation
- Performance tracking and optimization

## Quick Start Guide

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure your config/config.yaml has your GROK API key
```

### 2. Basic Portfolio Analysis

```python
from src.portfolio_analyzer import MultiChainPortfolioAnalyzer

# Initialize analyzer
analyzer = MultiChainPortfolioAnalyzer()

# Perform comprehensive analysis
analysis = analyzer.analyze_full_portfolio()

# Display results
analyzer.print_portfolio_report(analysis)
```

### 3. Start Autonomous Management

```python
from src.portfolio_agent import PortfolioAgent

# Initialize portfolio agent
agent = PortfolioAgent()

# Start autonomous monitoring
agent.start_autonomous_portfolio_management()

# Let it run continuously...
# Stop when needed
agent.stop_autonomous_portfolio_management()
```

### 4. Run Complete Demo

```bash
# Run the comprehensive demo
python src/comprehensive_demo.py

# Run individual components
python src/portfolio_demo.py
python src/portfolio_agent.py
```

## Configuration

### Risk Tolerance Settings

```python
# Adjust risk parameters in portfolio_agent.py
portfolio_manager = PortfolioManager(
    risk_tolerance=0.15,      # 15% risk tolerance
    rebalance_threshold=0.05  # 5% rebalancing threshold
)
```

### Monitoring Intervals

```python
# Adjust monitoring frequency in portfolio_agent.py
self.analysis_interval = 1800  # 30 minutes (in seconds)
```

## Your Wallet Addresses

### SUI Wallets (11 total)
- `0x2401cdd7771bd1d45050eb086c5a2793a2799cb5864f32afdc2badcf44b45563`
- `0x57b78bf1502af2c0b94801dfb3f4e8ec189b0404915314bb011d867115757179`
- `0x7498668ffe15c4aca457787a8ec4aa56c4973890f1c6921d330550dc8f26fbdb`
- `0x40447f2412efdf5a5dd32c77cb5c6ff6b4d46a19599b7cc520878a33a6edb3ba`
- `0x4cbbd223b85f3346bb77b0076a15b4af3f7faf54ab8ff815a571877c9b44bbfa`
- `0x5b54fbf8f54103982097c0bec1c40f9292326d963ca51a01b54b82740309e594`
- `0x9eba9b4f76732dcc70875fba7fbb7949c9329afebe9c5cc71ba1640ec7fd1bbc`
- `0x1ee3f6f78eef9c51b613fcca46bbcfab6cc22a9bc1cd89296055d7b8ef986fdb`
- `0x18a30a4878761bfdfed20646e5b6366f451121f7fb2c7f7f6fcbde8efe739a82`
- `0x0286e661e867e766c390a88e38d80dcd426059efbafcc41796c3f2a1e540cc67`
- `0xf92cf979a898d0e6aba1a0eae3906fd66fdb6ed49a76d349d8cf0e2431901ae5`

### Solana Wallets (2 total)
- `DX3wsNdaCw3PK3BWxPG7HdijzKKLAnSXDCJRVu8z74t2`
- `BqyvPBrPHvJ1VHz11m5WYUwqzZg3QnmvzzY1a9BnYbFw`

### Ethereum Wallet
- `0xc42E2eB90528291Bd0c68956302BB7e48E578827`

### Base Wallet
- `0xc42E2eB90528291Bd0c68956302BB7e48E578827`

### Sei Wallet
- `0xeea5e020ac1a3e364303924d0f3b29fd14027944`

## Usage Examples

### Get Portfolio Status

```python
from src.portfolio_agent import PortfolioAgent

agent = PortfolioAgent()
status = agent.get_portfolio_status()

print(f"Total Value: ${status['total_value']:,.2f}")
print(f"Risk Level: {status['risk_level']}")
print(f"Decisions Made: {status['total_decisions']}")
```

### Get AI Recommendations

```python
from src.portfolio_analyzer import MultiChainPortfolioAnalyzer

analyzer = MultiChainPortfolioAnalyzer()
analysis = analyzer.analyze_full_portfolio()

recommendations = analysis.get("recommendations", [])
for rec in recommendations:
    print(f"{rec.action} {rec.token}: {rec.reasoning}")
```

### Force Immediate Analysis

```python
from src.portfolio_agent import PortfolioAgent

agent = PortfolioAgent()
analysis = agent.force_portfolio_analysis()
```

## Alert System

The system generates alerts for:

- **RISK_THRESHOLD**: When portfolio risk exceeds safe levels
- **BALANCE_CHANGE**: Significant portfolio value changes (>5%)
- **REBALANCE_NEEDED**: When allocations drift from targets
- **OPPORTUNITY**: High-confidence trading opportunities

## Performance Tracking

The system tracks:
- Total decisions made
- Successful rebalances
- Risk reduction actions
- Opportunity captures
- Portfolio value changes

## Real-World Scenarios

### Morning Portfolio Check
```python
# Run this daily before market open
agent = PortfolioAgent()
analysis = agent.force_portfolio_analysis()
recommendations = agent.get_portfolio_recommendations()
```

### Continuous Monitoring
```python
# Set up 24/7 monitoring
agent = PortfolioAgent()
agent.start_autonomous_portfolio_management()
# System will run continuously until stopped
```

### Risk Assessment
```python
# Check current risk levels
analyzer = MultiChainPortfolioAnalyzer()
analysis = analyzer.analyze_full_portfolio()
risk_level = analysis["metrics"]["risk_level"]
diversification = analysis["metrics"]["diversification_score"]
```

## Integration with Existing System

The portfolio management system integrates seamlessly with your existing trading infrastructure:

- **AgenticElizaOS**: Enhanced with portfolio context
- **Market Monitoring**: Real-time price tracking
- **Technical Analysis**: Combined with portfolio metrics
- **Risk Management**: Unified risk assessment

## Advanced Features

### Custom Risk Tolerance
```python
# Adjust risk parameters
portfolio_manager = PortfolioManager(
    risk_tolerance=0.10,      # 10% risk tolerance (conservative)
    rebalance_threshold=0.03  # 3% rebalancing threshold (frequent)
)
```

### Scheduled Analysis
```python
# Set up custom monitoring intervals
agent = PortfolioAgent()
agent.analysis_interval = 900  # 15 minutes
agent.start_autonomous_portfolio_management()
```

### AI Model Configuration
```python
# Adjust AI parameters in portfolio_analyzer.py
completion = grok_client.chat.completions.create(
    model="grok-3",
    temperature=0.2,  # Lower for more conservative recommendations
    max_tokens=1500
)
```

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure your GROK API key is correctly set in `config/config.yaml`
2. **Network Errors**: Check RPC endpoints are accessible
3. **Missing Dependencies**: Run `pip install -r requirements.txt`
4. **Wallet Connection**: Verify wallet addresses are correct

### Error Handling

The system includes comprehensive error handling:
- Graceful fallbacks for API failures
- Synthetic data generation when needed
- Automatic retry mechanisms
- Detailed error logging

## Security Considerations

- **Read-Only Access**: System only reads wallet balances, never executes trades
- **No Private Keys**: Only public addresses are used
- **Secure API**: All API calls use HTTPS
- **Local Processing**: Portfolio analysis runs locally

## Next Steps

1. **Configure Risk Tolerance**: Adjust parameters based on your preferences
2. **Set Up Notifications**: Connect to Discord/Slack for alerts
3. **Review Recommendations**: Regularly check AI suggestions
4. **Monitor Performance**: Track system effectiveness over time
5. **Adjust Parameters**: Fine-tune based on market conditions

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the demo files for usage examples
3. Examine the configuration files for customization options
4. Test with the comprehensive demo first

---

**🚀 Your portfolio is now under comprehensive AI management across 5 blockchains with 15+ wallets monitored!** 