"""
Enhanced Trading Application
Comprehensive AI-powered trading system with security, analytics, and visualizations
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import logging
import numpy as np
from pathlib import Path

# Import our enhanced modules
from secure_config_manager import SecureConfigManager
from portfolio_visualizer import PortfolioVisualizer
from advanced_analytics_engine import AdvancedAnalyticsEngine
from secure_autonomous_trader import SecureAgenticElizaOS
from cybersecurity_framework import SecureTradingFramework
from portfolio_agent import PortfolioAgent
from trading_history_analyzer import TradingHistoryAnalyzer
from data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTradingApplication:
    """
    Enhanced Trading Application with comprehensive features:
    - Secure configuration management
    - Advanced portfolio visualization
    - AI-powered analytics and insights
    - Autonomous trading with cybersecurity
    - Real-time monitoring and alerts
    """
    
    def __init__(self):
        print("🚀 ENHANCED TRADING APPLICATION")
        print("=" * 80)
        
        # Initialize core components
        self.config_manager = SecureConfigManager()
        self.portfolio_visualizer = PortfolioVisualizer()
        self.analytics_engine = AdvancedAnalyticsEngine()
        self.data_fetcher = DataFetcher()
        
        # Initialize secure trading system
        self.secure_trading_system = None
        self.portfolio_agent = None
        self.trading_analyzer = None
        
        # Application state
        self.running = False
        self.monitoring_thread = None
        self.last_analysis_time = None
        self.performance_cache = {}
        
        # Validate configuration
        self.config_status = self.config_manager.validate_configuration()
        
        print("✅ Core components initialized")
        print(f"📊 Configuration Status: {self.config_status['configuration_status']}")
        print(f"🔒 Security Level: {self.config_status['security_level']}")
        print(f"📱 Wallets Configured: {self.config_status['wallet_count']}")
        
        if self.config_status['errors']:
            print("❌ Configuration Errors:")
            for error in self.config_status['errors']:
                print(f"   • {error}")
        
        if self.config_status['warnings']:
            print("⚠️  Configuration Warnings:")
            for warning in self.config_status['warnings']:
                print(f"   • {warning}")
    
    def initialize_secure_trading(self) -> bool:
        """Initialize secure trading system"""
        print("\n🔐 Initializing Secure Trading System...")
        
        try:
            # Get API key and security config
            grok_api_key = self.config_manager.get_api_key('grok_api_key')
            security_config = self.config_manager.get_security_config()
            
            if not grok_api_key:
                print("❌ GROK API key not configured")
                return False
            
            # Initialize secure trading system
            self.secure_trading_system = SecureAgenticElizaOS(
                master_key=security_config.get('master_password', 'default_key')
            )
            
            # Setup secure trading
            success = self.secure_trading_system.initialize_secure_trading(
                grok_api_key=grok_api_key,
"YOUR_PASSWORD_HERE"master_password', 'secure_trading_password_2024')
            )
            
            if not success:
                print("❌ Failed to initialize secure trading")
                return False
            
            # Initialize portfolio agent
            self.portfolio_agent = PortfolioAgent()
            
            # Initialize trading analyzer
            self.trading_analyzer = TradingHistoryAnalyzer()
            
            print("✅ Secure trading system initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing secure trading: {e}")
            return False
    
    def generate_comprehensive_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive trading dashboard"""
        print("\n📊 Generating Comprehensive Dashboard...")
        
        dashboard = {
            "timestamp": datetime.now().isoformat(),
            "configuration": self.config_manager.get_configuration_summary(),
            "portfolio_analysis": None,
            "trading_performance": None,
            "market_insights": None,
            "security_status": None,
            "recommendations": None,
            "visualizations": None,
            "alerts": []
        }
        
        try:
            # 1. Portfolio Analysis
            if self.portfolio_agent:
                print("   📈 Analyzing portfolio performance...")
                portfolio_analysis = self.portfolio_agent.force_portfolio_analysis()
                dashboard["portfolio_analysis"] = portfolio_analysis
            
            # 2. Trading Performance
            if self.trading_analyzer:
                print("   🎯 Analyzing trading performance...")
                # Import demo data for analysis
                self.trading_analyzer.import_trading_data("demo")
                performance_metrics = self.trading_analyzer.analyze_performance_metrics()
                trading_behavior = self.trading_analyzer.analyze_trading_behavior()
                
                dashboard["trading_performance"] = {
                    "metrics": performance_metrics.__dict__ if performance_metrics else {},
                    "behavior": trading_behavior.__dict__ if trading_behavior else {}
                }
            
            # 3. Market Insights
            print("   🌊 Gathering market insights...")
            market_data = self._gather_market_data()
            dashboard["market_insights"] = market_data
            
            # 4. Security Status
            if self.secure_trading_system:
                print("   🔒 Checking security status...")
                security_dashboard = self.secure_trading_system.get_security_dashboard()
                dashboard["security_status"] = security_dashboard
            
            # 5. Generate AI Recommendations
            print("   🧠 Generating AI recommendations...")
            recommendations = self.analytics_engine.generate_comprehensive_report(
                dashboard.get("portfolio_analysis", {}),
                self._get_trading_history(),
                dashboard.get("market_insights", {})
            )
            dashboard["recommendations"] = recommendations
            
            # 6. Create Visualizations
            print("   🎨 Creating visualizations...")
            visualizations = self.portfolio_visualizer.create_comprehensive_report(
                dashboard.get("portfolio_analysis", {}),
                self._get_trading_history(),
                dashboard.get("market_insights", {}),
                dashboard.get("security_status", {})
            )
            dashboard["visualizations"] = visualizations
            
            # 7. Generate Alerts
            dashboard["alerts"] = self._generate_alerts(dashboard)
            
            print("✅ Dashboard generated successfully")
            
        except Exception as e:
            print(f"❌ Error generating dashboard: {e}")
            dashboard["error"] = str(e)
        
        return dashboard
    
    def _get_trading_history(self) -> List[Dict[str, Any]]:
        """Get trading history for analysis"""
        # Generate demo trading history
        trading_history = []
        base_value = 100000.0
        
        for i in range(30):
            # Simulate trading data
            daily_change = base_value * np.random.normal(0.001, 0.02)
            new_value = base_value + daily_change
            
            trading_history.append({
                "timestamp": (datetime.now() - timedelta(days=30-i)).isoformat(),
                "total_value": new_value,
                "daily_change": daily_change,
                "daily_change_pct": (daily_change / base_value) * 100,
                "asset": "BTC" if i % 3 == 0 else "SUI" if i % 3 == 1 else "SOL",
                "action": "BUY" if daily_change > 0 else "SELL",
                "pnl": daily_change,
                "risk_amount": abs(daily_change) * 0.1,
                "confidence": np.random.uniform(0.5, 0.9)
            })
            
            base_value = new_value
        
        return trading_history
    
    def _gather_market_data(self) -> Dict[str, Any]:
        """Gather comprehensive market data"""
        market_data = {
            "timestamp": datetime.now().isoformat(),
            "assets": {},
            "volume_analysis": {},
            "sentiment": {},
            "technical_indicators": {},
            "trade_signals": []
        }
        
        assets = ['BTC', 'ETH', 'SUI', 'SOL', 'SEI']
        
        for asset in assets:
            try:
                # Fetch price and market cap
                price_data = self.data_fetcher.fetch_price_and_market_cap(asset)
                market_data["assets"][asset] = price_data
                
                # Fetch market data
                market_series = self.data_fetcher.fetch_market_data(asset, "1d")
                if market_series is not None and not market_series.empty:
                    market_data["volume_analysis"][asset] = market_series["volume"].iloc[-1] if "volume" in market_series.columns else 0
                    
                    # Calculate simple sentiment based on price momentum
                    if len(market_series) > 5:
                        recent_prices = market_series["price"].tail(5)
                        momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                        market_data["sentiment"][asset] = max(0, min(1, 0.5 + momentum * 5))
                    else:
                        market_data["sentiment"][asset] = 0.5
                
                # Generate mock trade signals
                if np.random.random() > 0.5:  # 50% chance of signal
                    market_data["trade_signals"].append({
                        "asset": asset,
                        "signal_type": np.random.choice(["BUY", "SELL"]),
                        "confidence": np.random.uniform(0.6, 0.9),
                        "timestamp": datetime.now().isoformat()
                    })
                
            except Exception as e:
                print(f"   ⚠️  Error fetching data for {asset}: {e}")
                # Provide default values
                market_data["assets"][asset] = {"price": 0, "market_cap": 0}
                market_data["volume_analysis"][asset] = 0
                market_data["sentiment"][asset] = 0.5
        
        # Mock technical indicators
        market_data["technical_indicators"] = {
            "RSI": np.random.uniform(40, 60),
            "MACD": np.random.uniform(-0.01, 0.01),
            "BB_Upper": np.random.uniform(120000, 125000),
            "BB_Lower": np.random.uniform(115000, 118000)
        }
        
        return market_data
    
    def _generate_alerts(self, dashboard: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate system alerts"""
        alerts = []
        
        # Configuration alerts
        if self.config_status['errors']:
            alerts.append({
                "type": "CONFIGURATION_ERROR",
                "severity": "HIGH",
                "message": f"{len(self.config_status['errors'])} configuration errors detected",
                "timestamp": datetime.now().isoformat(),
                "details": self.config_status['errors']
            })
        
        # Security alerts
        security_status = dashboard.get("security_status", {})
        if security_status and security_status.get("trading_status", {}).get("emergency_stop", False):
            alerts.append({
                "type": "EMERGENCY_STOP",
                "severity": "CRITICAL",
                "message": "Emergency stop activated - trading halted",
                "timestamp": datetime.now().isoformat(),
                "details": {"reason": "Portfolio protection"}
            })
        
        # Performance alerts
        recommendations = dashboard.get("recommendations", {})
        if recommendations:
            critical_insights = [
                i for i in recommendations.get("portfolio_insights", [])
                if i.priority == "CRITICAL"
            ]
            if critical_insights:
                alerts.append({
                    "type": "PERFORMANCE_ALERT",
                    "severity": "HIGH",
                    "message": f"{len(critical_insights)} critical performance issues detected",
                    "timestamp": datetime.now().isoformat(),
                    "details": [i.title for i in critical_insights]
                })
        
        return alerts
    
    def start_autonomous_trading(self) -> bool:
        """Start autonomous trading system"""
        print("\n🤖 Starting Autonomous Trading System...")
        
        if not self.secure_trading_system:
            print("❌ Secure trading system not initialized")
            return False
        
        try:
            # Start secure autonomous mode
            self.secure_trading_system.start_secure_autonomous_mode()
            
            # Start portfolio monitoring
            if self.portfolio_agent:
                self.portfolio_agent.start_autonomous_portfolio_management()
            
            self.running = True
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
            print("✅ Autonomous trading system started successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error starting autonomous trading: {e}")
            return False
    
    def stop_autonomous_trading(self):
        """Stop autonomous trading system"""
        print("\n⏹️  Stopping Autonomous Trading System...")
        
        self.running = False
        
        if self.secure_trading_system:
            self.secure_trading_system.stop_secure_autonomous_mode()
        
        if self.portfolio_agent:
            self.portfolio_agent.stop_autonomous_portfolio_management()
        
        print("✅ Autonomous trading system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Generate dashboard update
                dashboard = self.generate_comprehensive_dashboard()
                
                # Check for alerts
                alerts = dashboard.get("alerts", [])
                if alerts:
                    print(f"\n🚨 {len(alerts)} ALERTS DETECTED:")
                    for alert in alerts:
                        print(f"   {alert['severity']}: {alert['message']}")
                
                # Cache performance data
                self.performance_cache = {
                    "timestamp": datetime.now().isoformat(),
                    "portfolio_value": dashboard.get("portfolio_analysis", {}).get("metrics", {}).get("total_value", 0),
                    "daily_trades": dashboard.get("security_status", {}).get("trading_status", {}).get("daily_trades", 0),
                    "security_events": dashboard.get("security_status", {}).get("security_status", {}).get("security_events_today", 0)
                }
                
                # Sleep for monitoring interval
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def get_realtime_status(self) -> Dict[str, Any]:
        """Get real-time system status"""
        return {
            "system_running": self.running,
            "autonomous_trading": self.secure_trading_system.autonomous_mode if self.secure_trading_system else False,
            "portfolio_monitoring": self.portfolio_agent.autonomous_mode if self.portfolio_agent else False,
            "configuration_status": self.config_status['configuration_status'],
            "security_level": self.config_status['security_level'],
            "last_analysis": self.last_analysis_time,
            "performance_cache": self.performance_cache,
            "alerts_count": len(self._generate_alerts({}))
        }
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis and return results"""
        print("\n🔍 Running Comprehensive Analysis...")
        
        start_time = datetime.now()
        
        # Generate complete dashboard
        dashboard = self.generate_comprehensive_dashboard()
        
        # Add analysis metadata
        dashboard["analysis_metadata"] = {
            "analysis_time": (datetime.now() - start_time).total_seconds(),
            "components_analyzed": [
                "portfolio_performance",
                "trading_behavior",
                "market_insights",
                "security_status",
                "ai_recommendations",
                "visualizations"
            ],
            "data_sources": [
                "multi_chain_portfolio",
                "trading_history",
                "market_data",
                "security_events"
            ]
        }
        
        self.last_analysis_time = datetime.now().isoformat()
        
        print(f"✅ Analysis completed in {dashboard['analysis_metadata']['analysis_time']:.2f} seconds")
        
        return dashboard
    
    def create_setup_guide(self) -> str:
        """Create setup guide for new users"""
        guide = """
# 🚀 Enhanced Trading Application Setup Guide

## Prerequisites
- Python 3.8+
- Required packages (install with: pip install -r requirements.txt)

## Quick Setup

### 1. Environment Configuration
```bash
# Copy the environment template
cp .env.template .env

# Edit .env with your actual values
nano .env  # or use your preferred editor
```

### 2. Required Configuration
Fill in these critical values in your .env file:

**API Keys:**
- GROK_API_KEY: Your Grok-4 API key
- COINGECKO_API_KEY: CoinGecko API key (optional)

**Security:**
- MASTER_PASSWORD: Strong password for encryption
- SESSION_TIMEOUT: Session timeout in seconds (default: 3600)

**Wallet Addresses:**
- Update all wallet addresses with your actual addresses
- Ensure addresses are correct for each blockchain

### 3. Run the Application
```bash
# Initialize and run
python src/enhanced_trading_application.py

# Or run specific components
python src/portfolio_visualizer.py
python src/advanced_analytics_engine.py
python src/secure_config_manager.py
```

## Features Overview

### 🔐 Security Features
- Military-grade encryption for all sensitive data
- Multi-factor authentication system
- Comprehensive audit logging
- Real-time threat monitoring
- Emergency stop protection

### 📊 Portfolio Management
- Multi-chain portfolio analysis (SUI, Solana, Ethereum, Base, Sei)
- Real-time balance tracking
- Risk-adjusted performance metrics
- Automated rebalancing recommendations

### 🤖 AI-Powered Trading
- Grok-4 integration for market analysis
- Autonomous trading with risk management
- Behavioral analysis and pattern recognition
- Personalized strategy recommendations

### 📈 Advanced Analytics
- Performance attribution analysis
- Risk metrics and stress testing
- Anomaly detection
- Market opportunity identification

### 🎨 Visualizations
- Interactive portfolio dashboards
- Trading performance charts
- Risk analytics visualizations
- Security monitoring displays

### 🛡️ Monitoring & Alerts
- Real-time system monitoring
- Performance alerts
- Security event notifications
- Portfolio protection alerts

## Security Best Practices

1. **Strong Passwords**: Use complex passwords (20+ characters)
2. **API Key Security**: Store API keys in environment variables only
3. **Regular Backups**: Backup configuration and data regularly
4. **Access Control**: Limit system access to authorized users only
5. **Monitoring**: Review security logs regularly
6. **Updates**: Keep system updated with latest security patches

## Troubleshooting

### Common Issues:
- **Configuration Errors**: Check .env file format and values
- **API Key Issues**: Verify API keys are valid and have correct permissions
- **Wallet Connection**: Ensure wallet addresses are correct and accessible
- **Performance Issues**: Monitor system resources and optimize if needed

### Getting Help:
1. Check the logs in: enhanced_trading.log
2. Review configuration status in the dashboard
3. Verify environment variables are set correctly
4. Check network connectivity for API calls

## Advanced Configuration

### Custom Risk Settings:
```env
MAX_TRADE_AMOUNT=500.0          # Lower for conservative trading
RISK_TOLERANCE=0.01             # 1% portfolio risk
CONFIDENCE_THRESHOLD=0.8        # 80% confidence required
MAX_DAILY_TRADES=5              # Limit daily trades
```

### Performance Optimization:
```env
PERFORMANCE_MONITORING=true     # Enable performance tracking
DATABASE_POOL_SIZE=10           # Increase for better performance
CACHE_ENABLED=true              # Enable caching
```

## Support

For advanced support and customization:
- Review the comprehensive documentation
- Check the security audit logs
- Monitor performance metrics
- Consult the AI recommendations

Remember: This system handles real financial data and trading decisions. 
Always test thoroughly before live trading and ensure compliance with regulations.
"""
        
        # Save guide to file
        with open("SETUP_GUIDE.md", "w") as f:
            f.write(guide)
        
        return guide
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration for backup"""
        return {
            "timestamp": datetime.now().isoformat(),
            "configuration_summary": self.config_manager.get_configuration_summary(),
            "trading_config": self.config_manager.get_trading_config(),
            "security_config": self.config_manager.get_security_config(),
            "wallet_config": {
                "total_wallets": len(self.config_manager.wallet_configs),
                "chains_supported": list(set(w.chain for w in self.config_manager.wallet_configs))
            },
            "system_status": self.get_realtime_status()
        }


def main():
    """Main application entry point"""
    print("🚀 ENHANCED TRADING APPLICATION DEMO")
    print("=" * 80)
    
    # Initialize application
    app = EnhancedTradingApplication()
    
    # Create setup guide
    print("\n📚 Creating setup guide...")
    app.create_setup_guide()
    print("✅ Setup guide created: SETUP_GUIDE.md")
    
    # Create environment template
    print("\n🔧 Creating environment template...")
    app.config_manager.create_env_template()
    print("✅ Environment template created: .env.template")
    
    if app.config_status['valid']:
        # Initialize secure trading
        if app.initialize_secure_trading():
            # Run comprehensive analysis
            print("\n🔍 Running comprehensive analysis...")
            dashboard = app.run_comprehensive_analysis()
            
            # Display summary
            print("\n📊 ANALYSIS SUMMARY:")
            print("=" * 60)
            
            # Portfolio summary
            portfolio_analysis = dashboard.get("portfolio_analysis", {})
            if portfolio_analysis:
                metrics = portfolio_analysis.get("metrics", {})
                print(f"💰 Portfolio Value: ${metrics.get('total_value', 0):,.2f}")
                print(f"📈 Daily Change: {metrics.get('daily_change_pct', 0):+.2f}%")
                print(f"🔗 Chains: {len(metrics.get('chain_allocations', {}))}")
            
            # Recommendations summary
            recommendations = dashboard.get("recommendations", {})
            if recommendations:
                insights = recommendations.get("portfolio_insights", [])
                trading_recs = recommendations.get("trading_recommendations", [])
                print(f"🧠 Insights Generated: {len(insights)}")
                print(f"💡 Trading Recommendations: {len(trading_recs)}")
            
            # Visualizations
            visualizations = dashboard.get("visualizations", {})
            if visualizations:
                print(f"🎨 Visualizations Created: {len(visualizations)}")
                for name, path in visualizations.items():
                    print(f"   📊 {name}: {path}")
            
            # Security status
            security_status = dashboard.get("security_status", {})
            if security_status and 'error' not in security_status:
                auth_status = security_status.get("authentication", {})
                print(f"🔐 Authentication: {auth_status.get('user_id', 'N/A')}")
                print(f"🛡️  Security Events: {security_status.get('security_status', {}).get('security_events_today', 0)}")
            
            # Alerts
            alerts = dashboard.get("alerts", [])
            if alerts:
                print(f"🚨 Active Alerts: {len(alerts)}")
                for alert in alerts:
                    print(f"   {alert['severity']}: {alert['message']}")
            
            # Export configuration
            print("\n💾 Exporting configuration...")
            config_export = app.export_configuration()
            with open("config_backup.json", "w") as f:
                json.dump(config_export, f, indent=2)
            print("✅ Configuration exported: config_backup.json")
            
            print("\n🎉 ENHANCED TRADING APPLICATION DEMO COMPLETED!")
            print("=" * 80)
            print("🚀 Your comprehensive trading system is ready!")
            print("📊 Open the HTML visualizations to view interactive dashboards")
            print("📚 Review SETUP_GUIDE.md for detailed configuration instructions")
            print("🔐 Copy .env.template to .env and configure your settings")
            print("💡 Run the application with your configured environment")
            
        else:
            print("❌ Failed to initialize secure trading system")
            print("💡 Please check your configuration and try again")
    else:
        print("❌ Configuration validation failed")
        print("💡 Please review the configuration errors and warnings above")
        print("📝 Use the .env.template file to configure your environment")


if __name__ == "__main__":
    main() 