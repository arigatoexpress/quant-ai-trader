#!/usr/bin/env python3
"""
Quant AI Trader - Main Application
Enterprise-grade AI-powered trading system with free data sources integration
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import json

# Add src to path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Core imports
from data_fetcher import DataFetcher
from trading_agent import TradingAgent
from technical_analyzer import TechnicalAnalyzer
from sentiment_analysis_engine import SentimentAnalysisEngine
from risk_management_ai import RiskManagementAI
from portfolio_analyzer import PortfolioAnalyzer

# Free data sources integration
try:
    from free_data_sources import FreeDataSources, get_free_market_data
    FREE_DATA_AVAILABLE = True
except ImportError:
    FREE_DATA_AVAILABLE = False
    print("⚠️ Free data sources not available. Run: pip install -r requirements.txt")

# Enhanced systems
try:
    from singleton_manager import SingletonManager, SingletonError
    SINGLETON_AVAILABLE = True
except ImportError:
    SINGLETON_AVAILABLE = False
    print("⚠️ Singleton manager not available")

try:
    from secure_authentication import SecureAuthenticationSystem
    SECURE_AUTH_AVAILABLE = True
except ImportError:
    SECURE_AUTH_AVAILABLE = False
    print("⚠️ Secure authentication not available")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quant_trader.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class QuantAITrader:
    """
    Enterprise-grade AI-powered trading system with free data sources
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_free_data: bool = True):
        """
        Initialize the Quant AI Trader system
        
        Args:
            config: Configuration dictionary
            use_free_data: Whether to use free data sources (default: True)
        """
        self.config = config or self._load_default_config()
        self.use_free_data = use_free_data
        self.running = False
        
        # Initialize components
        self.data_fetcher = None
        self.free_data_sources = None
        self.trading_agent = None
        self.technical_analyzer = None
        self.sentiment_analyzer = None
        self.risk_manager = None
        self.portfolio_analyzer = None
        
        # Security and singleton management
        self.singleton_manager = None
        self.auth_system = None
        
        # Data cache
        self.market_data_cache = {}
        self.last_update = None
        
        logger.info("🚀 Quant AI Trader initialized")
        logger.info(f"   Free data mode: {use_free_data}")
        logger.info(f"   TradingView integration: {self._check_tradingview_available()}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "assets": ["BTC", "ETH", "SOL", "SUI", "SEI"],
            "trading": {
                "max_position_size": 0.1,
                "risk_tolerance": 0.02,
                "stop_loss_percentage": 0.05,
                "take_profit_percentage": 0.15,
                "paper_trading": True,
                "initial_portfolio_value": 10000
            },
            "data": {
                "update_interval": 300,  # 5 minutes
                "lookback_period": 100,
                "use_free_tier": True,
                "enable_tradingview": False
            },
            "ai": {
                "sentiment_analysis": True,
                "risk_management": True,
                "portfolio_optimization": True
            },
            "security": {
                "enable_2fa": True,
                "session_timeout": 3600,  # 1 hour
                "max_login_attempts": 3
            }
        }
    
    def _check_tradingview_available(self) -> bool:
        """Check if TradingView credentials are available"""
        tv_username = os.getenv('TRADINGVIEW_USERNAME')
        tv_password = os.getenv('TRADINGVIEW_PASSWORD')
        return bool(tv_username and tv_password and FREE_DATA_AVAILABLE)
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("🔄 Initializing system components...")
        
        try:
            # Initialize singleton manager
            if SINGLETON_AVAILABLE:
                self.singleton_manager = SingletonManager("quant_ai_trader")
                await self.singleton_manager.acquire_lock()
                logger.info("✅ Singleton lock acquired")
            
            # Initialize authentication system
            if SECURE_AUTH_AVAILABLE:
                self.auth_system = SecureAuthenticationSystem()
                await self.auth_system.initialize()
                logger.info("✅ Secure authentication initialized")
            
            # Initialize data sources
            if self.use_free_data and FREE_DATA_AVAILABLE:
                # Use free data sources
                tv_username = os.getenv('TRADINGVIEW_USERNAME')
                tv_password = os.getenv('TRADINGVIEW_PASSWORD')
                
                self.free_data_sources = FreeDataSources(
                    tradingview_username=tv_username,
                    tradingview_password=tv_password
                )
                await self.free_data_sources.__aenter__()
                logger.info("✅ Free data sources initialized")
                
                if tv_username and tv_password:
                    logger.info("✅ TradingView premium integration enabled")
            else:
                # Fallback to original data fetcher
                self.data_fetcher = DataFetcher()
                logger.info("✅ Standard data fetcher initialized")
            
            # Initialize AI components
            self.technical_analyzer = TechnicalAnalyzer()
            self.sentiment_analyzer = SentimentAnalysisEngine()
            self.risk_manager = RiskManagementAI()
            self.portfolio_analyzer = PortfolioAnalyzer()
            
            # Initialize trading agent
            self.trading_agent = TradingAgent(
                config=self.config,
                data_source=self.free_data_sources if self.use_free_data else self.data_fetcher
            )
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            await self.cleanup()
            raise
    
    async def get_market_data(self, symbols: list = None) -> Dict[str, Any]:
        """Get comprehensive market data"""
        if symbols is None:
            symbols = self.config["assets"]
        
        try:
            if self.use_free_data and self.free_data_sources:
                # Use free data sources
                data = await self.free_data_sources.get_comprehensive_free_data(symbols)
                
                # Cache the data
                self.market_data_cache = data
                self.last_update = datetime.now()
                
                logger.info(f"📊 Retrieved free market data for {len(symbols)} symbols")
                return data
            
            elif self.data_fetcher:
                # Use standard data fetcher
                data = {}
                for symbol in symbols:
                    symbol_data = await self.data_fetcher.fetch_real_time_data(symbol)
                    if symbol_data:
                        data[symbol] = symbol_data
                
                self.market_data_cache = data
                self.last_update = datetime.now()
                
                logger.info(f"📊 Retrieved standard market data for {len(data)} symbols")
                return data
            
            else:
                logger.warning("❌ No data sources available")
                return {}
        
        except Exception as e:
            logger.error(f"❌ Error fetching market data: {e}")
            return {}
    
    async def analyze_opportunities(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading opportunities using AI"""
        try:
            opportunities = {
                'technical_signals': {},
                'sentiment_scores': {},
                'risk_assessments': {},
                'portfolio_recommendations': {},
                'yield_opportunities': []
            }
            
            symbols = self.config["assets"]
            
            # Technical analysis
            for symbol in symbols:
                if symbol in market_data.get('validated_prices', {}):
                    price_data = market_data['validated_prices'][symbol]
                    
                    # Technical signals
                    signals = await self.technical_analyzer.analyze_symbol(symbol, price_data)
                    opportunities['technical_signals'][symbol] = signals
                    
                    # Sentiment analysis
                    sentiment = await self.sentiment_analyzer.analyze_symbol_sentiment(symbol)
                    opportunities['sentiment_scores'][symbol] = sentiment
                    
                    # Risk assessment
                    risk = await self.risk_manager.assess_symbol_risk(symbol, price_data)
                    opportunities['risk_assessments'][symbol] = risk
            
            # Portfolio optimization
            portfolio_rec = await self.portfolio_analyzer.optimize_portfolio(
                market_data, opportunities
            )
            opportunities['portfolio_recommendations'] = portfolio_rec
            
            # Yield opportunities
            if 'yield_opportunities' in market_data:
                opportunities['yield_opportunities'] = market_data['yield_opportunities']
            
            logger.info(f"🔍 Analyzed opportunities for {len(symbols)} symbols")
            return opportunities
        
        except Exception as e:
            logger.error(f"❌ Error analyzing opportunities: {e}")
            return {}
    
    async def execute_trades(self, opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trades based on opportunities (paper trading by default)"""
        try:
            if not self.trading_agent:
                logger.warning("❌ Trading agent not initialized")
                return {}
            
            # Execute trades through trading agent
            trade_results = await self.trading_agent.execute_trading_strategy(opportunities)
            
            logger.info(f"💰 Trade execution completed: {len(trade_results)} trades")
            return trade_results
        
        except Exception as e:
            logger.error(f"❌ Error executing trades: {e}")
            return {}
    
    async def run_trading_cycle(self) -> Dict[str, Any]:
        """Run one complete trading cycle"""
        cycle_start = datetime.now()
        
        try:
            logger.info("🔄 Starting trading cycle...")
            
            # 1. Fetch market data
            market_data = await self.get_market_data()
            if not market_data:
                logger.warning("⚠️ No market data available, skipping cycle")
                return {}
            
            # 2. Analyze opportunities
            opportunities = await self.analyze_opportunities(market_data)
            if not opportunities:
                logger.warning("⚠️ No opportunities found, skipping trades")
                return {'market_data': market_data}
            
            # 3. Execute trades
            trade_results = await self.execute_trades(opportunities)
            
            cycle_time = (datetime.now() - cycle_start).total_seconds()
            
            result = {
                'timestamp': datetime.now(),
                'cycle_time_seconds': cycle_time,
                'market_data': market_data,
                'opportunities': opportunities,
                'trade_results': trade_results,
                'status': 'success'
            }
            
            logger.info(f"✅ Trading cycle completed in {cycle_time:.2f}s")
            return result
        
        except Exception as e:
            logger.error(f"❌ Trading cycle failed: {e}")
            return {
                'timestamp': datetime.now(),
                'status': 'failed',
                'error': str(e)
            }
    
    async def run_continuous(self, update_interval: int = None):
        """Run continuous trading operations"""
        if update_interval is None:
            update_interval = self.config["data"]["update_interval"]
        
        self.running = True
        logger.info(f"🔄 Starting continuous trading (interval: {update_interval}s)")
        
        try:
            while self.running:
                # Run trading cycle
                cycle_result = await self.run_trading_cycle()
                
                # Log results
                if cycle_result.get('status') == 'success':
                    market_data = cycle_result.get('market_data', {})
                    trades = cycle_result.get('trade_results', {})
                    
                    logger.info(f"📊 Cycle summary:")
                    logger.info(f"   • Data sources: {len(market_data.get('metadata', {}).get('sources_used', []))}")
                    logger.info(f"   • Symbols analyzed: {len(market_data.get('market_data', []))}")
                    logger.info(f"   • Trades executed: {len(trades)}")
                
                # Wait for next cycle
                if self.running:
                    await asyncio.sleep(update_interval)
        
        except KeyboardInterrupt:
            logger.info("🛑 Received stop signal")
        except Exception as e:
            logger.error(f"❌ Continuous trading error: {e}")
        finally:
            self.running = False
            await self.cleanup()
    
    def stop(self):
        """Stop the trading system"""
        logger.info("🛑 Stopping trading system...")
        self.running = False
    
    async def cleanup(self):
        """Clean up system resources"""
        logger.info("🧹 Cleaning up system resources...")
        
        try:
            # Close data sources
            if self.free_data_sources:
                await self.free_data_sources.__aexit__(None, None, None)
            
            # Release singleton lock
            if self.singleton_manager:
                await self.singleton_manager.release_lock()
            
            logger.info("✅ Cleanup completed")
        
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

async def main():
    """Main application entry point"""
    print("🚀 Quant AI Trader - Enterprise Edition")
    print("=" * 60)
    print("🔹 Free data sources enabled")
    print("🔹 TradingView premium integration available")
    print("🔹 Enterprise security with 2FA")
    print("🔹 Asymmetric trading strategies")
    print("=" * 60)
    
    # Check environment setup
    use_free_tier = os.getenv('USE_FREE_TIER', 'true').lower() == 'true'
    paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
    
    print(f"📊 Data mode: {'Free tier' if use_free_tier else 'Premium APIs'}")
    print(f"💰 Trading mode: {'Paper trading' if paper_trading else 'Live trading'}")
    
    if use_free_tier:
        print("\n✅ Free Data Sources:")
        print("   • CoinGecko Free API")
        print("   • DexScreener API")
        print("   • CCXT Exchange APIs")
        print("   • Web scraping for yield opportunities")
        
        tv_available = os.getenv('TRADINGVIEW_USERNAME') and os.getenv('TRADINGVIEW_PASSWORD')
        if tv_available:
            print("   • TradingView Premium (configured)")
        else:
            print("   • TradingView Premium (not configured)")
    
    try:
        # Initialize trader
        trader = QuantAITrader(use_free_data=use_free_tier)
        await trader.initialize()
        
        # Run a single cycle for testing
        print("\n🔄 Running initial trading cycle...")
        result = await trader.run_trading_cycle()
        
        if result.get('status') == 'success':
            print("✅ Trading cycle completed successfully!")
            
            # Show data summary
            market_data = result.get('market_data', {})
            metadata = market_data.get('metadata', {})
            
            print(f"\n📊 Data Summary:")
            print(f"   • Fetch time: {metadata.get('fetch_time_seconds', 0):.2f}s")
            print(f"   • Sources used: {', '.join(metadata.get('sources_used', []))}")
            print(f"   • Symbols found: {metadata.get('symbols_found', 0)}")
            print(f"   • TradingView enabled: {metadata.get('tradingview_enabled', False)}")
            
            # Show opportunities
            opportunities = result.get('opportunities', {})
            print(f"   • Yield opportunities: {len(opportunities.get('yield_opportunities', []))}")
            print(f"   • Technical signals: {len(opportunities.get('technical_signals', {}))}")
        
        # Ask if user wants continuous mode
        print(f"\n🔄 Next Steps:")
        print(f"1. Test free data: python src/free_data_demo.py")
        print(f"2. Launch web dashboard: streamlit run src/web_dashboard.py")
        print(f"3. View deployment guide: cat DEPLOYMENT_GUIDE.md")
        
        # Clean up
        await trader.cleanup()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check .env configuration")
        print("2. Verify internet connection")
        print("3. Run: pip install -r requirements.txt")
        print("4. Check logs: tail -f quant_trader.log")

if __name__ == "__main__":
    # Run the main application
    asyncio.run(main()) 