"""
Quant AI Trader - Main Application Entry Point
==============================================

This is the main entry point for the Quant AI Trader application, which integrates
all advanced AI trading components including:

- Advanced ML Models (LSTM, Transformer, XGBoost, etc.)
- Sentiment Analysis Engine
- Maximum Profit Asymmetric Trading Framework
- Grok 4 Data Integration
- Risk Management with AI-driven VaR
- Performance Attribution
- Security and Monitoring
- Backtesting Framework
- Singleton Instance Management

The application follows a modular architecture with clear separation of concerns
and comprehensive error handling for production deployment.

Author: AI Assistant
Version: 3.0.0
License: MIT
"""

import asyncio
import logging
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import singleton management first
from singleton_manager import ensure_single_instance, set_global_singleton, get_global_singleton

# Import core trading components
from trading_agent import TradingAgent
from data_fetcher import DataFetcher
from technical_analyzer import TechnicalAnalyzer
from macro_analyzer import MacroAnalyzer
from news_fetcher import NewsFetcher
from onchain_analyzer import OnchainAnalyzer
from eliza_os import ElizaOS

# Import advanced AI components
from advanced_ml_models import AdvancedMLModels
from sentiment_analysis_engine import SentimentAnalysisEngine
from asymmetric_trading_framework import MaxProfitTradingFramework
from grok4_data_integration import Grok4DataIntegration
from risk_management_ai import RiskManagementAI
from performance_attribution import PerformanceAttribution
from ai_integration_framework import AIIntegrationFramework

# Import security and monitoring components
from security_audit_cleanup import SecurityAuditor
from comprehensive_testing_framework import ComprehensiveTestRunner
from deployment_validation import DeploymentValidator

# Import utility components
from utils import setup_logging, load_config, validate_config
from web_app import WebApp

# Configure logging for the main application
logger = logging.getLogger(__name__)

class QuantAITrader:
    """
    Main Quant AI Trader application class that orchestrates all trading components.
    
    This class serves as the central coordinator for:
    - Data collection and processing
    - AI model predictions and analysis
    - Trading signal generation and execution
    - Risk management and portfolio optimization
    - Performance monitoring and reporting
    - Security and compliance checks
    - Singleton instance management
    
    The application follows an event-driven architecture with real-time processing
    capabilities and comprehensive error handling.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Quant AI Trader application.
        
        Args:
            config_path: Path to the configuration file containing all trading parameters,
                        API keys, model configurations, and system settings.
        
        The initialization process:
        1. Ensures singleton instance
        2. Loads and validates configuration
        3. Sets up logging and monitoring
        4. Initializes all trading components
        5. Establishes data connections
        6. Validates system requirements
        """
        self.config_path = config_path
        self.config = None
        self.is_running = False
        self.start_time = None
        self.singleton_manager = None
        
        # Core trading components
        self.trading_agent = None
        self.data_fetcher = None
        self.technical_analyzer = None
        self.macro_analyzer = None
        self.news_fetcher = None
        self.onchain_analyzer = None
        self.eliza_os = None
        
        # Advanced AI components
        self.advanced_ml_models = None
        self.sentiment_engine = None
        self.asymmetric_framework = None
        self.grok4_integration = None
        self.risk_management = None
        self.performance_attribution = None
        self.ai_integration = None
        
        # Security and monitoring
        self.security_auditor = None
        self.test_runner = None
        self.deployment_validator = None
        self.web_app = None
        
        # Performance tracking
        self.performance_metrics = {
            'start_time': None,
            'trades_executed': 0,
            'total_profit_loss': 0.0,
            'successful_trades': 0,
            'failed_trades': 0,
            'uptime_hours': 0.0,
            'data_points_processed': 0,
            'ai_predictions_made': 0,
            'risk_events_detected': 0
        }
        
        logger.info("🚀 Quant AI Trader initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the Quant AI Trader application with singleton management.
        
        This method ensures only one instance runs and initializes all components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("🚀 Initializing Quant AI Trader...")
            
            # Step 1: Ensure singleton instance
            logger.info("🔒 Ensuring singleton instance...")
            try:
                self.singleton_manager = ensure_single_instance("quant_ai_trader")
                set_global_singleton(self.singleton_manager)
                logger.info("✅ Singleton lock acquired successfully")
            except RuntimeError as e:
                logger.error(f"❌ {str(e)}")
                return False
            
            # Step 2: Load and validate configuration
            logger.info("📋 Loading configuration...")
            self.config = load_config(self.config_path)
            if not validate_config(self.config):
                logger.error("❌ Configuration validation failed")
                return False
            
            # Step 3: Set up logging
            setup_logging(self.config.get('logging', {}))
            logger.info("✅ Configuration loaded successfully")
            
            # Step 4: Initialize core trading components
            logger.info("🔧 Initializing core trading components...")
            await self._initialize_core_components()
            
            # Step 5: Initialize advanced AI components
            logger.info("🤖 Initializing advanced AI components...")
            await self._initialize_ai_components()
            
            # Step 6: Initialize security and monitoring
            logger.info("🔒 Initializing security and monitoring...")
            await self._initialize_security_components()
            
            # Step 7: Initialize web interface
            logger.info("🌐 Initializing web interface...")
            await self._initialize_web_interface()
            
            # Step 8: Validate deployment
            logger.info("✅ Validating deployment...")
            if not await self._validate_deployment():
                logger.error("❌ Deployment validation failed")
                return False
            
            logger.info("🎉 Quant AI Trader initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    async def _initialize_core_components(self):
        """
        Initialize core trading components including data fetchers, analyzers, and trading agent.
        
        This method sets up the fundamental trading infrastructure:
        - Data collection from multiple sources (market data, news, onchain data)
        - Technical and fundamental analysis engines
        - Trading agent for signal generation and execution
        - Eliza OS for system orchestration
        """
        try:
            # Initialize data fetcher for market data collection
            self.data_fetcher = DataFetcher(self.config.get('data_sources', {}))
            await self.data_fetcher.initialize()
            
            # Initialize technical analyzer for price pattern recognition
            self.technical_analyzer = TechnicalAnalyzer(self.config.get('technical_analysis', {}))
            
            # Initialize macro analyzer for fundamental analysis
            self.macro_analyzer = MacroAnalyzer(self.config.get('macro_analysis', {}))
            
            # Initialize news fetcher for sentiment data
            self.news_fetcher = NewsFetcher(self.config.get('news_sources', {}))
            
            # Initialize onchain analyzer for blockchain data
            self.onchain_analyzer = OnchainAnalyzer(self.config.get('onchain_analysis', {}))
            
            # Initialize trading agent for signal generation and execution
            self.trading_agent = TradingAgent(
                config=self.config.get('trading', {}),
                data_fetcher=self.data_fetcher,
                technical_analyzer=self.technical_analyzer,
                macro_analyzer=self.macro_analyzer
            )
            
            # Initialize Eliza OS for system orchestration
            self.eliza_os = ElizaOS(self.config.get('eliza_os', {}))
            
            logger.info("✅ Core components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Core components initialization failed: {str(e)}")
            raise
    
    async def _initialize_ai_components(self):
        """Initialize advanced AI components."""
        try:
            # Initialize advanced ML models
            self.advanced_ml_models = AdvancedMLModels(self.config_path)
            await self.advanced_ml_models.initialize()
            
            # Initialize sentiment analysis engine
            self.sentiment_engine = SentimentAnalysisEngine(self.config_path)
            await self.sentiment_engine.initialize()
            
            # Initialize maximum profit asymmetric trading framework
            self.asymmetric_framework = MaxProfitTradingFramework(
                self.config, self.advanced_ml_models, self.sentiment_engine
            )
            await self.asymmetric_framework.initialize()
            
            # Initialize GROK 4 data integration
            self.grok4_integration = Grok4DataIntegration(self.config_path)
            await self.grok4_integration.initialize()
            
            # Initialize risk management AI
            self.risk_management = RiskManagementAI(self.config_path)
            await self.risk_management.initialize()
            
            # Initialize performance attribution
            self.performance_attribution = PerformanceAttribution(self.config_path)
            
            # Initialize AI integration framework
            self.ai_integration = AIIntegrationFramework(self.config_path)
            await self.ai_integration.initialize()
            
            logger.info("✅ Advanced AI components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI components: {str(e)}")
            raise
    
    async def _initialize_security_components(self):
        """
        Initialize security and monitoring components for system protection.
        
        This method sets up comprehensive security measures:
        - Security auditor for continuous monitoring
        - Comprehensive testing framework for system validation
        - Deployment validator for environment checks
        """
        try:
            # Initialize security auditor
            self.security_auditor = SecurityAuditor(".")
            
            # Initialize comprehensive testing framework
            self.test_runner = ComprehensiveTestRunner()
            
            # Initialize deployment validator
            self.deployment_validator = DeploymentValidator(self.config)
            
            logger.info("✅ Security components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Security components initialization failed: {str(e)}")
            raise
    
    async def _initialize_web_interface(self):
        """
        Initialize web interface for monitoring and control.
        
        This method sets up the web dashboard for:
        - Real-time trading performance monitoring
        - Portfolio visualization and analysis
        - System status and health checks
        - Manual trading controls and overrides
        """
        try:
            self.web_app = WebApp(
                config=self.config.get('web_app', {}),
                trading_agent=self.trading_agent,
                performance_metrics=self.performance_metrics
            )
            await self.web_app.initialize()
            
            logger.info("✅ Web interface initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Web interface initialization failed: {str(e)}")
            raise
    
    async def _validate_deployment(self) -> bool:
        """
        Validate the deployment environment and system requirements.
        
        Returns:
            bool: True if deployment is valid, False otherwise
            
        This method performs comprehensive validation:
        - System requirements check
        - Component initialization validation
        - Data connectivity verification
        - Model validation and testing
        - Security validation
        - Performance benchmarks
        """
        try:
            # Run deployment validation
            validation_result = await self.deployment_validator.validate_deployment()
            
            if not validation_result['success']:
                logger.error(f"❌ Deployment validation failed: {validation_result['errors']}")
                return False
            
            # Run security audit
            security_result = await self.security_auditor.run_full_audit()
            if not security_result['is_safe_for_github']:
                logger.warning("⚠️ Security audit found issues - review before deployment")
            
            # Run comprehensive tests
            test_result = await self.test_runner.run_comprehensive_tests()
            if test_result['success_rate'] < 0.8:
                logger.warning(f"⚠️ Test success rate below 80%: {test_result['success_rate']:.1%}")
            
            logger.info("✅ Deployment validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Deployment validation failed: {str(e)}")
            return False
    
    async def start(self):
        """
        Start the Quant AI Trader application.
        
        This method initiates the main trading loop and starts all services:
        1. Starts the trading agent
        2. Initiates data collection streams
        3. Starts AI model predictions
        4. Launches web interface
        5. Begins monitoring and reporting
        6. Starts health monitoring for singleton
        """
        try:
            logger.info("🚀 Starting Quant AI Trader...")
            
            if not self.is_running:
                self.is_running = True
                self.start_time = datetime.now()
                self.performance_metrics['start_time'] = self.start_time
                
                # Start trading agent
                await self.trading_agent.start()
                
                # Start data collection
                await self.data_fetcher.start_streaming()
                
                # Start AI components
                await self.ai_integration.start()
                
                # Start web interface
                await self.web_app.start()
                
                # Start monitoring loops
                asyncio.create_task(self._monitoring_loop())
                asyncio.create_task(self._health_monitoring_loop())
                
                logger.info("🎉 Quant AI Trader started successfully!")
                logger.info(f"📊 Web interface available at: {self.config.get('web_app', {}).get('host', 'localhost')}:{self.config.get('web_app', {}).get('port', 8080)}")
                logger.info(f"🔒 Singleton PID: {self.singleton_manager.current_pid}")
                
        except Exception as e:
            logger.error(f"❌ Failed to start Quant AI Trader: {str(e)}")
            logger.error(traceback.format_exc())
            await self.stop()
    
    async def _monitoring_loop(self):
        """
        Main monitoring loop for system health and performance tracking.
        
        This loop continuously monitors:
        - System health and component status
        - Trading performance metrics
        - Risk management alerts
        - Security monitoring
        - Performance attribution updates
        """
        while self.is_running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Check system health
                await self._check_system_health()
                
                # Update risk management
                await self._update_risk_management()
                
                # Update performance attribution
                await self._update_performance_attribution()
                
                # Sleep for monitoring interval
                await asyncio.sleep(self.config.get('monitoring', {}).get('interval', 60))
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {str(e)}")
                await asyncio.sleep(10)  # Shorter sleep on error
    
    async def _update_performance_metrics(self):
        """
        Update performance metrics with current trading data.
        
        This method calculates and updates:
        - Total trades and success rate
        - P&L and drawdown metrics
        - Sharpe ratio and other risk-adjusted returns
        - System uptime and performance statistics
        """
        try:
            if self.trading_agent:
                # Get current trading metrics
                metrics = await self.trading_agent.get_performance_metrics()
                
                # Update performance metrics
                self.performance_metrics.update(metrics)
                self.performance_metrics['last_update'] = datetime.now()
                
                # Log significant events
                if metrics.get('total_trades', 0) > self.performance_metrics.get('total_trades', 0):
                    logger.info(f"📈 New trade executed - Total P&L: ${metrics.get('total_pnl', 0):.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error updating performance metrics: {str(e)}")
    
    async def _check_system_health(self):
        """
        Check system health and component status.
        
        This method monitors:
        - Data connection health
        - AI model performance
        - Trading agent status
        - Risk management alerts
        - System resource usage
        """
        try:
            # Check data connections
            if self.data_fetcher and not await self.data_fetcher.is_healthy():
                logger.warning("⚠️ Data fetcher health check failed")
            
            # Check AI components
            if self.ai_integration and not await self.ai_integration.is_healthy():
                logger.warning("⚠️ AI integration health check failed")
            
            # Check trading agent
            if self.trading_agent and not await self.trading_agent.is_healthy():
                logger.warning("⚠️ Trading agent health check failed")
            
        except Exception as e:
            logger.error(f"❌ Error checking system health: {str(e)}")
    
    async def _update_risk_management(self):
        """
        Update risk management calculations and alerts.
        
        This method updates:
        - VaR calculations
        - Position sizing recommendations
        - Risk alerts and warnings
        - Portfolio stress testing
        """
        try:
            if self.risk_management:
                await self.risk_management.update_risk_metrics()
                
                # Check for risk alerts
                alerts = await self.risk_management.get_risk_alerts()
                for alert in alerts:
                    logger.warning(f"⚠️ Risk Alert: {alert}")
                    
        except Exception as e:
            logger.error(f"❌ Error updating risk management: {str(e)}")
    
    async def _update_performance_attribution(self):
        """
        Update performance attribution analysis.
        
        This method analyzes:
        - Sources of alpha and beta
        - Factor contributions
        - Risk factor exposures
        - Performance decomposition
        """
        try:
            if self.performance_attribution:
                await self.performance_attribution.update_attribution()
                
        except Exception as e:
            logger.error(f"❌ Error updating performance attribution: {str(e)}")
    
    async def _save_performance_metrics(self):
        """
        Save final performance metrics to persistent storage.
        
        This method saves:
        - Complete trading history
        - Performance statistics
        - Risk metrics
        - Attribution analysis
        """
        try:
            # Calculate final metrics
            if self.start_time:
                runtime = datetime.now() - self.start_time
                self.performance_metrics['runtime'] = str(runtime)
            
            # Save to file
            import json
            with open('performance_metrics.json', 'w') as f:
                json.dump(self.performance_metrics, f, indent=2, default=str)
            
            logger.info("✅ Performance metrics saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Error saving performance metrics: {str(e)}")
    
    async def _health_monitoring_loop(self):
        """Monitor singleton health and update health file."""
        while self.is_running:
            try:
                if self.singleton_manager:
                    self.singleton_manager.update_health()
                    
                await asyncio.sleep(30)  # Update health every 30 seconds
                
            except Exception as e:
                logger.warning(f"Health monitoring error: {str(e)}")
                await asyncio.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status and performance metrics.
        
        Returns:
            Dict containing system status, performance metrics, and component health
        """
        return {
            'is_running': self.is_running,
            'start_time': self.start_time,
            'performance_metrics': self.performance_metrics,
            'components': {
                'trading_agent': self.trading_agent is not None,
                'data_fetcher': self.data_fetcher is not None,
                'ai_integration': self.ai_integration is not None,
                'web_app': self.web_app is not None
            }
        }

async def main():
    """
    Main entry point for the Quant AI Trader application.
    
    This function:
    1. Creates and initializes the QuantAITrader instance
    2. Starts the application
    3. Handles graceful shutdown on interrupt
    4. Provides command-line interface for basic operations
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Quant AI Trader')
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--test', action='store_true', help='Run tests only')
    parser.add_argument('--validate', action='store_true', help='Validate deployment only')
    args = parser.parse_args()
    
    # Create trader instance
    trader = QuantAITrader(args.config)
    
    try:
        # Initialize the application
        if not await trader.initialize():
            logger.error("❌ Failed to initialize Quant AI Trader")
            sys.exit(1)
        
        # Handle different modes
        if args.test:
            logger.info("🧪 Running tests only...")
            test_runner = ComprehensiveTestRunner()
            result = await test_runner.run_comprehensive_tests()
            print(f"Test Results: {result['success_rate']:.1%} success rate")
            sys.exit(0 if result['success_rate'] >= 0.8 else 1)
        
        elif args.validate:
            logger.info("✅ Running validation only...")
            validator = DeploymentValidator(trader.config)
            result = await validator.validate_deployment()
            print(f"Validation Results: {'PASS' if result['success'] else 'FAIL'}")
            sys.exit(0 if result['success'] else 1)
        
        else:
            # Start the application
            await trader.start()
            
            # Keep the application running
            try:
                while trader.is_running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal, shutting down...")
                await trader.stop()
    
    except Exception as e:
        logger.error(f"❌ Application error: {str(e)}")
        logger.error(traceback.format_exc())
        await trader.stop()
        sys.exit(1)

if __name__ == "__main__":
    # Run the main application
    asyncio.run(main()) 