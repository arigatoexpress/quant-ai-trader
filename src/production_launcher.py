#!/usr/bin/env python3
"""
Production Launcher for Quant AI Trader
Configures environment and launches the system in production mode
"""

import os
import sys
import logging
import signal
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Setup environment
os.environ["ENVIRONMENT"] = "production"
os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment from {env_file}")
    else:
        print(f"⚠️  No .env file found at {env_file}")
        print(f"📝 Please copy env_template.txt to .env and configure your settings")
except ImportError:
    print("⚠️  python-dotenv not installed, skipping .env file loading")

# Configure logging for production
def setup_production_logging():
    """Setup comprehensive logging for production"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File handler for all logs
    file_handler = logging.FileHandler(log_dir / f"trading_system_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Console handler for production
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    
    print(f"✅ Production logging configured (level: {log_level})")
    return root_logger

class ProductionSystem:
    """Main production system orchestrator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.components = {}
        self.health_check_interval = 60  # seconds
        
        # Production configuration validation
        self.required_env_vars = [
            "GROK_API_KEY",
            "MASTER_PASSWORD",
            "MAX_TRADE_AMOUNT",
            "RISK_TOLERANCE"
        ]
        
        self.logger.info("🏭 Production System initializing...")
    
    def validate_environment(self) -> bool:
        """Validate production environment configuration"""
        missing_vars = []
        
        for var in self.required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"❌ Missing required environment variables: {missing_vars}")
            self.logger.error("📝 Please check your .env file and ensure all required variables are set")
            return False
        
        # Validate trading configuration
        try:
            max_trade_amount = float(os.getenv("MAX_TRADE_AMOUNT", "0"))
            risk_tolerance = float(os.getenv("RISK_TOLERANCE", "0"))
            
            if max_trade_amount <= 0:
                self.logger.error("❌ MAX_TRADE_AMOUNT must be greater than 0")
                return False
                
            if not 0 < risk_tolerance <= 0.1:
                self.logger.error("❌ RISK_TOLERANCE must be between 0 and 0.1 (0-10%)")
                return False
                
        except ValueError as e:
            self.logger.error(f"❌ Invalid trading configuration: {e}")
            return False
        
        # Check paper trading mode
        paper_trading = os.getenv("PAPER_TRADING_MODE", "true").lower() == "true"
        autonomous_trading = os.getenv("ENABLE_AUTONOMOUS_TRADING", "false").lower() == "true"
        
        if autonomous_trading and not paper_trading:
            self.logger.warning("⚠️  AUTONOMOUS TRADING ENABLED IN LIVE MODE")
            self.logger.warning("⚠️  Ensure you understand the risks before proceeding")
            
            # Require explicit confirmation for live trading
            confirm = os.getenv("CONFIRM_LIVE_TRADING", "false").lower() == "true"
            if not confirm:
                self.logger.error("❌ Set CONFIRM_LIVE_TRADING=true to enable live autonomous trading")
                return False
        
        self.logger.info("✅ Environment validation passed")
        return True
    
    async def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            # Import components (lazy loading for better error handling)
            from production_data_fetcher import ProductionDataFetcher
            from api_rate_limiter import rate_limiter
            from enhanced_trading_application import EnhancedTradingApplication
            from cybersecurity_framework import SecureTradingFramework
            
            # Initialize components
            self.logger.info("🔄 Initializing components...")
            
            # Data fetcher with real data sources
            self.components["data_fetcher"] = ProductionDataFetcher()
            self.logger.info("✅ Production data fetcher initialized")
            
            # Rate limiter
            self.components["rate_limiter"] = rate_limiter
            self.logger.info("✅ API rate limiter initialized")
            
            # Security framework
            self.components["security"] = SecureTradingFramework()
            self.logger.info("✅ Security framework initialized")
            
            # Enhanced trading application
            self.components["trading_app"] = EnhancedTradingApplication()
            self.logger.info("✅ Enhanced trading application initialized")
            
            # Initialize secure trading if enabled
            if os.getenv("ENABLE_AUTONOMOUS_TRADING", "false").lower() == "true":
                success = self.components["trading_app"].initialize_secure_trading()
                if success:
                    self.logger.info("✅ Autonomous trading enabled")
                else:
                    self.logger.error("❌ Failed to initialize autonomous trading")
                    return False
            
            self.logger.info("✅ All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            return False
    
    async def start_web_interface(self):
        """Start the web interface"""
        try:
            from web_app import WebApp
            
            web_host = os.getenv("WEB_HOST", "127.0.0.1")
            web_port = int(os.getenv("WEB_PORT", "8000"))
            
            web_app = WebApp()
            
            self.logger.info(f"🌐 Starting web interface on {web_host}:{web_port}")
            
            # Start web app in background
            import uvicorn
            config = uvicorn.Config(
                app=web_app.app,
                host=web_host,
                port=web_port,
                log_level="info",
                access_log=True
            )
            server = uvicorn.Server(config)
            
            # Store server for graceful shutdown
            self.components["web_server"] = server
            
            # Start server in background task
            asyncio.create_task(server.serve())
            
            self.logger.info(f"✅ Web interface started: http://{web_host}:{web_port}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start web interface: {e}")
            raise
    
    async def health_check_loop(self):
        """Continuous health monitoring"""
        while self.running:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"❌ Health check failed: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def perform_health_check(self):
        """Perform comprehensive health check"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check each component
        for name, component in self.components.items():
            try:
                if hasattr(component, "get_health_status"):
                    status = await component.get_health_status()
                    health_status["components"][name] = status
                else:
                    health_status["components"][name] = {"status": "running"}
            except Exception as e:
                health_status["components"][name] = {"status": "error", "error": str(e)}
                health_status["overall_status"] = "degraded"
        
        # Check system resources
        try:
            import psutil
            health_status["system"] = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent
            }
        except ImportError:
            health_status["system"] = {"status": "monitoring_unavailable"}
        
        # Log health status
        if health_status["overall_status"] == "healthy":
            self.logger.debug("💚 System health check: All systems operational")
        else:
            self.logger.warning(f"⚠️  System health check: {health_status['overall_status']}")
        
        # Save health status to file
        health_file = Path("logs") / "health_status.json"
        with open(health_file, "w") as f:
            json.dump(health_status, f, indent=2)
    
    async def start(self):
        """Start the production system"""
        self.logger.info("🚀 Starting production system...")
        
        # Validate environment
        if not self.validate_environment():
            return False
        
        # Initialize components
        if not await self.initialize_components():
            return False
        
        # Start web interface
        await self.start_web_interface()
        
        # Start health monitoring
        self.running = True
        health_task = asyncio.create_task(self.health_check_loop())
        
        self.logger.info("✅ Production system started successfully")
        self.logger.info("📊 Dashboard: http://127.0.0.1:8000")
        self.logger.info("📝 Logs: ./logs/")
        self.logger.info("🔍 Health status: ./logs/health_status.json")
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            self.logger.info(f"📨 Received signal {signum}, initiating graceful shutdown...")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Keep running until shutdown signal
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("📨 Keyboard interrupt received")
        finally:
            self.running = False
            health_task.cancel()
            await self.shutdown()
        
        return True
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        self.logger.info("🔄 Initiating graceful shutdown...")
        
        # Stop web server
        if "web_server" in self.components:
            try:
                self.components["web_server"].should_exit = True
                self.logger.info("✅ Web server stopped")
            except Exception as e:
                self.logger.error(f"❌ Error stopping web server: {e}")
        
        # Shutdown trading components
        if "trading_app" in self.components:
            try:
                self.components["trading_app"].stop_autonomous_trading()
                self.logger.info("✅ Trading system stopped")
            except Exception as e:
                self.logger.error(f"❌ Error stopping trading system: {e}")
        
        # Clear rate limiter cache
        if "rate_limiter" in self.components:
            try:
                self.components["rate_limiter"].clear_cache()
                self.logger.info("✅ Rate limiter cache cleared")
            except Exception as e:
                self.logger.error(f"❌ Error clearing rate limiter: {e}")
        
        self.logger.info("✅ Graceful shutdown completed")

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                      🤖 QUANT AI TRADER - PRODUCTION MODE                    ║
║                                                                              ║
║  🏭 Enterprise-Grade AI Trading Platform                                     ║
║  🔐 Advanced Security & Risk Management                                      ║
║  📊 Real-Time Data & Multi-Chain Portfolio Tracking                        ║
║  🧠 Grok-4 AI Integration with Sentiment Analysis                          ║
║                                                                              ║
║  ⚠️  PRODUCTION DEPLOYMENT - REAL MONEY AT RISK                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

async def main():
    """Main entry point"""
    print_banner()
    
    # Setup logging
    logger = setup_production_logging()
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8+ required for production deployment")
        return 1
    
    # Create production system
    system = ProductionSystem()
    
    try:
        success = await system.start()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1) 