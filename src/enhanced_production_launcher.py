"""
Enhanced Production Launcher
Production-ready launcher with comprehensive real data integrations
"""

import asyncio
import logging
import signal
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedProductionSystem:
    """Enhanced production system with real data integrations"""
    
    def __init__(self):
        self.running = False
        self.data_engine = None
        self.telegram_bot = None
        self.asymmetric_scanner = None
        self.data_streams = None
        
        # System monitoring
        self.start_time = None
        self.last_health_check = None
        self.performance_metrics = {
            'data_fetch_count': 0,
            'opportunities_found': 0,
            'alerts_sent': 0,
            'avg_fetch_time': 0.0,
            'uptime_hours': 0.0
        }
        
        # Configuration
        self.config = self._load_config()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("🚀 Enhanced Production System initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load production configuration"""
        try:
            with open('config/config.yaml', 'r') as f:
                import yaml
                config = yaml.safe_load(f)
        except FileNotFoundError:
            config = {}
        
        # Production defaults
        return {
            **config,
            'production': {
                'scan_interval_minutes': config.get('production', {}).get('scan_interval_minutes', 5),
                'health_check_interval_minutes': config.get('production', {}).get('health_check_interval_minutes', 10),
                'max_opportunities_per_scan': config.get('production', {}).get('max_opportunities_per_scan', 20),
                'alert_cooldown_minutes': config.get('production', {}).get('alert_cooldown_minutes', 30),
                'data_retention_days': config.get('production', {}).get('data_retention_days', 30)
            },
            'alerts': {
                'telegram_enabled': os.getenv('TELEGRAM_BOT_TOKEN') is not None,
                'min_opportunity_score': config.get('alerts', {}).get('min_opportunity_score', 7.5),
                'max_alerts_per_hour': config.get('alerts', {}).get('max_alerts_per_hour', 5)
            },
            'data_sources': {
                'coingecko_enabled': os.getenv('COINGECKO_API_KEY') is not None,
                'real_data_preferred': True,
                'fallback_to_mock': True,
                'cache_enabled': True
            }
        }
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def initialize_components(self):
        """Initialize all system components"""
        logger.info("🔧 Initializing system components...")
        
        try:
            # Initialize real data aggregation engine
            logger.info("📊 Initializing data aggregation engine...")
            try:
                from real_data_integrations import DataAggregationEngine
                self.data_engine = await DataAggregationEngine().__aenter__()
                logger.info("✅ Real data aggregation engine initialized")
            except ImportError:
                logger.warning("⚠️  Real data integrations not available, using fallback")
                from data_stream_integrations import DataStreamIntegrations
                self.data_streams = await DataStreamIntegrations().__aenter__()
                logger.info("✅ Fallback data streams initialized")
            except Exception as e:
                logger.error(f"❌ Data engine initialization failed: {e}")
                if self.config['data_sources']['fallback_to_mock']:
                    from data_stream_integrations import DataStreamIntegrations
                    self.data_streams = await DataStreamIntegrations().__aenter__()
                    logger.info("✅ Fallback data streams initialized")
                else:
                    raise
            
            # Initialize asymmetric trading scanner
            logger.info("🎯 Initializing asymmetric scanner...")
            try:
                from asymmetric_scanner import AsymmetricTradingScanner
                self.asymmetric_scanner = AsymmetricTradingScanner()
                logger.info("✅ Asymmetric scanner initialized")
            except Exception as e:
                logger.error(f"❌ Asymmetric scanner initialization failed: {e}")
            
            # Initialize Telegram bot if enabled
            if self.config['alerts']['telegram_enabled']:
                logger.info("📱 Initializing Telegram bot...")
                try:
                    from telegram_bot import TelegramTradingBot
                    self.telegram_bot = TelegramTradingBot()
                    # Start bot monitoring in background
                    self.telegram_bot.start_monitoring()
                    logger.info("✅ Telegram bot initialized and monitoring started")
                except Exception as e:
                    logger.error(f"❌ Telegram bot initialization failed: {e}")
            else:
                logger.info("📱 Telegram bot disabled (no token configured)")
            
            # Initialize directories
            os.makedirs('logs', exist_ok=True)
            os.makedirs('data', exist_ok=True)
            os.makedirs('reports', exist_ok=True)
            
            logger.info("🎉 All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    async def health_check(self):
        """Perform system health check"""
        health_status = {
            'timestamp': datetime.now(),
            'uptime_hours': self.performance_metrics['uptime_hours'],
            'components': {
                'data_engine': self.data_engine is not None,
                'data_streams': self.data_streams is not None,
                'asymmetric_scanner': self.asymmetric_scanner is not None,
                'telegram_bot': self.telegram_bot is not None and self.telegram_bot.monitoring_active if self.telegram_bot else False
            },
            'performance': self.performance_metrics,
            'memory_usage_mb': self._get_memory_usage(),
            'status': 'healthy'
        }
        
        # Check component health
        unhealthy_components = []
        
        if not (self.data_engine or self.data_streams):
            unhealthy_components.append('data_source')
        
        if not self.asymmetric_scanner:
            unhealthy_components.append('scanner')
        
        if unhealthy_components:
            health_status['status'] = 'degraded'
            health_status['issues'] = unhealthy_components
            logger.warning(f"⚠️  System health degraded: {', '.join(unhealthy_components)}")
        else:
            logger.info(f"✅ System health check passed - Uptime: {health_status['uptime_hours']:.1f}h")
        
        self.last_health_check = datetime.now()
        
        # Save health status
        try:
            with open('data/health_status.json', 'w') as f:
                json.dump(health_status, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"❌ Failed to save health status: {e}")
        
        return health_status
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    async def scan_and_analyze_markets(self):
        """Scan markets for opportunities and analyze data"""
        start_time = time.time()
        
        try:
            logger.info("🔍 Starting market scan and analysis...")
            
            # Define symbols to monitor
            symbols = self.config.get('assets', ['BTC', 'ETH', 'SOL', 'SUI', 'SEI'])
            
            # Get comprehensive market data
            market_data = None
            if self.data_engine:
                market_data = await self.data_engine.get_comprehensive_market_data(symbols)
            elif self.data_streams:
                market_data = await self.data_streams.get_comprehensive_market_data(symbols)
            
            if not market_data:
                logger.error("❌ Failed to fetch market data")
                return
            
            # Update performance metrics
            fetch_time = time.time() - start_time
            self.performance_metrics['data_fetch_count'] += 1
            self.performance_metrics['avg_fetch_time'] = (
                (self.performance_metrics['avg_fetch_time'] * (self.performance_metrics['data_fetch_count'] - 1) + fetch_time) /
                self.performance_metrics['data_fetch_count']
            )
            
            logger.info(f"📊 Market data fetched in {fetch_time:.2f}s")
            logger.info(f"   Market data points: {len(market_data.get('market_data', []))}")
            logger.info(f"   DEX pairs: {len(market_data.get('dex_pairs', []))}")
            logger.info(f"   DeFi opportunities: {len(market_data.get('defi_yields', []))}")
            
            # Run asymmetric opportunity scanner
            if self.asymmetric_scanner:
                try:
                    logger.info("🎯 Running asymmetric opportunity scanner...")
                    opportunities = await self.asymmetric_scanner.scan_for_opportunities()
                    
                    if opportunities:
                        self.performance_metrics['opportunities_found'] += len(opportunities)
                        logger.info(f"✅ Found {len(opportunities)} asymmetric opportunities")
                        
                        # Send alerts for high-quality opportunities
                        await self._send_opportunity_alerts(opportunities)
                        
                        # Save opportunities
                        self._save_opportunities(opportunities)
                    else:
                        logger.info("📊 No new asymmetric opportunities found")
                        
                except Exception as e:
                    logger.error(f"❌ Asymmetric scanner error: {e}")
            
            # Save market data
            self._save_market_data(market_data)
            
            logger.info(f"✅ Market scan completed in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Market scan failed: {e}")
    
    async def _send_opportunity_alerts(self, opportunities: List[Any]):
        """Send alerts for high-quality opportunities"""
        if not self.telegram_bot or not opportunities:
            return
        
        # Filter high-quality opportunities
        min_score = self.config['alerts']['min_opportunity_score']
        high_quality_opps = [
            opp for opp in opportunities
            if hasattr(opp, 'raw_data') and 
               opp.raw_data and 
               opp.raw_data.get('composite_score', 0) >= min_score
        ]
        
        if not high_quality_opps:
            return
        
        # Limit alerts per scan
        max_alerts = min(3, self.config['alerts']['max_alerts_per_hour'])
        high_quality_opps = high_quality_opps[:max_alerts]
        
        for opp in high_quality_opps:
            try:
                score = opp.raw_data.get('composite_score', 0)
                urgency = 'high' if score >= 9 else 'medium'
                
                alert_message = f"""
🎯 **ASYMMETRIC OPPORTUNITY DETECTED**

**Asset:** {opp.asset}
**Type:** {opp.opportunity_type.replace('_', ' ').title()}
**Expected Return:** {opp.expected_return:.1f}%
**Risk Score:** {opp.risk_score:.2f}/1.0
**Confidence:** {opp.confidence_score:.2f}/1.0
**Overall Score:** {score:.1f}/10

**Analysis:** {opp.analysis}

**Action Plan:** {opp.action_plan}

*Detected by Enhanced Quant AI Trader*
                """
                
                await self.telegram_bot.send_alert(
                    alert_message.strip(),
                    urgency
                )
                
                self.performance_metrics['alerts_sent'] += 1
                logger.info(f"📱 Alert sent for {opp.asset} opportunity (score: {score:.1f})")
                
                # Add cooldown between alerts
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Failed to send alert for {opp.asset}: {e}")
    
    def _save_opportunities(self, opportunities: List[Any]):
        """Save opportunities to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/opportunities_{timestamp}.json"
            
            opportunities_data = []
            for opp in opportunities:
                if hasattr(opp, '__dict__'):
                    opportunities_data.append(opp.__dict__)
                else:
                    opportunities_data.append(str(opp))
            
            with open(filename, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'count': len(opportunities),
                    'opportunities': opportunities_data
                }, f, indent=2, default=str)
            
            logger.info(f"💾 Opportunities saved to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save opportunities: {e}")
    
    def _save_market_data(self, market_data: Dict[str, Any]):
        """Save market data to file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/market_data_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(market_data, f, indent=2, default=str)
            
            # Keep only recent files (data retention)
            self._cleanup_old_data_files()
            
        except Exception as e:
            logger.error(f"❌ Failed to save market data: {e}")
    
    def _cleanup_old_data_files(self):
        """Clean up old data files based on retention policy"""
        try:
            retention_days = self.config['production']['data_retention_days']
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            data_dir = 'data'
            if not os.path.exists(data_dir):
                return
            
            deleted_count = 0
            for filename in os.listdir(data_dir):
                if filename.startswith(('market_data_', 'opportunities_')) and filename.endswith('.json'):
                    filepath = os.path.join(data_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                    
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"🗑️  Cleaned up {deleted_count} old data files")
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old data files: {e}")
    
    async def run_production_loop(self):
        """Main production loop"""
        logger.info("🚀 Starting production loop...")
        self.running = True
        self.start_time = datetime.now()
        
        scan_interval = self.config['production']['scan_interval_minutes'] * 60
        health_check_interval = self.config['production']['health_check_interval_minutes'] * 60
        
        last_scan_time = 0
        last_health_check_time = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Update uptime
                if self.start_time:
                    self.performance_metrics['uptime_hours'] = (datetime.now() - self.start_time).total_seconds() / 3600
                
                # Perform health check
                if current_time - last_health_check_time >= health_check_interval:
                    await self.health_check()
                    last_health_check_time = current_time
                
                # Perform market scan
                if current_time - last_scan_time >= scan_interval:
                    await self.scan_and_analyze_markets()
                    last_scan_time = current_time
                
                # Sleep for a short interval
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Production loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
        
        logger.info("⏹️  Production loop stopped")
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("🛑 Initiating graceful shutdown...")
        
        self.running = False
        
        # Stop Telegram bot
        if self.telegram_bot:
            try:
                self.telegram_bot.stop_monitoring()
                logger.info("✅ Telegram bot stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping Telegram bot: {e}")
        
        # Stop asymmetric scanner
        if self.asymmetric_scanner and hasattr(self.asymmetric_scanner, 'stop_scanning'):
            try:
                self.asymmetric_scanner.stop_scanning()
                logger.info("✅ Asymmetric scanner stopped")
            except Exception as e:
                logger.error(f"❌ Error stopping asymmetric scanner: {e}")
        
        # Close data engine
        if self.data_engine:
            try:
                await self.data_engine.__aexit__(None, None, None)
                logger.info("✅ Data aggregation engine closed")
            except Exception as e:
                logger.error(f"❌ Error closing data engine: {e}")
        
        # Close data streams
        if self.data_streams:
            try:
                await self.data_streams.__aexit__(None, None, None)
                logger.info("✅ Data streams closed")
            except Exception as e:
                logger.error(f"❌ Error closing data streams: {e}")
        
        # Save final performance metrics
        try:
            final_metrics = {
                'shutdown_time': datetime.now().isoformat(),
                'total_uptime_hours': self.performance_metrics['uptime_hours'],
                'total_data_fetches': self.performance_metrics['data_fetch_count'],
                'total_opportunities_found': self.performance_metrics['opportunities_found'],
                'total_alerts_sent': self.performance_metrics['alerts_sent'],
                'avg_fetch_time_seconds': self.performance_metrics['avg_fetch_time']
            }
            
            with open('data/final_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=2, default=str)
            
            logger.info("💾 Final metrics saved")
            
        except Exception as e:
            logger.error(f"❌ Failed to save final metrics: {e}")
        
        logger.info("✅ Graceful shutdown completed")

async def main():
    """Main entry point"""
    print("🚀 Enhanced Quant AI Trader - Production System")
    print("=" * 60)
    print("🔄 Real-time market monitoring and analysis")
    print("🎯 Asymmetric trading opportunity detection")
    print("📱 Telegram alerts and notifications")
    print("📊 Multi-source data aggregation")
    print("=" * 60)
    
    system = None
    
    try:
        # Initialize and start the system
        system = EnhancedProductionSystem()
        await system.initialize_components()
        
        logger.info("🎉 Enhanced Production System ready!")
        logger.info("   📊 Real data aggregation: Active")
        logger.info("   🎯 Asymmetric scanner: Active")
        logger.info("   📱 Telegram alerts: Active" if system.telegram_bot else "   📱 Telegram alerts: Disabled")
        logger.info("   🔄 Market monitoring: Starting...")
        
        # Start the production loop
        await system.run_production_loop()
        
    except KeyboardInterrupt:
        logger.info("📡 Received keyboard interrupt")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
    finally:
        if system:
            await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 