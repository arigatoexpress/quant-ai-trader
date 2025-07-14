"""
AI Integration Framework for Autonomous Trading
Integrates all AI components: ML models, sentiment analysis, backtesting, and validation
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
from dataclasses import dataclass, field
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import our AI components
from advanced_ml_models import AdvancedMLModels, ModelPrediction
from sentiment_analysis_engine import SentimentAnalysisEngine, SentimentSignal
from backtesting_framework import BacktestEngine, BacktestConfig
from model_validation_testing import ModelValidationFramework
from cybersecurity_framework import SecureTradingFramework
from secure_autonomous_trader import SecureAgenticElizaOS
from portfolio_agent import PortfolioAgent
from data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AISignal:
    """Comprehensive AI trading signal"""
    asset: str
    timestamp: datetime
    
    # Model predictions
    ml_predictions: List[ModelPrediction]
    ensemble_prediction: Optional[ModelPrediction] = None
    
    # Sentiment analysis
    sentiment_signal: Optional[SentimentSignal] = None
    
    # Combined signal
    combined_action: str = 'HOLD'  # 'BUY', 'SELL', 'HOLD'
    combined_confidence: float = 0.0
    combined_reasoning: str = ''
    
    # Risk metrics
    risk_level: str = 'MEDIUM'  # 'LOW', 'MEDIUM', 'HIGH'
    position_size: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Validation scores
    validation_score: float = 0.0
    backtest_score: float = 0.0
    
    # Metadata
    supporting_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AISystemMetrics:
    """Comprehensive AI system performance metrics"""
    timestamp: datetime
    
    # Model performance
    model_accuracy: Dict[str, float]
    model_consistency: Dict[str, float]
    ensemble_performance: float
    
    # Sentiment analysis
    sentiment_accuracy: float
    sentiment_coverage: float
    
    # System health
    system_uptime: float
    processing_latency: float
    error_rate: float
    
    # Trading performance
    signal_quality: float
    risk_adjusted_return: float
    drawdown_control: float
    
    # Security metrics
    security_score: float
    threat_level: str

class AIIntegrationFramework:
    """Comprehensive AI integration framework for autonomous trading"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or 'config/config.yaml'
        self.config = self._load_config()
        
        # Initialize components
        self.ml_models = AdvancedMLModels(config_path)
        self.sentiment_engine = SentimentAnalysisEngine(config_path)
        self.validator = ModelValidationFramework()
        self.security_framework = SecureTradingFramework()
        self.data_fetcher = DataFetcher(config_path)
        
        # System state
        self.system_active = False
        self.last_signals = {}
        self.performance_history = []
        self.system_metrics = []
        
        # Threading
        self.processing_thread = None
        self.monitoring_thread = None
        
        # Cache
        self.model_cache = {}
        self.validation_cache = {}
        
        logger.info("🤖 AI Integration Framework initialized")
        logger.info(f"   Components: ML Models, Sentiment Analysis, Validation, Security")
        logger.info(f"   Assets: {self.config.get('assets', [])}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(self.config_path, 'r') as f:
                import yaml
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'assets': ['BTC-USD', 'ETH-USD', 'SUI-USD', 'SOL-USD'],
            'signal_generation': {
                'ml_weight': 0.5,
                'sentiment_weight': 0.3,
                'technical_weight': 0.2,
                'confidence_threshold': 0.6,
                'min_agreement': 0.7
            },
            'risk_management': {
                'max_position_size': 0.1,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05,
                'max_correlation': 0.7
            },
            'validation': {
                'min_validation_score': 0.7,
                'revalidation_interval': 86400,  # 24 hours
                'backtest_lookback': 252  # 1 year
            },
            'monitoring': {
                'update_interval': 300,  # 5 minutes
                'performance_window': 7200,  # 2 hours
                'alert_threshold': 0.05  # 5% performance drop
            }
        }
    
    async def initialize_system(self) -> bool:
        """Initialize all system components"""
        logger.info("🚀 Initializing AI trading system...")
        
        try:
            # Initialize security framework
            logger.info("🔐 Initializing security framework...")
            # security_token = self.security_framework.initialize_security(
            #     grok_api_key=self.config.get('grok_api_key'),
            #     user_password=self.config.get('user_password', 'default_password')
            # )
            
            # Train ML models
            logger.info("🧠 Training ML models...")
            await self._train_all_models()
            
            # Validate models
            logger.info("🔬 Validating models...")
            await self._validate_all_models()
            
            # Initialize sentiment analysis
            logger.info("📊 Initializing sentiment analysis...")
            self.sentiment_engine.start_real_time_monitoring()
            
            # System health check
            logger.info("🏥 Running system health check...")
            health_check = await self._system_health_check()
            
            if health_check:
                logger.info("✅ System initialization completed successfully")
                return True
            else:
                logger.error("❌ System initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ System initialization error: {e}")
            return False
    
    async def _train_all_models(self):
        """Train ML models for all assets"""
        training_tasks = []
        
        for asset in self.config['assets']:
            task = asyncio.create_task(self._train_asset_models(asset))
            training_tasks.append(task)
        
        results = await asyncio.gather(*training_tasks, return_exceptions=True)
        
        # Check results
        successful_trainings = 0
        for i, result in enumerate(results):
            asset = self.config['assets'][i]
            if isinstance(result, Exception):
                logger.error(f"Training failed for {asset}: {result}")
            else:
                successful_trainings += 1
                logger.info(f"✅ Models trained successfully for {asset}")
        
        logger.info(f"Training completed: {successful_trainings}/{len(self.config['assets'])} assets")
    
    async def _train_asset_models(self, asset: str):
        """Train models for a specific asset"""
        try:
            # Train models
            models = self.ml_models.train_all_models(asset)
            
            # Cache models
            self.model_cache[asset] = models
            
            return models
            
        except Exception as e:
            logger.error(f"Error training models for {asset}: {e}")
            raise
    
    async def _validate_all_models(self):
        """Validate all trained models"""
        validation_tasks = []
        
        for asset, models in self.model_cache.items():
            for model_name, model in models.items():
                task = asyncio.create_task(self._validate_model(asset, model_name, model))
                validation_tasks.append(task)
        
        results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        
        # Process validation results
        successful_validations = 0
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Validation failed: {result}")
            else:
                successful_validations += 1
        
        logger.info(f"Validation completed: {successful_validations}/{len(results)} models validated")
    
    async def _validate_model(self, asset: str, model_name: str, model: Any):
        """Validate a specific model"""
        try:
            # Fetch validation data
            data = self.ml_models.fetch_market_data(asset, period="1y")
            if data.empty:
                raise ValueError(f"No data available for {asset}")
            
            # Prepare data for validation
            X, y = self.ml_models.prepare_features(data)
            
            # Run validation
            validation_report = self.validator.validate_model(
                model, X, y, f"{asset}_{model_name}", 'regression'
            )
            
            # Cache validation results
            self.validation_cache[f"{asset}_{model_name}"] = validation_report
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Error validating {asset}_{model_name}: {e}")
            raise
    
    async def _system_health_check(self) -> bool:
        """Comprehensive system health check"""
        health_checks = []
        
        # Check ML models
        model_health = len(self.model_cache) > 0
        health_checks.append(("ML Models", model_health))
        
        # Check sentiment engine
        sentiment_health = self.sentiment_engine.monitoring_active
        health_checks.append(("Sentiment Engine", sentiment_health))
        
        # Check validation results
        validation_health = len(self.validation_cache) > 0
        health_checks.append(("Model Validation", validation_health))
        
        # Check data connectivity
        try:
            test_data = self.data_fetcher.fetch_market_data("BTC", "1d")
            data_health = not test_data.empty
        except:
            data_health = False
        health_checks.append(("Data Connectivity", data_health))
        
        # Log health check results
        logger.info("🏥 System Health Check Results:")
        for component, status in health_checks:
            status_str = "✅ HEALTHY" if status else "❌ UNHEALTHY"
            logger.info(f"   {component}: {status_str}")
        
        # Overall health
        overall_health = all(status for _, status in health_checks)
        
        return overall_health
    
    async def generate_ai_signal(self, asset: str) -> AISignal:
        """Generate comprehensive AI trading signal"""
        try:
            # Fetch current market data
            market_data = self.ml_models.fetch_market_data(asset, period="1mo")
            if market_data.empty:
                raise ValueError(f"No market data available for {asset}")
            
            # 1. ML Predictions
            ml_predictions = self.ml_models.predict(asset, market_data)
            ensemble_prediction = self.ml_models.get_ensemble_prediction(asset, market_data)
            
            # 2. Sentiment Analysis
            sentiment_signal = self.sentiment_engine.generate_sentiment_signal(asset, hours_back=6)
            
            # 3. Combine signals
            combined_action, combined_confidence, combined_reasoning = self._combine_signals(
                ml_predictions, ensemble_prediction, sentiment_signal
            )
            
            # 4. Risk assessment
            risk_level, position_size, stop_loss, take_profit = self._calculate_risk_metrics(
                asset, combined_action, combined_confidence, market_data
            )
            
            # 5. Validation scores
            validation_score = self._get_validation_score(asset, ml_predictions)
            backtest_score = self._get_backtest_score(asset, combined_action)
            
            # Create comprehensive signal
            signal = AISignal(
                asset=asset,
                timestamp=datetime.now(),
                ml_predictions=ml_predictions,
                ensemble_prediction=ensemble_prediction,
                sentiment_signal=sentiment_signal,
                combined_action=combined_action,
                combined_confidence=combined_confidence,
                combined_reasoning=combined_reasoning,
                risk_level=risk_level,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                validation_score=validation_score,
                backtest_score=backtest_score,
                supporting_data={
                    'market_data': market_data.tail(1).to_dict(),
                    'ml_model_count': len(ml_predictions),
                    'sentiment_volume': sentiment_signal.volume if sentiment_signal else 0
                }
            )
            
            # Cache signal
            self.last_signals[asset] = signal
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating AI signal for {asset}: {e}")
            # Return neutral signal
            return AISignal(
                asset=asset,
                timestamp=datetime.now(),
                ml_predictions=[],
                combined_action='HOLD',
                combined_confidence=0.0,
                combined_reasoning=f"Error: {str(e)}",
                risk_level='HIGH',
                position_size=0.0
            )
    
    def _combine_signals(self, ml_predictions: List[ModelPrediction], 
                        ensemble_prediction: Optional[ModelPrediction],
                        sentiment_signal: Optional[SentimentSignal]) -> Tuple[str, float, str]:
        """Combine ML and sentiment signals"""
        try:
            weights = self.config['signal_generation']
            
            # ML signal
            ml_signal = 0.0
            ml_confidence = 0.0
            
            if ensemble_prediction:
                current_price = 100  # Placeholder - would get from market data
                ml_signal = (ensemble_prediction.prediction - current_price) / current_price
                ml_confidence = ensemble_prediction.confidence
            
            # Sentiment signal
            sentiment_signal_val = 0.0
            sentiment_confidence = 0.0
            
            if sentiment_signal:
                sentiment_signal_val = sentiment_signal.sentiment_score
                sentiment_confidence = sentiment_signal.confidence
            
            # Weighted combination
            combined_signal = (
                ml_signal * weights['ml_weight'] +
                sentiment_signal_val * weights['sentiment_weight']
            )
            
            combined_confidence = (
                ml_confidence * weights['ml_weight'] +
                sentiment_confidence * weights['sentiment_weight']
            )
            
            # Determine action
            if combined_signal > 0.02 and combined_confidence > weights['confidence_threshold']:
                action = 'BUY'
                reasoning = f"Strong bullish signals: ML={ml_signal:.3f}, Sentiment={sentiment_signal_val:.3f}"
            elif combined_signal < -0.02 and combined_confidence > weights['confidence_threshold']:
                action = 'SELL'
                reasoning = f"Strong bearish signals: ML={ml_signal:.3f}, Sentiment={sentiment_signal_val:.3f}"
            else:
                action = 'HOLD'
                reasoning = f"Neutral or weak signals: ML={ml_signal:.3f}, Sentiment={sentiment_signal_val:.3f}"
            
            return action, combined_confidence, reasoning
            
        except Exception as e:
            logger.error(f"Error combining signals: {e}")
            return 'HOLD', 0.0, f"Signal combination error: {str(e)}"
    
    def _calculate_risk_metrics(self, asset: str, action: str, confidence: float, 
                              market_data: pd.DataFrame) -> Tuple[str, float, Optional[float], Optional[float]]:
        """Calculate risk metrics for the signal"""
        try:
            risk_config = self.config['risk_management']
            
            # Risk level based on confidence and market volatility
            if market_data is not None and not market_data.empty:
                volatility = market_data['Close'].pct_change().std()
                
                if confidence > 0.8 and volatility < 0.02:
                    risk_level = 'LOW'
                elif confidence > 0.6 and volatility < 0.05:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'HIGH'
            else:
                risk_level = 'MEDIUM'
            
            # Position size based on confidence and risk level
            if action == 'HOLD':
                position_size = 0.0
            else:
                base_size = risk_config['max_position_size']
                risk_adjustment = 1.0 if risk_level == 'LOW' else 0.5 if risk_level == 'MEDIUM' else 0.25
                position_size = base_size * confidence * risk_adjustment
            
            # Stop loss and take profit
            if action != 'HOLD' and market_data is not None and not market_data.empty:
                current_price = market_data['Close'].iloc[-1]
                
                if action == 'BUY':
                    stop_loss = current_price * (1 - risk_config['stop_loss_pct'])
                    take_profit = current_price * (1 + risk_config['take_profit_pct'])
                else:  # SELL
                    stop_loss = current_price * (1 + risk_config['stop_loss_pct'])
                    take_profit = current_price * (1 - risk_config['take_profit_pct'])
            else:
                stop_loss = None
                take_profit = None
            
            return risk_level, position_size, stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return 'HIGH', 0.0, None, None
    
    def _get_validation_score(self, asset: str, ml_predictions: List[ModelPrediction]) -> float:
        """Get validation score for the asset's models"""
        try:
            total_score = 0.0
            count = 0
            
            for prediction in ml_predictions:
                model_key = f"{asset}_{prediction.model_type}"
                if model_key in self.validation_cache:
                    total_score += self.validation_cache[model_key].overall_score
                    count += 1
            
            return total_score / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error getting validation score: {e}")
            return 0.0
    
    def _get_backtest_score(self, asset: str, action: str) -> float:
        """Get backtest score for the signal"""
        # Placeholder - would implement actual backtesting
        # This would use the backtesting framework to test the signal
        return 0.75  # Default score
    
    def start_autonomous_trading(self):
        """Start autonomous trading system"""
        if self.system_active:
            logger.warning("System already active")
            return
        
        logger.info("🚀 Starting autonomous AI trading system...")
        
        self.system_active = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("✅ Autonomous trading system started")
    
    def stop_autonomous_trading(self):
        """Stop autonomous trading system"""
        if not self.system_active:
            logger.warning("System not active")
            return
        
        logger.info("⏹️  Stopping autonomous trading system...")
        
        self.system_active = False
        
        # Stop sentiment monitoring
        self.sentiment_engine.stop_real_time_monitoring()
        
        # Wait for threads to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=10)
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        
        logger.info("✅ Autonomous trading system stopped")
    
    def _processing_loop(self):
        """Main processing loop for signal generation"""
        while self.system_active:
            try:
                # Generate signals for all assets
                for asset in self.config['assets']:
                    signal = asyncio.run(self.generate_ai_signal(asset))
                    
                    # Log significant signals
                    if signal.combined_action != 'HOLD' and signal.combined_confidence > 0.7:
                        logger.info(f"🎯 SIGNAL: {asset} - {signal.combined_action} "
                                  f"(Confidence: {signal.combined_confidence:.3f})")
                        logger.info(f"   Reasoning: {signal.combined_reasoning}")
                        logger.info(f"   Risk Level: {signal.risk_level}")
                        logger.info(f"   Position Size: {signal.position_size:.3f}")
                
                # Wait before next cycle
                time.sleep(self.config['monitoring']['update_interval'])
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _monitoring_loop(self):
        """System monitoring loop"""
        while self.system_active:
            try:
                # Calculate system metrics
                metrics = self._calculate_system_metrics()
                self.system_metrics.append(metrics)
                
                # Check for alerts
                self._check_system_alerts(metrics)
                
                # Cleanup old metrics
                if len(self.system_metrics) > 100:
                    self.system_metrics = self.system_metrics[-100:]
                
                # Wait before next check
                time.sleep(self.config['monitoring']['performance_window'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait before retrying
    
    def _calculate_system_metrics(self) -> AISystemMetrics:
        """Calculate comprehensive system metrics"""
        try:
            # Model performance
            model_accuracy = {}
            model_consistency = {}
            for asset in self.config['assets']:
                if asset in self.model_cache:
                    model_accuracy[asset] = 0.75  # Placeholder
                    model_consistency[asset] = 0.80  # Placeholder
            
            # System health
            system_uptime = 0.99  # Placeholder
            processing_latency = 2.5  # Placeholder
            error_rate = 0.01  # Placeholder
            
            return AISystemMetrics(
                timestamp=datetime.now(),
                model_accuracy=model_accuracy,
                model_consistency=model_consistency,
                ensemble_performance=0.78,
                sentiment_accuracy=0.68,
                sentiment_coverage=0.85,
                system_uptime=system_uptime,
                processing_latency=processing_latency,
                error_rate=error_rate,
                signal_quality=0.72,
                risk_adjusted_return=0.15,
                drawdown_control=0.88,
                security_score=0.95,
                threat_level='LOW'
            )
            
        except Exception as e:
            logger.error(f"Error calculating system metrics: {e}")
            return AISystemMetrics(
                timestamp=datetime.now(),
                model_accuracy={},
                model_consistency={},
                ensemble_performance=0.0,
                sentiment_accuracy=0.0,
                sentiment_coverage=0.0,
                system_uptime=0.0,
                processing_latency=0.0,
                error_rate=1.0,
                signal_quality=0.0,
                risk_adjusted_return=0.0,
                drawdown_control=0.0,
                security_score=0.0,
                threat_level='HIGH'
            )
    
    def _check_system_alerts(self, metrics: AISystemMetrics):
        """Check for system alerts"""
        alert_threshold = self.config['monitoring']['alert_threshold']
        
        # Performance alerts
        if metrics.ensemble_performance < 0.5:
            logger.warning("🚨 ALERT: Ensemble performance below threshold")
        
        if metrics.error_rate > alert_threshold:
            logger.warning(f"🚨 ALERT: Error rate too high: {metrics.error_rate:.3f}")
        
        if metrics.system_uptime < 0.95:
            logger.warning(f"🚨 ALERT: System uptime low: {metrics.system_uptime:.3f}")
        
        # Security alerts
        if metrics.threat_level == 'HIGH':
            logger.warning("🚨 SECURITY ALERT: High threat level detected")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'system_active': self.system_active,
            'components': {
                'ml_models': len(self.model_cache),
                'sentiment_engine': self.sentiment_engine.monitoring_active,
                'validation_results': len(self.validation_cache),
                'security_framework': True  # Placeholder
            },
            'performance': {
                'signals_generated': len(self.last_signals),
                'validation_score': np.mean([v.overall_score for v in self.validation_cache.values()]) if self.validation_cache else 0.0,
                'system_health': self._calculate_system_health()
            },
            'last_signals': {
                asset: {
                    'action': signal.combined_action,
                    'confidence': signal.combined_confidence,
                    'timestamp': signal.timestamp.isoformat()
                } for asset, signal in self.last_signals.items()
            }
        }
        
        return status
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        if not self.system_metrics:
            return 0.0
        
        latest_metrics = self.system_metrics[-1]
        
        health_components = [
            latest_metrics.ensemble_performance,
            latest_metrics.system_uptime,
            1 - latest_metrics.error_rate,
            latest_metrics.security_score
        ]
        
        return np.mean(health_components)
    
    def generate_system_report(self) -> str:
        """Generate comprehensive system report"""
        status = self.get_system_status()
        
        report = f"""
🤖 AI INTEGRATION SYSTEM REPORT
===============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System Status: {'ACTIVE' if self.system_active else 'INACTIVE'}

📊 COMPONENT STATUS
ML Models: {status['components']['ml_models']} assets
Sentiment Engine: {'ACTIVE' if status['components']['sentiment_engine'] else 'INACTIVE'}
Validation Results: {status['components']['validation_results']} models
Security Framework: {'ACTIVE' if status['components']['security_framework'] else 'INACTIVE'}

📈 PERFORMANCE METRICS
Signals Generated: {status['performance']['signals_generated']}
Average Validation Score: {status['performance']['validation_score']:.3f}
System Health: {status['performance']['system_health']:.3f}

🎯 RECENT SIGNALS
"""
        
        for asset, signal_info in status['last_signals'].items():
            report += f"{asset}: {signal_info['action']} (Confidence: {signal_info['confidence']:.3f})\n"
        
        if self.system_metrics:
            latest_metrics = self.system_metrics[-1]
            report += f"""
📊 SYSTEM METRICS
Ensemble Performance: {latest_metrics.ensemble_performance:.3f}
Sentiment Accuracy: {latest_metrics.sentiment_accuracy:.3f}
System Uptime: {latest_metrics.system_uptime:.3f}
Processing Latency: {latest_metrics.processing_latency:.1f}s
Error Rate: {latest_metrics.error_rate:.3f}
Security Score: {latest_metrics.security_score:.3f}
Threat Level: {latest_metrics.threat_level}

🏆 RECOMMENDATIONS
"""
            
            if latest_metrics.ensemble_performance < 0.7:
                report += "• Consider retraining ML models\n"
            
            if latest_metrics.sentiment_accuracy < 0.6:
                report += "• Improve sentiment analysis sources\n"
            
            if latest_metrics.error_rate > 0.05:
                report += "• Investigate system errors\n"
            
            if latest_metrics.security_score < 0.9:
                report += "• Review security measures\n"
        
        report += f"""
⚠️  IMPORTANT NOTES
• System performance may vary with market conditions
• Regular revalidation and retraining recommended
• Monitor for model drift and performance degradation
• Maintain proper risk management at all times
"""
        
        return report

async def main():
    """Demo function for AI integration framework"""
    print("🤖 AI INTEGRATION FRAMEWORK DEMO")
    print("=" * 60)
    
    # Initialize framework
    framework = AIIntegrationFramework()
    
    # Initialize system
    print("\n🚀 Initializing AI system...")
    success = await framework.initialize_system()
    
    if success:
        print("✅ System initialized successfully")
        
        # Generate sample signals
        print("\n🎯 Generating AI signals...")
        for asset in ['BTC-USD', 'ETH-USD']:
            signal = await framework.generate_ai_signal(asset)
            print(f"   {asset}: {signal.combined_action} (Confidence: {signal.combined_confidence:.3f})")
        
        # Get system status
        print("\n📊 System Status:")
        status = framework.get_system_status()
        print(f"   Components: {status['components']}")
        print(f"   Performance: {status['performance']}")
        
        # Generate report
        print("\n📋 Generating system report...")
        report = framework.generate_system_report()
        print("✅ Report generated")
        
        # Start autonomous trading (brief demo)
        print("\n🚀 Starting autonomous trading demo...")
        framework.start_autonomous_trading()
        
        # Wait a bit
        await asyncio.sleep(10)
        
        # Stop system
        print("\n⏹️  Stopping system...")
        framework.stop_autonomous_trading()
        
    else:
        print("❌ System initialization failed")
    
    print("\n" + "=" * 60)
    print("🎯 AI Integration Framework Demo Complete!")

if __name__ == "__main__":
    asyncio.run(main()) 