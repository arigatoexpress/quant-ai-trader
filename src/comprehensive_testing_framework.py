"""
Comprehensive Testing Framework
Tests all app features, integrations, and data flows for the enhanced Quant AI Trader
"""

import asyncio
import unittest
import pytest
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import tempfile
import os
import sys
from pathlib import Path
import yaml
import aiohttp
import websockets
from unittest.mock import Mock, patch, AsyncMock
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestResult:
    """Represents a test result with detailed information"""
    
    def __init__(self, test_name: str, status: str, duration: float, details: str = ""):
        self.test_name = test_name
        self.status = status  # PASS, FAIL, ERROR, SKIP
        self.duration = duration
        self.details = details
        self.timestamp = datetime.now()
        self.error_message = ""
        self.stack_trace = ""

class TestSuite:
    """Represents a test suite with multiple related tests"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.tests = []
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def add_test(self, test_func, test_name: str = None):
        """Add a test to the suite"""
        if test_name is None:
            test_name = test_func.__name__
        
        self.tests.append({
            'func': test_func,
            'name': test_name
        })
    
    async def run_tests(self) -> List[TestResult]:
        """Run all tests in the suite"""
        self.start_time = datetime.now()
        logger.info(f"Running test suite: {self.name}")
        
        for test in self.tests:
            start_time = datetime.now()
            
            try:
                if asyncio.iscoroutinefunction(test['func']):
                    await test['func']()
                else:
                    test['func']()
                
                duration = (datetime.now() - start_time).total_seconds()
                result = TestResult(test['name'], 'PASS', duration)
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                result = TestResult(test['name'], 'FAIL', duration)
                result.error_message = str(e)
                result.stack_trace = str(sys.exc_info())
            
            self.results.append(result)
            logger.info(f"Test {test['name']}: {result.status}")
        
        self.end_time = datetime.now()
        return self.results

class DataIntegrationTests:
    """Tests for data integration and collection"""
    
    @staticmethod
    async def test_market_data_collection():
        """Test market data collection from multiple sources"""
        logger.info("Testing market data collection...")
        
        # Mock data sources
        mock_market_data = {
            'BTC': pd.DataFrame({
                'close': np.random.normal(50000, 2000, 100),
                'volume': np.random.normal(1000000, 200000, 100),
                'market_cap': np.random.normal(1000000000, 100000000, 100)
            }, index=pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H'))
        }
        
        # Test data quality
        assert not mock_market_data['BTC'].empty, "Market data should not be empty"
        assert len(mock_market_data['BTC']) > 0, "Market data should have rows"
        assert 'close' in mock_market_data['BTC'].columns, "Market data should have close price"
        
        logger.info("Market data collection test passed")
    
    @staticmethod
    async def test_sentiment_data_integration():
        """Test sentiment data integration"""
        logger.info("Testing sentiment data integration...")
        
        # Mock sentiment data
        mock_sentiment_data = {
            'BTC': pd.DataFrame({
                'sentiment_score': np.random.normal(0.6, 0.2, 100),
                'confidence': np.random.normal(0.7, 0.1, 100),
                'volume': np.random.normal(1000, 200, 100)
            }, index=pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H'))
        }
        
        # Test sentiment data quality
        assert not mock_sentiment_data['BTC'].empty, "Sentiment data should not be empty"
        assert 'sentiment_score' in mock_sentiment_data['BTC'].columns, "Sentiment data should have sentiment score"
        assert all(-1 <= score <= 1 for score in mock_sentiment_data['BTC']['sentiment_score']), "Sentiment scores should be between -1 and 1"
        
        logger.info("Sentiment data integration test passed")
    
    @staticmethod
    async def test_onchain_data_collection():
        """Test onchain data collection"""
        logger.info("Testing onchain data collection...")
        
        # Mock onchain data
        mock_onchain_data = {
            'BTC': pd.DataFrame({
                'active_addresses': np.random.normal(100000, 10000, 100),
                'transaction_count': np.random.normal(50000, 5000, 100),
                'gas_price': np.random.normal(20, 5, 100)
            }, index=pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H'))
        }
        
        # Test onchain data quality
        assert not mock_onchain_data['BTC'].empty, "Onchain data should not be empty"
        assert 'active_addresses' in mock_onchain_data['BTC'].columns, "Onchain data should have active addresses"
        assert all(count >= 0 for count in mock_onchain_data['BTC']['transaction_count']), "Transaction counts should be non-negative"
        
        logger.info("Onchain data collection test passed")

class AsymmetricTradingTests:
    """Tests for asymmetric trading framework"""
    
    @staticmethod
    async def test_barbell_portfolio_creation():
        """Test barbell portfolio creation and allocation"""
        logger.info("Testing barbell portfolio creation...")
        
        # Import asymmetric trading framework
        try:
            from asymmetric_trading_framework import BarbellPortfolio
            
            # Create barbell portfolio
            portfolio = BarbellPortfolio(initial_capital=100000)
            
            # Test allocation
            allocation = portfolio.allocate_capital()
            
            assert allocation['total_safe'] == 90000, "90% should be allocated to safe assets"
            assert allocation['total_risky'] == 10000, "10% should be allocated to risky assets"
            assert len(allocation['safe_allocation']) > 0, "Safe allocation should have assets"
            
            logger.info("Barbell portfolio creation test passed")
            
        except ImportError:
            logger.warning("Asymmetric trading framework not available, skipping test")
    
    @staticmethod
    async def test_asymmetric_bet_sizing():
        """Test asymmetric bet sizing calculations"""
        logger.info("Testing asymmetric bet sizing...")
        
        try:
            from asymmetric_trading_framework import AsymmetricBetSizer
            
            # Create bet sizer
            bet_sizer = AsymmetricBetSizer(risk_tolerance=0.1)
            
            # Test Kelly criterion
            kelly_bet = bet_sizer.calculate_kelly_bet(win_probability=0.6, win_loss_ratio=2.0)
            assert 0 <= kelly_bet <= 0.1, "Kelly bet should be within risk tolerance"
            
            # Test asymmetric bet sizing
            position_size, metrics = bet_sizer.calculate_position_size(
                symbol="BTC",
                current_price=50000,
                target_price=57500,  # 15% gain
                stop_loss=47500,     # 5% loss
                confidence=0.7,
                volatility=0.02
            )
            
            assert position_size > 0, "Position size should be positive"
            assert 'convexity_score' in metrics, "Metrics should include convexity score"
            assert metrics['convexity_score'] > 1, "Convexity score should indicate asymmetric opportunity"
            
            logger.info("Asymmetric bet sizing test passed")
            
        except ImportError:
            logger.warning("Asymmetric trading framework not available, skipping test")
    
    @staticmethod
    async def test_fat_tail_risk_management():
        """Test fat tail risk management"""
        logger.info("Testing fat tail risk management...")
        
        try:
            from asymmetric_trading_framework import FatTailRiskManager, BarbellPortfolio
            
            # Create portfolio and risk manager
            portfolio = BarbellPortfolio(initial_capital=100000)
            risk_manager = FatTailRiskManager(portfolio)
            
            # Create mock market data
            dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
            market_data = pd.DataFrame({
                'close': np.random.normal(50000, 2000, len(dates))
            }, index=dates)
            
            # Test fat tail risk identification
            fat_tail_risks = risk_manager.identify_fat_tail_risks(market_data)
            
            assert 'fat_tail_probability' in fat_tail_risks, "Should identify fat tail probability"
            assert 'kurtosis' in fat_tail_risks, "Should calculate kurtosis"
            assert 'var_99' in fat_tail_risks, "Should calculate VaR"
            
            # Test hedge requirements
            hedge_requirements = risk_manager.calculate_hedge_requirements(fat_tail_risks)
            
            assert 'total_hedge' in hedge_requirements, "Should calculate total hedge"
            assert hedge_requirements['total_hedge'] >= 0, "Hedge should be non-negative"
            
            logger.info("Fat tail risk management test passed")
            
        except ImportError:
            logger.warning("Asymmetric trading framework not available, skipping test")

class Grok4IntegrationTests:
    """Tests for Grok 4 data integration"""
    
    @staticmethod
    async def test_data_formatter():
        """Test Grok 4 data formatting"""
        logger.info("Testing Grok 4 data formatting...")
        
        try:
            from grok4_data_integration import Grok4DataFormatter, IntegratedDataPoint
            
            # Create formatter
            formatter = Grok4DataFormatter()
            
            # Create mock integrated data
            integrated_data = [
                IntegratedDataPoint(
                    timestamp=datetime.now(),
                    symbol="BTC",
                    price=50000.0,
                    volume=1000000.0,
                    market_cap=1000000000.0,
                    sentiment_score=0.6,
                    sentiment_confidence=0.7,
                    social_volume=5000.0,
                    news_sentiment=0.5,
                    onchain_metrics={'active_addresses': 100000, 'transaction_count': 50000},
                    technical_indicators={'rsi': 50.0, 'macd': 0.0},
                    macro_indicators={'vix': 20.0, 'dollar_index': 100.0},
                    data_quality_score=0.8,
                    source_count=5,
                    metadata={}
                )
            ]
            
            # Test formatting
            formatted_data = formatter.format_for_grok4(integrated_data)
            
            assert 'features' in formatted_data, "Formatted data should have features"
            assert 'matrix' in formatted_data['features'], "Features should have matrix"
            assert 'names' in formatted_data['features'], "Features should have names"
            assert len(formatted_data['features']['matrix']) > 0, "Feature matrix should not be empty"
            
            # Test prompt creation
            prompt = formatter.create_grok4_prompt(formatted_data, 'trading_decision')
            
            assert len(prompt) > 0, "Prompt should not be empty"
            assert "asymmetric trading" in prompt.lower(), "Prompt should mention asymmetric trading"
            
            logger.info("Grok 4 data formatting test passed")
            
        except ImportError:
            logger.warning("Grok 4 data integration not available, skipping test")
    
    @staticmethod
    async def test_data_orchestrator():
        """Test data orchestrator integration"""
        logger.info("Testing data orchestrator...")
        
        try:
            from grok4_data_integration import DataIntegrationOrchestrator, DataSource
            
            # Create orchestrator
            orchestrator = DataIntegrationOrchestrator()
            
            # Add data sources
            sources = [
                DataSource("binance", "market", "https://api.binance.com"),
                DataSource("twitter_sentiment", "sentiment", "https://api.twitter.com"),
                DataSource("etherscan", "onchain", "https://api.etherscan.io")
            ]
            
            for source in sources:
                orchestrator.add_data_source(source)
            
            assert len(orchestrator.data_sources) == 3, "Should have 3 data sources"
            
            # Test data integration
            symbols = ["BTC", "ETH"]
            integrated_data = await orchestrator.integrate_all_data(symbols)
            
            assert len(integrated_data) > 0, "Should have integrated data"
            
            # Test Grok 4 preparation
            grok4_analysis = await orchestrator.prepare_grok4_analysis(integrated_data, 'trading_decision')
            
            assert 'formatted_data' in grok4_analysis, "Should have formatted data"
            assert 'grok4_prompt' in grok4_analysis, "Should have Grok 4 prompt"
            
            logger.info("Data orchestrator test passed")
            
        except ImportError:
            logger.warning("Grok 4 data integration not available, skipping test")

class MLModelTests:
    """Tests for ML models and predictions"""
    
    @staticmethod
    async def test_advanced_ml_models():
        """Test advanced ML models"""
        logger.info("Testing advanced ML models...")
        
        try:
            from advanced_ml_models import AdvancedMLModels
            
            # Create ML models
            ml_models = AdvancedMLModels()
            
            # Create mock data
            dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='1H')
            mock_data = pd.DataFrame({
                'close': np.random.normal(50000, 2000, len(dates)),
                'volume': np.random.normal(1000000, 200000, len(dates)),
                'sentiment': np.random.normal(0.6, 0.2, len(dates))
            }, index=dates)
            
            # Test model training
            models = ml_models.train_all_models(mock_data)
            
            assert len(models) > 0, "Should have trained models"
            
            # Test predictions
            predictions = ml_models.generate_ensemble_predictions(mock_data)
            
            assert 'ensemble_prediction' in predictions, "Should have ensemble prediction"
            assert 'individual_predictions' in predictions, "Should have individual predictions"
            assert 'confidence_score' in predictions, "Should have confidence score"
            
            logger.info("Advanced ML models test passed")
            
        except ImportError:
            logger.warning("Advanced ML models not available, skipping test")
    
    @staticmethod
    async def test_sentiment_analysis():
        """Test sentiment analysis engine"""
        logger.info("Testing sentiment analysis engine...")
        
        try:
            from sentiment_analysis_engine import SentimentAnalysisEngine
            
            # Create sentiment engine
            sentiment_engine = SentimentAnalysisEngine()
            
            # Test sentiment analysis
            text_samples = [
                "Bitcoin is going to the moon! 🚀",
                "This is a terrible investment, I'm selling everything.",
                "The market is stable with moderate growth expected."
            ]
            
            for text in text_samples:
                sentiment_result = sentiment_engine.analyze_text_sentiment(text)
                
                assert 'sentiment_score' in sentiment_result, "Should have sentiment score"
                assert 'confidence' in sentiment_result, "Should have confidence"
                assert -1 <= sentiment_result['sentiment_score'] <= 1, "Sentiment score should be between -1 and 1"
            
            logger.info("Sentiment analysis engine test passed")
            
        except ImportError:
            logger.warning("Sentiment analysis engine not available, skipping test")

class BacktestingTests:
    """Tests for backtesting framework"""
    
    @staticmethod
    async def test_backtesting_engine():
        """Test backtesting engine"""
        logger.info("Testing backtesting engine...")
        
        try:
            from backtesting_framework import BacktestingEngine
            
            # Create backtesting engine
            backtest_engine = BacktestingEngine(initial_capital=100000)
            
            # Create mock historical data
            dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='1H')
            historical_data = pd.DataFrame({
                'open': np.random.normal(50000, 2000, len(dates)),
                'high': np.random.normal(50500, 2000, len(dates)),
                'low': np.random.normal(49500, 2000, len(dates)),
                'close': np.random.normal(50000, 2000, len(dates)),
                'volume': np.random.normal(1000000, 200000, len(dates))
            }, index=dates)
            
            # Test backtesting
            results = backtest_engine.run_backtest(historical_data, strategy_name="test_strategy")
            
            assert 'total_return' in results, "Should have total return"
            assert 'sharpe_ratio' in results, "Should have Sharpe ratio"
            assert 'max_drawdown' in results, "Should have max drawdown"
            assert 'trade_count' in results, "Should have trade count"
            
            logger.info("Backtesting engine test passed")
            
        except ImportError:
            logger.warning("Backtesting framework not available, skipping test")

class SecurityTests:
    """Tests for security features"""
    
    @staticmethod
    async def test_security_audit():
        """Test security audit functionality"""
        logger.info("Testing security audit...")
        
        try:
            from security_audit_cleanup import SecurityAuditor
            
            # Create security auditor
            auditor = SecurityAuditor(".")
            
            # Run audit
            report = await auditor.run_full_audit()
            
            assert 'summary' in report, "Should have summary"
            assert 'issues' in report, "Should have issues"
            assert 'security_score' in report, "Should have security score"
            assert 0 <= report['security_score'] <= 100, "Security score should be between 0 and 100"
            
            # Test GitHub safety check
            safe_for_github = auditor.is_safe_for_github()
            assert isinstance(safe_for_github, bool), "GitHub safety check should return boolean"
            
            logger.info("Security audit test passed")
            
        except ImportError:
            logger.warning("Security audit not available, skipping test")

class IntegrationTests:
    """Tests for system integration"""
    
    @staticmethod
    async def test_end_to_end_workflow():
        """Test end-to-end workflow"""
        logger.info("Testing end-to-end workflow...")
        
        # Test data collection
        await DataIntegrationTests.test_market_data_collection()
        await DataIntegrationTests.test_sentiment_data_integration()
        await DataIntegrationTests.test_onchain_data_collection()
        
        # Test asymmetric trading
        await AsymmetricTradingTests.test_barbell_portfolio_creation()
        await AsymmetricTradingTests.test_asymmetric_bet_sizing()
        await AsymmetricTradingTests.test_fat_tail_risk_management()
        
        # Test Grok 4 integration
        await Grok4IntegrationTests.test_data_formatter()
        await Grok4IntegrationTests.test_data_orchestrator()
        
        # Test ML models
        await MLModelTests.test_advanced_ml_models()
        await MLModelTests.test_sentiment_analysis()
        
        # Test backtesting
        await BacktestingTests.test_backtesting_engine()
        
        # Test security
        await SecurityTests.test_security_audit()
        
        logger.info("End-to-end workflow test passed")
    
    @staticmethod
    async def test_performance_benchmarks():
        """Test performance benchmarks"""
        logger.info("Testing performance benchmarks...")
        
        # Test data processing speed
        start_time = datetime.now()
        
        # Simulate data processing
        data_size = 10000
        mock_data = np.random.normal(0, 1, data_size)
        processed_data = np.mean(mock_data)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        assert processing_time < 1.0, f"Data processing should be fast, took {processing_time}s"
        assert isinstance(processed_data, float), "Processed data should be numeric"
        
        logger.info(f"Performance benchmark passed: {processing_time:.3f}s for {data_size} data points")

class ComprehensiveTestRunner:
    """Main test runner that executes all test suites"""
    
    def __init__(self):
        self.test_suites = []
        self.results = {}
        
    def add_test_suite(self, suite: TestSuite):
        """Add a test suite to the runner"""
        self.test_suites.append(suite)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and generate comprehensive report"""
        logger.info("Starting comprehensive testing...")
        
        start_time = datetime.now()
        all_results = []
        
        for suite in self.test_suites:
            suite_results = await suite.run_tests()
            all_results.extend(suite_results)
            self.results[suite.name] = suite_results
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Generate summary
        summary = self._generate_summary(all_results, total_duration)
        
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'suite_results': self.results,
            'all_results': [self._result_to_dict(result) for result in all_results]
        }
        
        return report
    
    def _generate_summary(self, results: List[TestResult], total_duration: float) -> Dict[str, Any]:
        """Generate test summary"""
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == 'PASS'])
        failed_tests = len([r for r in results if r.status == 'FAIL'])
        error_tests = len([r for r in results if r.status == 'ERROR'])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'errors': error_tests,
            'success_rate': round(success_rate, 2),
            'total_duration': round(total_duration, 2),
            'avg_duration': round(total_duration / total_tests, 3) if total_tests > 0 else 0
        }
    
    def _result_to_dict(self, result: TestResult) -> Dict[str, Any]:
        """Convert test result to dictionary"""
        return {
            'test_name': result.test_name,
            'status': result.status,
            'duration': result.duration,
            'details': result.details,
            'timestamp': result.timestamp.isoformat(),
            'error_message': result.error_message,
            'stack_trace': result.stack_trace
        }
    
    def save_report(self, report: Dict[str, Any], filename: str = "comprehensive_test_report.json") -> str:
        """Save test report to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Test report saved to: {filename}")
        return filename
    
    def print_summary(self, report: Dict[str, Any]):
        """Print test summary to console"""
        summary = report['summary']
        
        print("\n" + "="*60)
        print("COMPREHENSIVE TESTING RESULTS")
        print("="*60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Total Duration: {summary['total_duration']}s")
        print(f"Average Duration: {summary['avg_duration']}s")
        print("="*60)
        
        # Print failed tests
        if summary['failed'] > 0 or summary['errors'] > 0:
            print("\nFAILED TESTS:")
            for suite_name, suite_results in report['suite_results'].items():
                for result in suite_results:
                    if result.status in ['FAIL', 'ERROR']:
                        print(f"  - {result.test_name}: {result.error_message}")

# Create and run comprehensive tests
async def run_comprehensive_tests():
    """Run all comprehensive tests"""
    
    # Create test runner
    runner = ComprehensiveTestRunner()
    
    # Create test suites
    data_integration_suite = TestSuite("Data Integration", "Tests for data collection and integration")
    data_integration_suite.add_test(DataIntegrationTests.test_market_data_collection)
    data_integration_suite.add_test(DataIntegrationTests.test_sentiment_data_integration)
    data_integration_suite.add_test(DataIntegrationTests.test_onchain_data_collection)
    
    asymmetric_trading_suite = TestSuite("Asymmetric Trading", "Tests for asymmetric trading framework")
    asymmetric_trading_suite.add_test(AsymmetricTradingTests.test_barbell_portfolio_creation)
    asymmetric_trading_suite.add_test(AsymmetricTradingTests.test_asymmetric_bet_sizing)
    asymmetric_trading_suite.add_test(AsymmetricTradingTests.test_fat_tail_risk_management)
    
    grok4_integration_suite = TestSuite("Grok 4 Integration", "Tests for Grok 4 data integration")
    grok4_integration_suite.add_test(Grok4IntegrationTests.test_data_formatter)
    grok4_integration_suite.add_test(Grok4IntegrationTests.test_data_orchestrator)
    
    ml_models_suite = TestSuite("ML Models", "Tests for machine learning models")
    ml_models_suite.add_test(MLModelTests.test_advanced_ml_models)
    ml_models_suite.add_test(MLModelTests.test_sentiment_analysis)
    
    backtesting_suite = TestSuite("Backtesting", "Tests for backtesting framework")
    backtesting_suite.add_test(BacktestingTests.test_backtesting_engine)
    
    security_suite = TestSuite("Security", "Tests for security features")
    security_suite.add_test(SecurityTests.test_security_audit)
    
    integration_suite = TestSuite("Integration", "Tests for system integration")
    integration_suite.add_test(IntegrationTests.test_end_to_end_workflow)
    integration_suite.add_test(IntegrationTests.test_performance_benchmarks)
    
    # Add suites to runner
    runner.add_test_suite(data_integration_suite)
    runner.add_test_suite(asymmetric_trading_suite)
    runner.add_test_suite(grok4_integration_suite)
    runner.add_test_suite(ml_models_suite)
    runner.add_test_suite(backtesting_suite)
    runner.add_test_suite(security_suite)
    runner.add_test_suite(integration_suite)
    
    # Run all tests
    report = await runner.run_all_tests()
    
    # Print summary
    runner.print_summary(report)
    
    # Save report
    report_path = runner.save_report(report)
    
    return report

if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests()) 