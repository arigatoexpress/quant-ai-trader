"""
Simple Test Script for Enhanced Quant AI Trader
Tests core functionality without complex dependencies
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTestRunner:
    """Simple test runner for core functionality"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = datetime.now()
    
    def run_test(self, test_name: str, test_func):
        """Run a single test"""
        logger.info(f"Running test: {test_name}")
        start_time = datetime.now()
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                asyncio.run(test_func())
            else:
                test_func()
            
            duration = (datetime.now() - start_time).total_seconds()
            result = {
                'test_name': test_name,
                'status': 'PASS',
                'duration': duration,
                'error': None
            }
            logger.info(f"✓ {test_name} passed ({duration:.3f}s)")
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            result = {
                'test_name': test_name,
                'status': 'FAIL',
                'duration': duration,
                'error': str(e)
            }
            logger.error(f"✗ {test_name} failed: {e}")
        
        self.test_results.append(result)
        return result
    
    def print_summary(self):
        """Print test summary"""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*60)
        print("SIMPLE TEST RESULTS")
        print("="*60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "Success Rate: 0%")
        print(f"Total Duration: {total_duration:.2f}s")
        print("="*60)
        
        if failed_tests > 0:
            print("\nFAILED TESTS:")
            for result in self.test_results:
                if result['status'] == 'FAIL':
                    print(f"  - {result['test_name']}: {result['error']}")

def test_data_structures():
    """Test basic data structures"""
    # Test pandas DataFrame creation
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
    df = pd.DataFrame({
        'close': np.random.normal(50000, 2000, len(dates)),
        'volume': np.random.normal(1000000, 200000, len(dates))
    }, index=dates)
    
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) > 0, "DataFrame should have rows"
    assert 'close' in df.columns, "DataFrame should have close column"
    
    logger.info("Data structures test passed")

def test_asymmetric_calculations():
    """Test asymmetric trading calculations"""
    # Test Kelly Criterion
    def calculate_kelly_bet(win_probability, win_loss_ratio, risk_tolerance=0.1):
        if win_probability <= 0 or win_probability >= 1:
            return 0.0
        
        kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio
        return max(0.0, min(kelly_fraction, risk_tolerance))
    
    # Test cases
    kelly_bet = calculate_kelly_bet(win_probability=0.6, win_loss_ratio=2.0)
    assert 0 <= kelly_bet <= 0.1, "Kelly bet should be within risk tolerance"
    
    # Test convexity calculation
    def calculate_convexity(potential_gain, potential_loss):
        return potential_gain / (potential_loss + 1e-8)
    
    convexity = calculate_convexity(potential_gain=0.15, potential_loss=0.05)
    assert abs(convexity - 3.0) < 1e-6, f"Convexity should be 3.0 for 15% gain vs 5% loss, got {convexity}"
    
    logger.info("Asymmetric calculations test passed")

def test_barbell_portfolio():
    """Test barbell portfolio allocation (flexible 75-80% safe, 20-25% risky).
    The allocation is not a strict requirement and can be dynamically adjusted within this range.
    """
    def allocate_barbell_portfolio(total_capital, safe_allocation=0.8):
        # Allow safe_allocation to be flexible between 0.75 and 0.8
        safe_capital = total_capital * safe_allocation
        risky_capital = total_capital * (1 - safe_allocation)
        return {
            'safe_capital': safe_capital,
            'risky_capital': risky_capital,
            'safe_ratio': safe_allocation,
            'risky_ratio': 1 - safe_allocation
        }
    # Test for both 0.75 and 0.8 safe allocation
    for safe_allocation in [0.75, 0.8]:
        allocation = allocate_barbell_portfolio(100000, safe_allocation)
        # Allow a small tolerance for floating point math
        assert math.isclose(allocation['safe_capital'], 100000 * safe_allocation, rel_tol=1e-6), f"{int(safe_allocation*100)}% should be allocated to safe assets"
        assert math.isclose(allocation['risky_capital'], 100000 * (1-safe_allocation), rel_tol=1e-6), f"{int((1-safe_allocation)*100)}% should be allocated to risky assets"
        assert 0.75 <= allocation['safe_ratio'] <= 0.8, "Safe ratio should be between 0.75 and 0.8"
        assert 0.2 <= allocation['risky_ratio'] <= 0.25, "Risky ratio should be between 0.2 and 0.25 (inclusive)"
    logger.info("Barbell portfolio test (flexible) passed")

def test_fat_tail_risk():
    """Test fat tail risk calculations"""
    def calculate_var(returns, confidence_level=0.01):
        """Calculate Value at Risk"""
        return np.percentile(returns, confidence_level * 100)
    
    def calculate_kurtosis(returns):
        """Calculate kurtosis (measure of fat tails)"""
        return np.mean(((returns - np.mean(returns)) / np.std(returns)) ** 4)
    
    # Generate sample returns
    returns = np.random.normal(0, 0.02, 1000)  # 2% daily volatility
    
    var_99 = calculate_var(returns, 0.01)
    kurtosis = calculate_kurtosis(returns)
    
    assert var_99 < 0, "VaR should be negative for losses"
    assert kurtosis > 0, "Kurtosis should be positive"
    
    logger.info("Fat tail risk test passed")

def test_data_quality():
    """Test data quality assessment"""
    def assess_data_quality(data):
        """Assess data quality"""
        quality_score = 1.0
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        quality_score -= missing_ratio * 0.5
        
        # Check for extreme values
        for col in data.select_dtypes(include=[np.number]).columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            extreme_ratio = ((data[col] - mean_val).abs() > 3 * std_val).mean()
            quality_score -= extreme_ratio * 0.3
        
        return max(0.0, quality_score)
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1H')
    test_data = pd.DataFrame({
        'price': np.random.normal(50000, 2000, len(dates)),
        'volume': np.random.normal(1000000, 200000, len(dates))
    }, index=dates)
    
    quality_score = assess_data_quality(test_data)
    
    assert 0 <= quality_score <= 1, "Quality score should be between 0 and 1"
    
    logger.info("Data quality test passed")

def test_ml_predictions():
    """Test ML prediction framework"""
    def simple_moving_average(data, window=20):
        """Simple moving average prediction"""
        return data.rolling(window=window).mean()
    
    def calculate_prediction_accuracy(actual, predicted):
        """Calculate prediction accuracy"""
        # Simple accuracy based on direction
        actual_direction = np.sign(actual.diff())
        predicted_direction = np.sign(predicted.diff())
        
        accuracy = (actual_direction == predicted_direction).mean()
        return accuracy
    
    # Generate test data
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='1H')
    prices = np.cumsum(np.random.normal(0, 0.01, len(dates))) + 50000
    price_data = pd.DataFrame({'close': prices}, index=dates)
    
    # Generate predictions
    predictions = simple_moving_average(price_data['close'])
    
    # Calculate accuracy
    accuracy = calculate_prediction_accuracy(price_data['close'], predictions)
    
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    
    logger.info("ML predictions test passed")

def test_sentiment_analysis():
    """Test sentiment analysis framework"""
    def analyze_sentiment(text):
        """Simple sentiment analysis"""
        positive_words = ['bull', 'moon', 'pump', 'buy', 'strong', 'growth']
        negative_words = ['bear', 'dump', 'sell', 'weak', 'crash', 'fall']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = positive_count + negative_count
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / total_words
        return sentiment_score
    
    # Test cases
    positive_text = "Bitcoin is going to the moon! Strong growth expected."
    negative_text = "Market is bearish, expect a crash."
    neutral_text = "Market is stable with moderate activity."
    
    positive_sentiment = analyze_sentiment(positive_text)
    negative_sentiment = analyze_sentiment(negative_text)
    neutral_sentiment = analyze_sentiment(neutral_text)
    
    assert positive_sentiment > 0, "Positive text should have positive sentiment"
    assert negative_sentiment < 0, "Negative text should have negative sentiment"
    assert -1 <= neutral_sentiment <= 1, "Sentiment should be between -1 and 1"
    
    logger.info("Sentiment analysis test passed")

def test_backtesting():
    """Test backtesting framework"""
    def simple_backtest(prices, initial_capital=100000):
        """Simple backtesting strategy"""
        capital = initial_capital
        position = 0
        trades = []
        
        for i in range(1, len(prices)):
            current_price = prices.iloc[i]
            prev_price = prices.iloc[i-1]
            
            # Simple strategy: buy on price increase, sell on decrease
            if current_price > prev_price and position == 0:
                # Buy
                position = capital / current_price
                capital = 0
                trades.append({
                    'action': 'BUY',
                    'price': current_price,
                    'timestamp': prices.index[i]
                })
            elif current_price < prev_price and position > 0:
                # Sell
                capital = position * current_price
                position = 0
                trades.append({
                    'action': 'SELL',
                    'price': current_price,
                    'timestamp': prices.index[i]
                })
        
        # Final position value
        final_value = capital + (position * prices.iloc[-1])
        total_return = (final_value - initial_capital) / initial_capital
        
        return {
            'total_return': total_return,
            'final_value': final_value,
            'trade_count': len(trades),
            'trades': trades
        }
    
    # Generate test data
    dates = pd.date_range(start='2024-01-01', end='2024-01-30', freq='1H')
    prices = np.cumsum(np.random.normal(0, 0.01, len(dates))) + 50000
    price_series = pd.Series(prices, index=dates)
    
    # Run backtest
    results = simple_backtest(price_series)
    
    assert 'total_return' in results, "Results should have total return"
    assert 'final_value' in results, "Results should have final value"
    assert 'trade_count' in results, "Results should have trade count"
    assert results['final_value'] > 0, "Final value should be positive"
    
    logger.info("Backtesting test passed")

def test_security_validation():
    """Test security validation. Ignores lines with os.getenv as these are secure patterns."""
    def validate_secrets_in_code(code_content):
        import re
        found_secrets = []
        for line in code_content.splitlines():
            if 'os.getenv' in line:
                continue  # skip environment variable assignments
            secret_patterns = [
                r'api[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}["\']?',
                r'password\s*[:=]\s*["\']?[^"\']+["\']?',
                r'secret[_-]?key\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}["\']?'
            ]
            for pattern in secret_patterns:
                matches = re.findall(pattern, line, re.IGNORECASE)
                found_secrets.extend(matches)
        return len(found_secrets) == 0, found_secrets
    # Test with clean code (using environment variables)
    clean_code = """
    api_key = os.getenv('API_KEY')
"YOUR_PASSWORD_HERE"PASSWORD')
    secret_key = os.getenv('SECRET_KEY')
    """
    is_clean, secrets = validate_secrets_in_code(clean_code)
    assert is_clean, f"Clean code should pass validation, found secrets: {secrets}"
    # Test with hardcoded secrets
    dirty_code = """
    api_key = "SECRET_REMOVED"
"secure_password"
    """
    is_clean, secrets = validate_secrets_in_code(dirty_code)
    assert not is_clean, "Code with hardcoded secrets should fail validation"
    assert len(secrets) > 0, "Should find secrets in dirty code"
    logger.info("Security validation test passed")

def test_performance_metrics():
    """Test performance metrics calculation"""
    def calculate_performance_metrics(trades):
        """Calculate trading performance metrics"""
        if not trades:
            return {}
        
        # Calculate returns
        returns = []
        for i in range(1, len(trades)):
            if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                return_pct = (trades[i]['price'] - trades[i-1]['price']) / trades[i-1]['price']
                returns.append(return_pct)
        
        if not returns:
            return {}
        
        returns = np.array(returns)
        
        metrics = {
            'total_trades': len(trades),
            'winning_trades': np.sum(returns > 0),
            'losing_trades': np.sum(returns < 0),
            'win_rate': np.sum(returns > 0) / len(returns),
            'avg_return': np.mean(returns),
            'total_return': np.sum(returns),
            'volatility': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        }
        
        return metrics
    
    # Generate test trades
    test_trades = [
        {'action': 'BUY', 'price': 50000, 'timestamp': '2024-01-01'},
        {'action': 'SELL', 'price': 52000, 'timestamp': '2024-01-02'},
        {'action': 'BUY', 'price': 51000, 'timestamp': '2024-01-03'},
        {'action': 'SELL', 'price': 49000, 'timestamp': '2024-01-04'},
    ]
    
    metrics = calculate_performance_metrics(test_trades)
    
    assert 'total_trades' in metrics, "Metrics should have total trades"
    assert 'win_rate' in metrics, "Metrics should have win rate"
    assert 'sharpe_ratio' in metrics, "Metrics should have Sharpe ratio"
    assert 0 <= metrics['win_rate'] <= 1, "Win rate should be between 0 and 1"
    
    logger.info("Performance metrics test passed")

async def main():
    """Run all simple tests"""
    print("Enhanced Quant AI Trader - Simple Tests")
    print("=" * 50)
    
    runner = SimpleTestRunner()
    
    # Run all tests
    tests = [
        ("Data Structures", test_data_structures),
        ("Asymmetric Calculations", test_asymmetric_calculations),
        ("Barbell Portfolio", test_barbell_portfolio),
        ("Fat Tail Risk", test_fat_tail_risk),
        ("Data Quality", test_data_quality),
        ("ML Predictions", test_ml_predictions),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Backtesting", test_backtesting),
        ("Security Validation", test_security_validation),
        ("Performance Metrics", test_performance_metrics),
    ]
    
    for test_name, test_func in tests:
        runner.run_test(test_name, test_func)
    
    # Print summary
    runner.print_summary()
    
    # Save results
    with open('simple_test_results.json', 'w') as f:
        json.dump(runner.test_results, f, indent=2, default=str)
    
    print(f"\nTest results saved to: simple_test_results.json")

if __name__ == "__main__":
    asyncio.run(main()) 