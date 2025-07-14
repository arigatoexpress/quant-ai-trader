"""
AI-Driven Risk Management Module
- Value at Risk (VaR) estimation (historical, parametric, Monte Carlo)
- Stress testing (historical and synthetic scenarios)
- Dynamic position sizing based on real-time risk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class VaRCalculator:
    """Calculates Value at Risk using various methods"""
    def __init__(self, confidence_level: float = 0.99):
        self.confidence_level = confidence_level

    def historical_var(self, returns: pd.Series) -> float:
        return float(-np.percentile(returns.dropna(), (1 - self.confidence_level) * 100))

    def parametric_var(self, returns: pd.Series) -> float:
        mu = returns.mean()
        sigma = returns.std()
        from scipy.stats import norm
        return float(-norm.ppf(1 - self.confidence_level, mu, sigma))

    def monte_carlo_var(self, returns: pd.Series, simulations: int = 10000) -> float:
        mu = returns.mean()
        sigma = returns.std()
        simulated = np.random.normal(mu, sigma, simulations)
        return float(-np.percentile(simulated, (1 - self.confidence_level) * 100))

class StressTester:
    """Performs stress testing using historical and synthetic scenarios"""
    def __init__(self, scenarios: Optional[List[Dict[str, Any]]] = None):
        self.scenarios = scenarios if scenarios is not None else []

    def add_scenario(self, name: str, shock_func):
        self.scenarios.append({'name': name, 'shock_func': shock_func})

    def run(self, prices: pd.Series) -> Dict[str, float]:
        results = {}
        for scenario in self.scenarios:
            shocked = scenario['shock_func'](prices)
            drawdown = (shocked.min() - shocked.iloc[0]) / shocked.iloc[0]
            results[scenario['name']] = drawdown
        return results

class DynamicPositionSizer:
    """Calculates position size based on VaR, volatility, and confidence"""

    def calculate_size(self, capital: float, entry: float, stop: float, var: float, confidence: float) -> float:
        risk_tolerance = 0.02  # 2% of capital at risk
        max_loss = capital * risk_tolerance
        shares = max_loss / abs(entry - stop) if abs(entry - stop) > 0 else 0
        return max(0, min(shares, capital / entry * 0.1))  # Max 10% of capital

class RiskManagementAI:
    """AI-driven risk management orchestrator that integrates VaR, stress testing, and position sizing"""
    
    def __init__(self, config: Dict[str, Any] = None, ml_models = None):
        self.config = config or {}
        self.ml_models = ml_models
        
        # Initialize components
        confidence_level = self.config.get('confidence_level', 0.99)
        self.var_calculator = VaRCalculator(confidence_level)
        self.stress_tester = StressTester()
        self.position_sizer = DynamicPositionSizer()
        
        # Add default stress scenarios
        self._setup_default_scenarios()
    
    def _setup_default_scenarios(self):
        """Setup default stress testing scenarios"""
        # Market crash scenario (20% drop)
        def market_crash(prices):
            return prices * 0.8
        
        # High volatility scenario (double volatility)
        def high_volatility(prices):
            returns = prices.pct_change().dropna()
            vol_factor = 2.0
            new_returns = returns * vol_factor
            return prices.iloc[0] * (1 + new_returns).cumprod()
        
        self.stress_tester.add_scenario("Market Crash", market_crash)
        self.stress_tester.add_scenario("High Volatility", high_volatility)
    
    def calculate_portfolio_risk(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for portfolio"""
        risk_metrics = {}
        
        # VaR calculations
        risk_metrics['historical_var'] = self.var_calculator.historical_var(returns)
        risk_metrics['parametric_var'] = self.var_calculator.parametric_var(returns)
        risk_metrics['monte_carlo_var'] = self.var_calculator.monte_carlo_var(returns)
        
        # Additional risk metrics
        risk_metrics['volatility'] = float(returns.std())
        risk_metrics['max_drawdown'] = float((returns.cumsum() - returns.cumsum().cummax()).min())
        risk_metrics['sharpe_ratio'] = float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0
        
        return risk_metrics
    
    def run_stress_tests(self, prices: pd.Series) -> Dict[str, float]:
        """Run stress tests on price series"""
        return self.stress_tester.run(prices)
    
    def calculate_position_size(self, capital: float, entry: float, stop: float, returns: pd.Series) -> Dict[str, Any]:
        """Calculate optimal position size based on risk metrics"""
        # Calculate VaR
        var = self.var_calculator.historical_var(returns)
        
        # Calculate confidence based on recent performance
        confidence = min(0.9, max(0.1, 1.0 - abs(returns.tail(10).mean())))
        
        # Calculate position size
        position_size = self.position_sizer.calculate_size(capital, entry, stop, var, confidence)
        
        return {
            'position_size': position_size,
            'var': var,
            'confidence': confidence,
            'risk_pct': (position_size * abs(entry - stop)) / capital if capital > 0 else 0
        }
    
    def generate_risk_report(self, returns: pd.Series, prices: pd.Series, capital: float) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        risk_metrics = self.calculate_portfolio_risk(returns)
        stress_results = self.run_stress_tests(prices)
        
        # Mock entry and stop for demonstration
        entry = float(prices.iloc[-1])
        stop = entry * 0.95  # 5% stop loss
        
        position_info = self.calculate_position_size(capital, entry, stop, returns)
        
        return {
            'timestamp': datetime.now(),
            'risk_metrics': risk_metrics,
            'stress_test_results': stress_results,
            'position_sizing': position_info,
            'recommendations': self._generate_recommendations(risk_metrics, stress_results)
        }
    
    def _generate_recommendations(self, risk_metrics: Dict[str, float], stress_results: Dict[str, float]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        # VaR-based recommendations
        if risk_metrics['historical_var'] > 0.05:  # 5% VaR threshold
            recommendations.append("High VaR detected - consider reducing position sizes")
        
        # Volatility-based recommendations
        if risk_metrics['volatility'] > 0.03:  # 3% daily volatility threshold
            recommendations.append("High volatility detected - implement tighter stop losses")
        
        # Stress test recommendations
        for scenario, drawdown in stress_results.items():
            if drawdown < -0.15:  # 15% drawdown threshold
                recommendations.append(f"Vulnerable to {scenario} - consider hedging strategies")
        
        if not recommendations:
            recommendations.append("Risk levels within acceptable parameters")
        
        return recommendations 