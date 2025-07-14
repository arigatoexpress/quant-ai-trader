"""
Performance Attribution Module
- Decomposes returns into alpha, beta, and risk factors
- Attributes P&L to signals and strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

class PerformanceAttribution:
    """Attributes performance to alpha, beta, and risk factors"""
    def __init__(self, returns: pd.Series, benchmark: pd.Series, signals: Dict[str, pd.Series]):
        self.returns = returns
        self.benchmark = benchmark
        self.signals = signals

    def calculate_alpha_beta(self) -> Dict[str, float]:
        # Linear regression for alpha/beta
        X = np.array(self.benchmark.values).reshape(-1, 1)
        y = np.array(self.returns.values)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]
        alpha = model.intercept_
        return {'alpha': alpha, 'beta': beta}

    def signal_attribution(self) -> Dict[str, float]:
        # Correlate each signal with returns
        attributions = {}
        for name, signal in self.signals.items():
            if len(signal) == len(self.returns):
                attributions[name] = np.corrcoef(self.returns, signal)[0, 1]
        return attributions

    def risk_attribution(self) -> Dict[str, float]:
        # Decompose risk by signal volatility
        risks = {}
        for name, signal in self.signals.items():
            risks[name] = np.std(signal)
        return risks 