import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yaml

class DataFetcher:
    """Generate synthetic market data for offline usage."""

    SUPPLY = {
        "BTC": 19_700_000,
        "ETH": 120_000_000,
        "SOL": 440_000_000,
        "SUI": 10_000_000_000,
        "USDT": 110_000_000_000,
        "USDC": 33_000_000_000,
    }

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {
                "assets": ["BTC", "ETH", "SOL", "SUI", "USDT", "USDC"],
                "data": {"lookback_period": 100, "timeframes": ["1d"]},
                "trading": {"risk_tolerance": 0.02, "asymmetry_threshold": 3, "pareto_weight": 0.2},
            }

        # Ensure trading defaults exist if missing
        self.config.setdefault("trading", {"risk_tolerance": 0.02, "asymmetry_threshold": 3, "pareto_weight": 0.2})

    def _generate_synthetic_data(self, asset, timeframe):
        """Create a deterministic random walk for the given asset."""
        lookback = self.config["data"]["lookback_period"]
        freq = "1D" if timeframe == "1d" else "1h"
        end = datetime.utcnow()
        rng = pd.date_range(end=end, periods=lookback, freq=freq)
        seed = abs(hash(f"{asset}_{timeframe}")) % (2**32)
        rs = np.random.RandomState(seed)
        prices = 100 + rs.randn(len(rng)).cumsum()
        return pd.DataFrame({"price": prices}, index=rng)

    def fetch_price_and_market_cap(self, asset):
        """Return synthetic price and market cap for the given asset."""
        df = self._generate_synthetic_data(asset, "1d")
        price = float(df["price"].iloc[-1])
        supply = self.SUPPLY.get(asset, 1_000_000)
        market_cap = price * supply
        return price, market_cap

    def fetch_market_data(self, asset, timeframe):
        """Return OHLC data as a DataFrame."""
        return self._generate_synthetic_data(asset, timeframe)

