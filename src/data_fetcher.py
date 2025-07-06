import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yaml

class DataFetcher:
    """Fetch market data from CoinGecko."""

    ID_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "SUI": "sui",
        "USDT": "tether",
        "USDC": "usd-coin",
        "SEI": "sei-network",
    }

    SUPPLY = {
        "BTC": 19_700_000,
        "ETH": 120_000_000,
        "SOL": 440_000_000,
        "SUI": 10_000_000_000,
        "USDT": 110_000_000_000,
        "USDC": 33_000_000_000,
        "SEI": 10_000_000_000,
    }

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {
                "assets": ["BTC", "SOL", "SUI", "SEI"],
                "data": {"lookback_period": 100, "timeframes": ["1d", "1h"]},
                "trading": {
                    "risk_tolerance": 0.02,
                    "asymmetry_threshold": 3,
                    "pareto_weight": 0.2,
                },
            }

        # Ensure trading defaults exist if missing
        self.config.setdefault("trading", {"risk_tolerance": 0.02, "asymmetry_threshold": 3, "pareto_weight": 0.2})

    def _generate_synthetic_data(self, asset, timeframe):
        """Create deterministic synthetic prices for offline fallback."""
        lookback = self.config["data"].get("lookback_period", 30)
        freq = "1D" if timeframe == "1d" else "1h"
        end = datetime.utcnow()
        rng = pd.date_range(end=end, periods=lookback, freq=freq)
        seed = abs(hash(f"{asset}_{timeframe}")) % (2**32)
        rs = np.random.RandomState(seed)
        prices = 100 + rs.randn(len(rng)).cumsum()
        return pd.DataFrame({"price": prices}, index=rng)


    def fetch_price_and_market_cap(self, asset):
        """Return current price, market cap and 24h change."""
        coin_id = self.ID_MAP.get(asset)
        if coin_id:
            try:
                url = "https://api.coingecko.com/api/v3/simple/price"
                params = {
                    "ids": coin_id,
                    "vs_currencies": "usd",
                    "include_market_cap": "true",
                    "include_24hr_change": "true",
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                info = resp.json()[coin_id]
                return (
                    info.get("usd"),
                    info.get("usd_market_cap"),
                    info.get("usd_24h_change", 0),
                )
            except Exception:
                pass

        df = self._generate_synthetic_data(asset, "1d")
        price = float(df["price"].iloc[-1])
        supply = self.SUPPLY.get(asset, 1_000_000)
        market_cap = price * supply
        change_24h = df["price"].pct_change().iloc[-1] * 100 if len(df) > 1 else 0
        return price, market_cap, change_24h

    def fetch_market_data(self, asset, timeframe):
        """Return historical price data as a DataFrame."""
        coin_id = self.ID_MAP.get(asset)
        if coin_id:
            try:
                if timeframe == "1d":
                    days = self.config["data"].get("lookback_period", 30)
                    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
                else:
                    days = min(7, self.config["data"].get("lookback_period", 7))
                    params = {"vs_currency": "usd", "days": days, "interval": "hourly"}
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                prices = resp.json()["prices"]
                df = pd.DataFrame(prices, columns=["timestamp", "price"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return df
            except Exception:
                pass
        return self._generate_synthetic_data(asset, timeframe)

    def fetch_week_change(self, asset):
        """Return 7-day percent change for the asset."""
        coin_id = self.ID_MAP.get(asset)
        if coin_id:
            try:
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                resp = requests.get(url, params={"vs_currency": "usd", "days": 7, "interval": "daily"}, timeout=10)
                resp.raise_for_status()
                prices = resp.json()["prices"]
                start = prices[0][1]
                end = prices[-1][1]
                return ((end - start) / start) * 100
            except Exception:
                pass
        df = self._generate_synthetic_data(asset, "1d")
        start = df["price"].iloc[-7] if len(df) >= 7 else df["price"].iloc[0]
        end = df["price"].iloc[-1]
        return ((end - start) / start) * 100

    def fetch_ecosystem_coins(self, asset, limit=5):
        """Return symbols of top coins in the asset's ecosystem."""
        coin_id = self.ID_MAP.get(asset)
        if not coin_id:
            return []
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "category": f"{coin_id}-ecosystem",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": "false",
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [d.get("symbol", "").upper() for d in data]
        except Exception:
            return []

