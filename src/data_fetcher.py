import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import yaml
import yfinance as yf

class DataFetcher:
    """Fetch market data from public APIs."""

    ID_MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "SUI": "sui",
        "USDT": "tether",
        "USDC": "usd-coin",
    }

    SUPPLY = {
        "BTC": 19_700_000,
        "ETH": 120_000_000,
        "SOL": 440_000_000,
        "SUI": 10_000_000_000,
        "USDT": 110_000_000_000,
        "USDC": 33_000_000_000,
        "BNB": 150_000_000,
        "DOGE": 140_000_000_000,
        "MSTR": 15_000_000,
        "TSLA": 3_200_000_000,
        "CYFRF": 300_000_000,
    }

    BASE_PRICE = {
        "BTC": 60_000,
        "ETH": 3_000,
        "SOL": 150,
        "SUI": 1,
        "USDT": 1,
        "USDC": 1,
        "BNB": 400,
        "DOGE": 0.2,
        "MSTR": 1500,
        "TSLA": 180,
        "CYFRF": 30,
    }

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            self.config = {
                "assets": [
                    "BTC",
                    "ETH",
                    "SOL",
                    "SUI",
                    "BNB",
                    "DOGE",
                    "MSTR",
                    "TSLA",
                    "CYFRF",
                    "SUI/BTC",
                    "SOL/BTC",
                    "ETH/BTC",
                ],
                "data": {"lookback_period": 100, "timeframes": ["1d"]},
                "trading": {"risk_tolerance": 0.02, "asymmetry_threshold": 3, "pareto_weight": 0.2},
            }

        # Ensure trading defaults exist if missing
        self.config.setdefault("trading", {"risk_tolerance": 0.02, "asymmetry_threshold": 3, "pareto_weight": 0.2})

    def _generate_synthetic_data(self, asset, timeframe, end=None):
        """Fallback deterministic data generator."""
        import numpy as np
        end = end or datetime.utcnow()
        lookback = self.config["data"].get("lookback_period", 30)
        freq = "1D" if timeframe == "1d" else "1h"
        rng = pd.date_range(end=end, periods=lookback, freq=freq)
        seed = abs(hash(f"{asset}_{timeframe}")) % (2**32)
        rs = np.random.RandomState(seed)
        base_price = self.BASE_PRICE.get(asset, 1)
        returns = rs.normal(0, 0.01, len(rng))
        prices = base_price * (1 + returns).cumprod()
        return pd.DataFrame({"price": prices}, index=rng)


    def fetch_price_and_market_cap(self, asset):
        """Return current price, market cap and 24h change."""
        if "/" in asset:
            base, quote = asset.split("/")
            base_p, base_cap, _ = self.fetch_price_and_market_cap(base)
            quote_p, _, _ = self.fetch_price_and_market_cap(quote)
            price = base_p / quote_p if quote_p else 0
            df = self.fetch_market_data(asset, "1d")
            change_24h = df["price"].pct_change().iloc[-1] * 100 if len(df) > 1 else 0
            return price, base_cap, change_24h

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
                resp = requests.get(url, params=params, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                info = resp.json()[coin_id]
                return (
                    info.get("usd"),
                    info.get("usd_market_cap"),
                    info.get("usd_24h_change", 0),
                )
            except Exception:
                pass

        # Fallback to Yahoo Finance for stocks
        try:
            ticker = yf.Ticker(asset)
            info = ticker.history(period="2d")
            if not info.empty:
                price = info["Close"].iloc[-1]
                prev = info["Close"].iloc[-2] if len(info) > 1 else price
                change_24h = ((price - prev) / prev) * 100
                market_cap = ticker.info.get("marketCap")
                return float(price), market_cap, float(change_24h)
        except Exception:
            pass

        df = self._generate_synthetic_data(asset, "1d", end=datetime.utcnow())
        price = float(df["price"].iloc[-1])
        supply = self.SUPPLY.get(asset, 1_000_000)
        market_cap = price * supply
        change_24h = df["price"].pct_change().iloc[-1] * 100 if len(df) > 1 else 0
        return price, market_cap, change_24h

    def fetch_market_data(self, asset, timeframe):
        """Return historical price data as a DataFrame."""
        if "/" in asset:
            base, quote = asset.split("/")
            base_df = self.fetch_market_data(base, timeframe)
            quote_df = self.fetch_market_data(quote, timeframe)
            df = pd.DataFrame(index=base_df.index)
            df["price"] = base_df["price"] / quote_df["price"]
            return df

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
                resp = requests.get(url, params=params, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                prices = resp.json()["prices"]
                df = pd.DataFrame(prices, columns=["timestamp", "price"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return df
            except Exception:
                pass

        # Stocks via Yahoo Finance
        try:
            period = f"{self.config['data'].get('lookback_period', 30)}d"
            interval = "1d" if timeframe == "1d" else "1h"
            df = yf.download(asset, period=period, interval=interval, progress=False)
            if not df.empty:
                df = df[["Close"]].rename(columns={"Close": "price"})
                df.index.name = "timestamp"
                return df
        except Exception:
            pass

        return self._generate_synthetic_data(asset, timeframe, end=datetime.utcnow())

    def fetch_week_change(self, asset):
        """Return 7-day percent change for the asset."""
        if "/" in asset:
            df = self.fetch_market_data(asset, "1d")
            if len(df) < 2:
                return 0
            start = df["price"].iloc[-7] if len(df) >= 7 else df["price"].iloc[0]
            end = df["price"].iloc[-1]
            return ((end - start) / start) * 100

        coin_id = self.ID_MAP.get(asset)
        if coin_id:
            try:
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                resp = requests.get(
                    url,
                    params={"vs_currency": "usd", "days": 7, "interval": "daily"},
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                prices = resp.json()["prices"]
                start = prices[0][1]
                end = prices[-1][1]
                return ((end - start) / start) * 100
            except Exception:
                pass

        # Stocks via Yahoo Finance
        try:
            df = yf.download(asset, period="7d", interval="1d", progress=False)
            if len(df) >= 2:
                start = df["Close"].iloc[0]
                end = df["Close"].iloc[-1]
                return ((end - start) / start) * 100
        except Exception:
            pass

        df = self._generate_synthetic_data(asset, "1d", end=datetime.utcnow())
        start = df["price"].iloc[-7] if len(df) >= 7 else df["price"].iloc[0]
        end = df["price"].iloc[-1]
        return ((end - start) / start) * 100

