import pandas as pd
from sklearn.linear_model import LinearRegression
from .utils import calculate_momentum, calculate_risk_reward

class TradingAgent:
    def __init__(self, config, data_fetcher):
        self.config = config
        self.data_fetcher = data_fetcher
        self.model = LinearRegression()

    def train_model(self, asset_data):
        X = asset_data.index.map(lambda x: x.timestamp()).values.reshape(-1, 1)
        y = asset_data["price"].values
        self.model.fit(X, y)

    def predict(self, asset_data, timeframe):
        last_timestamp = asset_data.index[-1].timestamp()
        step = 86400 if timeframe == "1d" else 3600  # Daily or hourly
        future_timestamps = [[last_timestamp + step]]
        return self.model.predict(future_timestamps)[0]

    def generate_trade_signals(self):
        signals = {}
        for timeframe in self.config["data"]["timeframes"]:
            for asset in self.config["assets"]:
                asset_data = self.data_fetcher.fetch_market_data(asset, timeframe)
                if asset_data is None or asset_data.empty:
                    continue
                momentum = calculate_momentum(asset_data["price"])
                self.train_model(asset_data)
                predicted_price = self.predict(asset_data, timeframe)
                current_price = asset_data["price"].iloc[-1]
                risk_reward = calculate_risk_reward(current_price, predicted_price, self.config["trading"]["risk_tolerance"])

                # Aggressive strategy: Prioritize high asymmetry (3:1+ reward-to-risk)
                if risk_reward >= self.config["trading"]["asymmetry_threshold"] and momentum > 0:
                    signals[f"{asset}_{timeframe}"] = {
                        "signal": "BUY",
                        "current_price": current_price,
                        "predicted_price": predicted_price,
                        "risk_reward": risk_reward,
                        "insight": f"{asset} ({timeframe}): High asymmetry trade detected."
                    }
                elif momentum < -0.02:  # Exit condition
                    signals[f"{asset}_{timeframe}"] = {
                        "signal": "SELL",
                        "current_price": current_price,
                        "predicted_price": predicted_price,
                        "risk_reward": risk_reward,
                        "insight": f"{asset} ({timeframe}): Momentum fading."
                    }
                else:
                    signals[f"{asset}_{timeframe}"] = {
                        "signal": "HOLD",
                        "current_price": current_price,
                        "predicted_price": predicted_price,
                        "risk_reward": risk_reward,
                        "insight": f"{asset} ({timeframe}): No asymmetric opportunity."
                    }

        # Apply Pareto rule: Sort by risk_reward and take top 20%
        sorted_signals = sorted(signals.items(), key=lambda x: x[1]["risk_reward"], reverse=True)
        top_count = max(1, int(len(sorted_signals) * self.config["trading"]["pareto_weight"]))
        return dict(sorted_signals[:top_count])

