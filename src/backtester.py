import numpy as np
from sklearn.linear_model import LinearRegression

class Backtester:
    """Simple regression-based backtester for a single asset."""

    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        self.model = LinearRegression()

    def run(self, asset, timeframe="1d", train_ratio=0.7):
        df = self.data_fetcher.fetch_market_data(asset, timeframe)
        if df is None or len(df) < 10:
            return {"asset": asset, "error": "Insufficient data"}

        df = df.dropna()
        split = int(len(df) * train_ratio)
        train = df.iloc[:split]
        test = df.iloc[split:]

        X_train = train.index.astype(int).values.reshape(-1, 1) // 10**9
        y_train = train["price"].values
        self.model.fit(X_train, y_train)

        X_test = test.index.astype(int).values.reshape(-1, 1) // 10**9
        predictions = self.model.predict(X_test)

        mse = float(np.mean((predictions - test["price"].values) ** 2))
        return {
            "asset": asset,
            "mse": mse,
            "last_pred": float(predictions[-1]),
            "last_actual": float(test["price"].values[-1]),
        }
