import torch
import torch.nn as nn
import numpy as np


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class AIAnalyzer:
    """Simple neural network predictor for price data."""

    def __init__(self, lookback=30, epochs=20, lr=0.001):
        self.lookback = lookback
        self.epochs = epochs
        self.lr = lr

    def _build_model(self, input_size):
        return nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def train_and_predict(self, price_series):
        if len(price_series) <= self.lookback:
            return price_series.iloc[-1]
        data = price_series.values.astype("float32")
        X = []
        y = []
        for i in range(len(data) - self.lookback):
            X.append(data[i : i + self.lookback])
            y.append(data[i + self.lookback])
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        X = torch.tensor(X, device=device())
        y = torch.tensor(y, device=device())
        model = self._build_model(self.lookback).to(device())
        loss_fn = nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            optim.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()
        last_seq = torch.tensor(
            data[-self.lookback :], dtype=torch.float32, device=device()
        ).view(1, -1)
        with torch.no_grad():
            pred = model(last_seq).item()
        return float(pred)
