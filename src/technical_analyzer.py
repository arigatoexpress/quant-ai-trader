class TechnicalAnalyzer:
    """Provide simple technical analysis signals."""
    def __init__(self, asset_data):
        self.asset_data = asset_data

    def _moving_average(self, series, window):
        return series.rolling(window=window).mean()

    def _rsi(self, series, window=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def analyze(self):
        price = self.asset_data['price']
        if len(price) < 50:
            return ["Insufficient data for technical analysis."]
        ma20 = self._moving_average(price, 20)
        ma50 = self._moving_average(price, 50)
        signals = []
        if ma20.iloc[-1] > ma50.iloc[-1] and ma20.iloc[-2] <= ma50.iloc[-2]:
            signals.append("Bullish MA20/50 crossover")
        elif ma20.iloc[-1] < ma50.iloc[-1] and ma20.iloc[-2] >= ma50.iloc[-2]:
            signals.append("Bearish MA20/50 crossover")
        rsi = self._rsi(price).iloc[-1]
        if rsi > 70:
            signals.append("RSI overbought")
        elif rsi < 30:
            signals.append("RSI oversold")
        else:
            signals.append("RSI neutral")
        return signals
