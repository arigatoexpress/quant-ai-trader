import pandas as pd


class TechnicalAnalyzer:
    """Generate simple technical analysis insights."""

    def rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def analyze(self, price_series):
        insights = []
        sma50 = price_series.rolling(50).mean().iloc[-1]
        sma200 = price_series.rolling(200).mean().iloc[-1]
        last_price = price_series.iloc[-1]
        if pd.notna(sma50) and pd.notna(sma200):
            if sma50 > sma200:
                insights.append("Bullish 50/200 SMA crossover in place.")
            else:
                insights.append("Bearish 50/200 SMA posture.")
        rsi_val = self.rsi(price_series).iloc[-1]
        if rsi_val > 70:
            insights.append("RSI indicates overbought conditions.")
        elif rsi_val < 30:
            insights.append("RSI indicates oversold conditions.")
        else:
            insights.append("RSI is neutral.")
        return insights
