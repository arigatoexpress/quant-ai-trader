from flask import Flask, render_template_string
import os
import base64

from .data_fetcher import DataFetcher
from .macro_analyzer import MacroAnalyzer
from .onchain_analyzer import OnChainAnalyzer
from .news_analyzer import NewsAnalyzer
from .technical_analyzer import TechnicalAnalyzer
from .trading_agent import TradingAgent
from .utils import plot_price_chart, sanitize_filename

app = Flask(__name__)
fetcher = DataFetcher()

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Quant AI Trader</title>
</head>
<body>
<h1>Market Data Summary</h1>
<table border="1" cellpadding="5">
<tr><th>Asset</th><th>Price (USD)</th><th>Market Cap (USD)</th><th>24h %</th><th>7d %</th></tr>
{% for asset, data in assets.items() %}
<tr>
  <td>{{ asset }}</td>
  <td>{{ data.price }}</td>
  <td>{{ data.market_cap }}</td>
  <td>{{ data.change_24h }}</td>
  <td>{{ data.change_7d }}</td>
</tr>
{% endfor %}
</table>

<h2>Market Highlights</h2>
<ul>
{% for item in highlights %}
  <li>{{ item }}</li>
{% endfor %}
</ul>

<h2>Macro Insights</h2>
<ul>
{% for item in macro_insights %}
  <li>{{ item }}</li>
{% endfor %}
</ul>

<h2>Market News</h2>
<ul>
{% for item in news %}
  <li>{{ item }}</li>
{% endfor %}
</ul>

<h2>On-chain Insights</h2>
<ul>
{% for item in onchain_insights %}
  <li>{{ item }}</li>
{% endfor %}
</ul>

<h2>Technical Analysis</h2>
<ul>
{% for pair, t_list in ta.items() %}
  <li><strong>{{ pair }}:</strong> {{ ' '.join(t_list) }}</li>
{% endfor %}
</ul>

<h2>Trading Signals</h2>
<table border="1" cellpadding="5">
<tr><th>Pair</th><th>Signal</th><th>Current</th><th>Predicted</th><th>R/R</th><th>Insight</th><th>AI</th></tr>
{% for pair, sig in signals.items() %}
<tr>
  <td>{{ pair }}</td>
  <td>{{ sig.signal }}</td>
  <td>{{ "%.2f"|format(sig.current_price) }}</td>
  <td>{{ "%.2f"|format(sig.predicted_price) }}</td>
  <td>{{ "%.2f"|format(sig.risk_reward) }}</td>
  <td>{{ sig.insight }}</td>
  <td>{{ sig.ai_insight }}</td>
</tr>
{% endfor %}
</table>

<h2>Charts</h2>
{% for pair, img in charts.items() %}
  <h3>{{ pair }}</h3>
  <img src="data:image/png;base64,{{ img }}" alt="{{ pair }} chart">
{% endfor %}
</body>
</html>
"""

@app.route("/")
def index():
    assets_data = {}
    highlights_calc = []
    for asset in fetcher.config["assets"]:
        try:
            price, market_cap, change_24h = fetcher.fetch_price_and_market_cap(asset)
            week_change = fetcher.fetch_week_change(asset)
        except Exception:
            continue
        highlights_calc.append({"asset": asset, "change_24h": change_24h, "week": week_change})
        assets_data[asset] = {
            "price": f"{price:,.2f}" if price else "N/A",
            "market_cap": f"{market_cap:,.0f}" if market_cap else "N/A",
            "change_24h": f"{change_24h:.2f}%",
            "change_7d": f"{week_change:.2f}%",
        }

    best_24h = max(highlights_calc, key=lambda x: x["change_24h"])
    best_week = max(highlights_calc, key=lambda x: x["week"])
    highlights = [
        f"Top 24h Gainer: {best_24h['asset']} ({best_24h['change_24h']:.2f}%)",
        f"Top 7d Gainer: {best_week['asset']} ({best_week['week']:.2f}%)",
    ]

    news_items = NewsAnalyzer().fetch_news()
    ta = {}
    ta_engine = TechnicalAnalyzer()
    for asset in fetcher.config["assets"]:
        try:
            df = fetcher.fetch_market_data(asset, "1d")
            ta[asset] = ta_engine.analyze(df["price"])
        except Exception:
            ta[asset] = ["Data unavailable"]

    macro_data = {"ten_year_treasury": 4.7, "inflation": 3.2, "global_m2": 106e12}
    onchain_data = {"bitcoin_dominance": 58, "sui_dominance": 0.6}
    macro_insights = MacroAnalyzer(macro_data).analyze()
    onchain_insights = OnChainAnalyzer(onchain_data).analyze()

    agent = TradingAgent(fetcher.config, fetcher)
    signals = agent.generate_trade_signals()
    charts = {}
    os.makedirs('charts', exist_ok=True)
    for pair, sig in signals.items():
        asset, tf = pair.split('_')
        try:
            df = fetcher.fetch_market_data(asset, tf)
            filename = sanitize_filename(pair)
            chart_path = f"charts/{filename}.png"
            plot_price_chart(df['price'], sig['predicted_price'], chart_path)
            with open(chart_path, 'rb') as f:
                charts[pair] = base64.b64encode(f.read()).decode('utf-8')
        except Exception:
            continue

    return render_template_string(
        TEMPLATE,
        assets=assets_data,
        macro_insights=macro_insights,
        onchain_insights=onchain_insights,
        news=news_items,
        ta=ta,
        signals=signals,
        charts=charts,
        highlights=highlights,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

