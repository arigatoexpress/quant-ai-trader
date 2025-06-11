from flask import Flask, render_template_string

from .data_fetcher import DataFetcher
from .macro_analyzer import MacroAnalyzer
from .onchain_analyzer import OnChainAnalyzer
from .trading_agent import TradingAgent

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

<h2>On-chain Insights</h2>
<ul>
{% for item in onchain_insights %}
  <li>{{ item }}</li>
{% endfor %}
</ul>

<h2>Trading Signals</h2>
<table border="1" cellpadding="5">
<tr><th>Pair</th><th>Signal</th><th>Current</th><th>Predicted</th><th>R/R</th><th>Insight</th></tr>
{% for pair, sig in signals.items() %}
<tr>
  <td>{{ pair }}</td>
  <td>{{ sig.signal }}</td>
  <td>{{ "%.2f"|format(sig.current_price) }}</td>
  <td>{{ "%.2f"|format(sig.predicted_price) }}</td>
  <td>{{ "%.2f"|format(sig.risk_reward) }}</td>
  <td>{{ sig.insight }}</td>
</tr>
{% endfor %}
</table>
</body>
</html>
"""

@app.route("/")
def index():
    assets_data = {}
    highlights_calc = []
    for asset in fetcher.config["assets"]:
        price, market_cap, change_24h = fetcher.fetch_price_and_market_cap(asset)
        week_change = fetcher.fetch_week_change(asset)
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

    macro_data = {"ten_year_treasury": 4.7, "inflation": 3.2, "global_m2": 106e12}
    onchain_data = {"bitcoin_dominance": 58, "sui_dominance": 0.6}
    macro_insights = MacroAnalyzer(macro_data).analyze()
    onchain_insights = OnChainAnalyzer(onchain_data).analyze()

    agent = TradingAgent(fetcher.config, fetcher)
    signals = agent.generate_trade_signals()

    return render_template_string(
        TEMPLATE,
        assets=assets_data,
        macro_insights=macro_insights,
        onchain_insights=onchain_insights,
        signals=signals,
        highlights=highlights,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

