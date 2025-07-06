from flask import Flask, render_template_string, jsonify

from .data_fetcher import DataFetcher
from .macro_analyzer import MacroAnalyzer
from .onchain_analyzer import OnChainAnalyzer
from .trading_agent import TradingAgent
from .news_fetcher import NewsFetcher
from .technical_analyzer import TechnicalAnalyzer

app = Flask(__name__)
fetcher = DataFetcher()
news_fetcher = NewsFetcher()

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
<tr><th>Asset</th><th>Price (USD)</th><th>Market Cap (USD)</th><th>24h %</th><th>7d %</th><th>Ecosystem Coins</th></tr>
{% for asset, data in assets.items() %}
<tr>
  <td>{{ asset }}</td>
  <td>{{ data.price }}</td>
  <td>{{ data.market_cap }}</td>
  <td>{{ data.change_24h }}</td>
  <td>{{ data.change_7d }}</td>
  <td>{{ data.ecosystem }}</td>
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

<h2>Market Outlook</h2>
<p>{{ outlook }}</p>

<h2>Technical Analysis</h2>
<ul>
{% for line in technical %}
  <li>{{ line }}</li>
{% endfor %}
</ul>

<h2>Crypto Headlines</h2>
<ul>
{% for item in crypto_headlines %}
  <li>{{ item }}</li>
{% endfor %}
</ul>

<h2>Macro Headlines</h2>
<ul>
{% for item in macro_headlines %}
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
        ecos = fetcher.fetch_ecosystem_coins(asset)
        highlights_calc.append({"asset": asset, "change_24h": change_24h, "week": week_change})
        assets_data[asset] = {
            "price": f"{price:,.2f}" if price else "N/A",
            "market_cap": f"{market_cap:,.0f}" if market_cap else "N/A",
            "change_24h": f"{change_24h:.2f}%" if change_24h is not None else "N/A",
            "change_7d": f"{week_change:.2f}%" if week_change is not None else "N/A",
            "ecosystem": ", ".join(ecos) if ecos else "",
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
    outlook = "Bullish" if any("breakout" in i or "boost" in i or "favor" in i for i in (macro_insights + onchain_insights)) else "Neutral"

    technical_lines = []
    for asset in fetcher.config["assets"]:
        df = fetcher.fetch_market_data(asset, "1d")
        if df is None or df.empty:
            continue
        for t in TechnicalAnalyzer(df).analyze():
            technical_lines.append(f"{asset}: {t}")

    try:
        crypto_headlines = news_fetcher.fetch_crypto_headlines(fetcher.config["assets"])
        macro_headlines = news_fetcher.fetch_macro_headlines()
    except Exception:
        crypto_headlines = []
        macro_headlines = []

    agent = TradingAgent(fetcher.config, fetcher)
    signals = agent.generate_trade_signals()

    return render_template_string(
        TEMPLATE,
        assets=assets_data,
        macro_insights=macro_insights,
        onchain_insights=onchain_insights,
        signals=signals,
        highlights=highlights,
        crypto_headlines=crypto_headlines,
        macro_headlines=macro_headlines,
        technical=technical_lines,
        outlook=outlook,
    )


@app.route("/api/summary")
def api_summary():
    """Return market data and trading signals as JSON."""
    assets_data = {}
    highlights_calc = []
    for asset in fetcher.config["assets"]:
        price, market_cap, change_24h = fetcher.fetch_price_and_market_cap(asset)
        week_change = fetcher.fetch_week_change(asset)
        ecos = fetcher.fetch_ecosystem_coins(asset)
        highlights_calc.append({"asset": asset, "change_24h": change_24h, "week": week_change})
        assets_data[asset] = {
            "price": price,
            "market_cap": market_cap,
            "change_24h": change_24h,
            "change_7d": week_change,
            "ecosystem": ecos,
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
    outlook = "Bullish" if any(
        "breakout" in i or "boost" in i or "favor" in i for i in (macro_insights + onchain_insights)
    ) else "Neutral"

    technical_lines = []
    for asset in fetcher.config["assets"]:
        df = fetcher.fetch_market_data(asset, "1d")
        if df is None or df.empty:
            continue
        for t in TechnicalAnalyzer(df).analyze():
            technical_lines.append(f"{asset}: {t}")

    try:
        crypto_headlines = news_fetcher.fetch_crypto_headlines(fetcher.config["assets"])
        macro_headlines = news_fetcher.fetch_macro_headlines()
    except Exception:
        crypto_headlines = []
        macro_headlines = []

    agent = TradingAgent(fetcher.config, fetcher)
    signals = agent.generate_trade_signals()

    return jsonify(
        {
            "assets": assets_data,
            "macro_insights": macro_insights,
            "onchain_insights": onchain_insights,
            "signals": signals,
            "highlights": highlights,
            "crypto_headlines": crypto_headlines,
            "macro_headlines": macro_headlines,
            "technical": technical_lines,
            "outlook": outlook,
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

