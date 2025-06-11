from flask import Flask, render_template_string
from .data_fetcher import DataFetcher

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
<tr><th>Asset</th><th>Price (USD)</th><th>Market Cap (USD)</th></tr>
{% for asset, data in assets.items() %}
<tr>
  <td>{{ asset }}</td>
  <td>{{ data.price }}</td>
  <td>{{ data.market_cap }}</td>
</tr>
{% endfor %}
</table>
</body>
</html>
"""

@app.route("/")
def index():
    assets_data = {}
    for asset in fetcher.config["assets"]:
        price, market_cap = fetcher.fetch_price_and_market_cap(asset)
        assets_data[asset] = {
            "price": f"{price:,.2f}" if price else "N/A",
            "market_cap": f"{market_cap:,.0f}" if market_cap else "N/A",
        }
    return render_template_string(TEMPLATE, assets=assets_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

