# Quant AI Trader

This project fetches cryptocurrency market data from CoinGecko and now also
supports pulling prices from CSV files hosted on GitHub. If online APIs are
unreachable you can point the app at your own historical datasets so it never
needs to rely on synthetic prices.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

Set a `NEWS_API_KEY` environment variable to enable news headlines fetching.

## Command-Line Usage

Run the tool from the command line to launch the enhanced **ElizaOS** agent.
It prints consolidated market summaries, macro/on-chain analysis, technical
signals, recent news and actionable trade ideas for BTC, SOL, SUI and SEI:

```bash
python -m src.main
```

## Web Interface

A small Flask server is provided for a simple web interface.
Start it with:

```bash
python -m src.web_app
```

It will listen on `http://localhost:5000`. To expose it to your local
network, run:

```bash
FLASK_APP=src.web_app flask run --host 0.0.0.0
```

Then access the interface from another device using your machine's IP
address.

The web view displays the same insights as the command line including a table of
trading signals, technical analysis, 24h/7d performance metrics and market
headlines. The app will gracefully fall back to generated data if CoinGecko is
unreachable, ensuring all features continue to work.

The new `ElizaOS` component orchestrates these modules to provide a single
entry point for rich analytics and trade ideas, making the application act as a
comprehensive AI trading assistant.

### Using Custom Data Sources

Add CSV URLs under the `data_urls` section of `config/config.yaml` to load
historical prices from GitHub (or any direct link). Example:

```yaml
data_urls:
  BTC:
    raw: https://raw.githubusercontent.com/Zombie-3000/Bitfinex-historical-data/master/BTCUSD/Candles_1m/2013/merged.csv
```

ElizaOS will resample these files to the requested timeframe and use them before
falling back to APIs or synthetic data.

