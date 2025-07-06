# Quant AI Trader

This project fetches cryptocurrency market data from CoinGecko and generates
basic trading signals. If the API is unreachable, deterministic synthetic data
is used so the app continues to run without network access.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

Set a `NEWS_API_KEY` environment variable to enable news headlines fetching.

## Command-Line Usage

Run the tool from the command line to print market summaries, technical
analysis, macro/on-chain insights, recent news and example trading signals for
BTC, SOL, SUI and SEI:

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

### API Endpoint

The web app also exposes a JSON API for integration with other tools. Start the
server and query:

```bash
curl http://localhost:5000/api/summary
```

This returns the latest market data, headlines and trading signals.

### ElizaOS Integration

A sample ElizaOS plugin is provided under `plugins/plugin-quanttrader`. Build it
with `bun run build` and include it in your Eliza project to let agents fetch
trading insights from this backend.

