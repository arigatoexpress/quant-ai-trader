# Quant AI Trader

This project fetches cryptocurrency market data from CoinGecko and generates
basic trading signals. If the API is unreachable the tool falls back to
deterministic synthetic data so it always produces output.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Command-Line Usage

Run the tool from the command line to print market summaries, macro/on-chain
insights and example trading signals:

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
trading signals, 24h/7d performance metrics and a short list of market
highlights.

