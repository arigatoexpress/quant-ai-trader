# Quant AI Trader

This project fetches cryptocurrency price and market cap data using the CoinGecko API.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Command-Line Usage

Run the basic data fetcher:

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

