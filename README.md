# Quant AI Trader

This project fetches cryptocurrency and stock market data from public APIs such
as CoinGecko and Yahoo Finance to generate basic trading signals. If those
services fail the tool falls back to CryptoCompare and finally to deterministic
synthetic data. Set `data.allow_synthetic: false` in `config.yaml` (the default)
to raise an error instead of using synthetic prices. If network access is not
available, the application loads cached historical prices from the `data/`
folder when the environment variable `QUANT_ALLOW_CACHE=1` is set before
resorting to synthetic values.
Market data and price history are cached in memory during execution to reduce
API calls and speed up analysis.
The application also integrates the **pysui** client to query basic
information from the Sui blockchain such as the current gas price. Set the
`sui.rpc_url` value in the configuration file to point to your preferred Sui
RPC endpoint. If network access is blocked these calls simply return
``None``.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

The application now bundles a small portion of the **ElizaOS** framework. The
embedded `ConfigLoader` lets you override settings through environment
variables or HashiCorp Vault secrets if available. By default it loads
`config/config.yaml` but you can specify a different directory via the
environment variable `QUANT_CONFIG_DIR` or the constructor.

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

Additional assets and synthetic pairs such as `SUI/BTC`, `SOL/BTC` and
`ETH/BTC` are included. The application also prints a brief market news
summary and simple technical analysis for each configured asset. A lightweight
PyTorch model trains on recent prices to provide AI-based predictions and
trading insights. Charts with the predicted next value are generated for each
asset. Chart file names are sanitized to remove unsafe characters so pairs like
`SUI/BTC` save correctly in the `charts/` folder.

If you have a CUDA-enabled GPU such as a 3060 Ti, PyTorch will
automatically utilize it when generating AI predictions.

Basic Sui network metrics are fetched using the `pysui` SDK. When accessible,
the application prints the current gas price and connected chain ID as part of
the on-chain insights.

