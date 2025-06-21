# Quant AI Trader

This project fetches cryptocurrency and stock market data from several public
APIs including CoinGecko, CryptoCompare and Binance. Stocks are retrieved from
Yahoo Finance. If all online feeds fail the tool can fall back to deterministic
synthetic data. Set `data.allow_synthetic: false` in `config.yaml` (the
default) to raise an error instead of using synthetic prices. When network
access is blocked you may set `QUANT_ALLOW_CACHE=1` to load the bundled CSV
history in the `data/` folder before resorting to synthetic values.
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

The application bundles a small portion of the **ElizaOS** framework. Its
`ConfigLoader` reads `config/config.yaml` and allows overrides via environment
variables or HashiCorp Vault. Set `QUANT_CONFIG_DIR` to point at an alternative
configuration directory if you want to maintain multiple configurations. Values
in the YAML can reference environment variables using the `<%= ENV['VAR'] %>`
syntax and Vault secrets with `<%= VAULT['path/to/secret'] %>`. When Vault is
not configured the loader simply ignores those references.

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

