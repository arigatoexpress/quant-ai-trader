# Quant AI Trader

This project generates example cryptocurrency metrics and trading signals. Real-time
API access is disabled in this environment, so the data fetcher creates synthetic
price series for demonstration purposes.

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
synthetic trading signals and macro/on-chain bullet points.

