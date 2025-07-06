# Quant Trader Plugin

This is an ElizaOS plugin that fetches market summaries and trading signals from the Quant AI Trader backend.

## Usage

1. Install dependencies and build:
   ```bash
   bun install
   bun run build
   ```
2. Copy this plugin into your ElizaOS project or use `elizaos plugins add`.
3. Start the Quant AI Trader web server:
   ```bash
   python -m src.web_app
   ```
4. Launch ElizaOS with this plugin enabled:
   ```bash
   elizaos start
   ```
