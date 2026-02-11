# quant-ai-trader

Enterprise quantitative trading system with free-tier data, AI analysis, 2FA web dashboard, and multi-chain portfolio tracking.

## Commands

| Command | Description |
|---------|-------------|
| `python start_trader.py` | Start trading engine |
| `python src/web_app.py` | Launch web dashboard |
| `docker-compose up -d` | Start full stack (postgres, grafana, app) |
| `pytest` | Run tests with coverage |
| `cp .env.template .env` | Create env file from template |

## Architecture

```
quant-ai-trader/
  src/
    main.py               # QuantAITrader orchestrator
    secure_web_app.py      # 2FA dashboard (Flask/FastAPI)
    simple_free_data.py    # CoinGecko + DeFi Llama + DexScreener
    web_app_legacy.py      # Legacy dashboard
  config/                  # Credentials (gitignored: .encryption_key, .jwt_secret)
  scripts/                 # Deployment scripts (deploy_gcp.sh)
  start_trader.py          # Entry point with env validation
  docker-compose.yml       # Postgres + Grafana + app
  y/google-cloud-sdk/      # Vendored GCP SDK (~4000 files, dockerignored)
```

**Flow:** DataFetcher → TechnicalAnalyzer → SentimentAnalyzer → TradingAgent

## Environment

Required:
- `GROK_API_KEY` or `OPENAI_API_KEY` - AI analysis engine
- `MASTER_PASSWORD` - Web dashboard login (no hardcoded fallback)
- `JWT_SECRET_KEY` - Session tokens (32+ chars)
- `TOTP_SECRET` - 2FA authenticator secret

Optional:
- `PAPER_TRADING=true` - Default; set `false` for live trading
- `USE_FREE_TIER=true` - Free data sources (CoinGecko, DeFi Llama)
- `SUI_WALLET_ADDRESS`, `SOL_WALLET_ADDRESS`, `ETH_WALLET_ADDRESS` - Portfolio tracking
- `TRADINGVIEW_USERNAME` + `TRADINGVIEW_PASSWORD` - Premium charts

## Gotchas

- **Branch is `master`**, not `main`
- 2FA secret must be generated and added to authenticator app before first login
- Hardcoded secret fallbacks were removed - env vars are now mandatory
- `config/.encryption_key` and `config/.jwt_secret` are gitignored
- Docker-compose passwords use `${POSTGRES_PASSWORD:-changeme}` and `${GRAFANA_ADMIN_PASSWORD:-changeme}`
- Free tier costs ~$0.50-5/month (AI API usage only)
- `y/google-cloud-sdk/` is vendored and excluded from Docker builds
