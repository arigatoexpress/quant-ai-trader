from .data_fetcher import DataFetcher
from .macro_analyzer import MacroAnalyzer
from .onchain_analyzer import OnChainAnalyzer
from .trading_agent import TradingAgent
from .news_fetcher import NewsFetcher
from .technical_analyzer import TechnicalAnalyzer

def main():
    print("Starting Quant AI Trader...")
    fetcher = DataFetcher()
    news_fetcher = NewsFetcher()

    print("\n--- Market Data Summary ---")
    highlights = []
    for asset in fetcher.config["assets"]:
        price, market_cap, change_24h = fetcher.fetch_price_and_market_cap(asset)
        week_change = fetcher.fetch_week_change(asset)
        ecos = fetcher.fetch_ecosystem_coins(asset)
        highlights.append({"asset": asset, "change_24h": change_24h, "week": week_change})
        print(f"{asset}:")
        if price is not None:
            print(f"  - Current Price: ${price:,.2f}")
            print(f"  - Market Cap: ${market_cap:,.0f}")
            ch24 = f"{change_24h:.2f}%" if change_24h is not None else "N/A"
            wk = f"{week_change:.2f}%" if week_change is not None else "N/A"
            print(f"  - 24h Change: {ch24}")
            print(f"  - 7d Change: {wk}")
        else:
            print("  - Data unavailable")
        if ecos:
            print(f"  - Top Ecosystem Coins: {', '.join(ecos)}")
        print("-" * 50)

    filtered = [h for h in highlights if h["change_24h"] is not None and h["week"] is not None]
    if filtered:
        best_24h = max(filtered, key=lambda x: x["change_24h"])
        best_week = max(filtered, key=lambda x: x["week"])
    else:
        best_24h = best_week = {"asset": "N/A", "change_24h": 0, "week": 0}

    print("\n--- Market Highlights ---")
    print(f"Top 24h Gainer: {best_24h['asset']} ({best_24h['change_24h']:.2f}%)")
    print(f"Top 7d Gainer: {best_week['asset']} ({best_week['week']:.2f}%)")

    print("\n--- Technical Analysis ---")
    for asset in fetcher.config["assets"]:
        df = fetcher.fetch_market_data(asset, "1d")
        if df is None or df.empty:
            continue
        tech = TechnicalAnalyzer(df).analyze()
        print(f"{asset}:")
        for t in tech:
            print(f"  * {t}")

    try:
        crypto_headlines = news_fetcher.fetch_crypto_headlines(fetcher.config["assets"])
        macro_headlines = news_fetcher.fetch_macro_headlines()
    except Exception as e:
        print("\n--- News Fetch Error ---")
        print(str(e))
        crypto_headlines = []
        macro_headlines = []

    if crypto_headlines:
        print("\n--- Crypto Headlines ---")
        for h in crypto_headlines:
            print(f"* {h}")
    if macro_headlines:
        print("\n--- Macro Headlines ---")
        for h in macro_headlines:
            print(f"* {h}")
    # Example macro and on-chain data for demonstration
    macro_data = {"ten_year_treasury": 4.7, "inflation": 3.2, "global_m2": 106e12}
    onchain_data = {"bitcoin_dominance": 58, "sui_dominance": 0.6}

    print("\n--- Macro Insights ---")
    for insight in MacroAnalyzer(macro_data).analyze():
        print(f"* {insight}")

    print("\n--- On-chain Insights ---")
    for insight in OnChainAnalyzer(onchain_data).analyze():
        print(f"* {insight}")

    outlook = "Bullish" if any("breakout" in i or "boost" in i or "favor" in i for i in (MacroAnalyzer(macro_data).analyze() + OnChainAnalyzer(onchain_data).analyze())) else "Neutral"
    print("\n--- Market Outlook ---")
    print(outlook)

    print("\n--- Trading Signals ---")
    agent = TradingAgent(fetcher.config, fetcher)
    signals = agent.generate_trade_signals()
    for key, data in signals.items():
        print(f"{key}: {data['signal']} @ ${data['current_price']:.2f} -> {data['predicted_price']:.2f} (RR {data['risk_reward']:.2f})")
        print(f"  Insight: {data['insight']}")
        print("-" * 50)

if __name__ == "__main__":
    main()

