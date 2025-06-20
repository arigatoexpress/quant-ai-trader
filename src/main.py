import os
from .data_fetcher import DataFetcher
from .macro_analyzer import MacroAnalyzer
from .onchain_analyzer import OnChainAnalyzer
from .sui_client import SuiClient
from .news_analyzer import NewsAnalyzer
from .technical_analyzer import TechnicalAnalyzer
from .trading_agent import TradingAgent
from .utils import plot_price_chart, sanitize_filename


def main():
    print("Starting Quant AI Trader...")
    fetcher = DataFetcher()
    news = NewsAnalyzer()
    tech = TechnicalAnalyzer()

    print("\n--- Market Data Summary ---")
    highlights = []
    for asset in fetcher.config["assets"]:
        try:
            price, market_cap, change_24h = fetcher.fetch_price_and_market_cap(asset)
            week_change = fetcher.fetch_week_change(asset)
        except Exception as e:
            print(f"Failed to fetch data for {asset}: {e}")
            continue
        highlights.append(
            {"asset": asset, "change_24h": change_24h, "week": week_change}
        )
        print(f"{asset}:")
        print(f"  - Current Price: ${price:,.2f}")
        if market_cap:
            print(f"  - Market Cap: ${market_cap:,.0f}")
        print(f"  - 24h Change: {change_24h:.2f}%")
        print(f"  - 7d Change: {week_change:.2f}%")
        print("-" * 50)

    if highlights:
        best_24h = max(highlights, key=lambda x: x["change_24h"])
        best_week = max(highlights, key=lambda x: x["week"])

        print("\n--- Market Highlights ---")
        print(f"Top 24h Gainer: {best_24h['asset']} ({best_24h['change_24h']:.2f}%)")
        print(f"Top 7d Gainer: {best_week['asset']} ({best_week['week']:.2f}%)")
    else:
        print("\nNo market data available.")

    print("\n--- Market News ---")
    for headline in news.fetch_news():
        print(f"* {headline}")
    # Example macro and on-chain data for demonstration
    macro_data = {"ten_year_treasury": 4.7, "inflation": 3.2, "global_m2": 106e12}
    onchain_data = {"bitcoin_dominance": 58, "sui_dominance": 0.6}

    print("\n--- Macro Insights ---")
    for insight in MacroAnalyzer(macro_data).analyze():
        print(f"* {insight}")

    print("\n--- On-chain Insights ---")
    sui_client = SuiClient(
        fetcher.config.get("sui", {}).get(
            "rpc_url", "https://fullnode.mainnet.sui.io:443"
        )
    )
    for insight in OnChainAnalyzer(onchain_data, sui_client=sui_client).analyze():
        print(f"* {insight}")

    print("\n--- Technical Analysis ---")
    for asset in fetcher.config["assets"]:
        try:
            data = fetcher.fetch_market_data(asset, "1d")
        except Exception as e:
            print(f"TA failed for {asset}: {e}")
            continue
        ta_insights = tech.analyze(data["price"])
        print(f"{asset}: {' '.join(ta_insights)}")

    print("\n--- Trading Signals ---")
    agent = TradingAgent(fetcher.config, fetcher)
    signals = agent.generate_trade_signals()
    for key, data in signals.items():
        print(
            f"{key}: {data['signal']} @ ${data['current_price']:.2f} -> {data['predicted_price']:.2f} (RR {data['risk_reward']:.2f})"
        )
        print(f"  Insight: {data['insight']}")
        print(f"  AI Insight: {data['ai_insight']}")
        asset, tf = key.split("_")
        try:
            df = fetcher.fetch_market_data(asset, tf)
            filename = sanitize_filename(key)
            chart_path = f"charts/{filename}.png"
            os.makedirs("charts", exist_ok=True)
            plot_price_chart(df["price"], data["predicted_price"], chart_path)
            print(f"  Chart saved to {chart_path}")
        except Exception as e:
            print(f"  Failed to create chart for {key}: {e}")
        print("-" * 50)


if __name__ == "__main__":
    main()
