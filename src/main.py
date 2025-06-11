from .data_fetcher import DataFetcher
from .macro_analyzer import MacroAnalyzer
from .onchain_analyzer import OnChainAnalyzer
from .trading_agent import TradingAgent

def main():
    print("Starting Quant AI Trader...")
    fetcher = DataFetcher()

    print("\n--- Market Data Summary ---")
    highlights = []
    for asset in fetcher.config["assets"]:
        price, market_cap, change_24h = fetcher.fetch_price_and_market_cap(asset)
        week_change = fetcher.fetch_week_change(asset)
        highlights.append({"asset": asset, "change_24h": change_24h, "week": week_change})
        print(f"{asset}:")
        print(f"  - Current Price: ${price:,.2f}")
        print(f"  - Market Cap: ${market_cap:,.0f}")
        print(f"  - 24h Change: {change_24h:.2f}%")
        print(f"  - 7d Change: {week_change:.2f}%")
        print("-" * 50)

    best_24h = max(highlights, key=lambda x: x["change_24h"])
    best_week = max(highlights, key=lambda x: x["week"])

    print("\n--- Market Highlights ---")
    print(f"Top 24h Gainer: {best_24h['asset']} ({best_24h['change_24h']:.2f}%)")
    print(f"Top 7d Gainer: {best_week['asset']} ({best_week['week']:.2f}%)")
    # Example macro and on-chain data for demonstration
    macro_data = {"ten_year_treasury": 4.7, "inflation": 3.2, "global_m2": 106e12}
    onchain_data = {"bitcoin_dominance": 58, "sui_dominance": 0.6}

    print("\n--- Macro Insights ---")
    for insight in MacroAnalyzer(macro_data).analyze():
        print(f"* {insight}")

    print("\n--- On-chain Insights ---")
    for insight in OnChainAnalyzer(onchain_data).analyze():
        print(f"* {insight}")

    print("\n--- Trading Signals ---")
    agent = TradingAgent(fetcher.config, fetcher)
    signals = agent.generate_trade_signals()
    for key, data in signals.items():
        print(f"{key}: {data['signal']} @ ${data['current_price']:.2f} -> {data['predicted_price']:.2f} (RR {data['risk_reward']:.2f})")
        print(f"  Insight: {data['insight']}")
        print("-" * 50)

if __name__ == "__main__":
    main()

