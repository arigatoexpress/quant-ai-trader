from .data_fetcher import DataFetcher

def main():
    print("Starting Quant AI Trader...")
    fetcher = DataFetcher()

    print("\n--- Market Data Summary ---")
    for asset in fetcher.config["assets"]:
        price, market_cap = fetcher.fetch_price_and_market_cap(asset)
        if price and market_cap:
            print(f"{asset}:")
            print(f"  - Current Price: ${price:,.2f}")
            print(f"  - Market Cap: ${market_cap:,.0f}")
        else:
            print(f"{asset}: Data not available")
        print("-" * 50)

if __name__ == "__main__":
    main()

