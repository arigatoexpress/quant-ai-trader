class ElizaOS:
    """Orchestrate data gathering and provide actionable insights."""

    def __init__(self, config_path=None, macro_data=None, onchain_data=None):
        from data_fetcher import DataFetcher
        from news_fetcher import NewsFetcher
        from trading_agent import TradingAgent
        from macro_analyzer import MacroAnalyzer
        from onchain_analyzer import OnChainAnalyzer
        from technical_analyzer import TechnicalAnalyzer

        self.fetcher = DataFetcher(config_path)
        self.news_fetcher = NewsFetcher()
        self.macro_data = macro_data or {
            "ten_year_treasury": 4.7,
            "inflation": 3.2,
            "global_m2": 106e12,
        }
        self.onchain_data = onchain_data or {
            "bitcoin_dominance": 58,
            "sui_dominance": 0.6,
        }
        self.trading_agent = TradingAgent(self.fetcher.config, self.fetcher)
        self.MacroAnalyzer = MacroAnalyzer
        self.OnChainAnalyzer = OnChainAnalyzer
        self.TechnicalAnalyzer = TechnicalAnalyzer

    def gather_data(self):
        """Collect market data, insights and trade signals."""
        assets_summary = {}
        highlights = []
        for asset in self.fetcher.config["assets"]:
            price, market_cap, change_24h = self.fetcher.fetch_price_and_market_cap(asset)
            week_change = self.fetcher.fetch_week_change(asset)
            ecos = self.fetcher.fetch_ecosystem_coins(asset)
            assets_summary[asset] = {
                "price": price,
                "market_cap": market_cap,
                "change_24h": change_24h,
                "change_7d": week_change,
                "ecosystem": ecos,
            }
            highlights.append({"asset": asset, "change_24h": change_24h, "week": week_change})

        filtered = [h for h in highlights if h["change_24h"] is not None and h["week"] is not None]
        if filtered:
            best_24h = max(filtered, key=lambda x: x["change_24h"])
            best_week = max(filtered, key=lambda x: x["week"])
        else:
            best_24h = best_week = {"asset": "N/A", "change_24h": 0, "week": 0}
        market_highlights = [
            f"Top 24h Gainer: {best_24h['asset']} ({best_24h['change_24h']:.2f}%)",
            f"Top 7d Gainer: {best_week['asset']} ({best_week['week']:.2f}%)",
        ]

        technical = {}
        for asset in self.fetcher.config["assets"]:
            df = self.fetcher.fetch_market_data(asset, "1d")
            if df is None or df.empty:
                continue
            technical[asset] = self.TechnicalAnalyzer(df).analyze()

        try:
            crypto_headlines = self.news_fetcher.fetch_crypto_headlines(self.fetcher.config["assets"])
            macro_headlines = self.news_fetcher.fetch_macro_headlines()
        except Exception:
            crypto_headlines = []
            macro_headlines = []

        macro_insights = self.MacroAnalyzer(self.macro_data).analyze()
        onchain_insights = self.OnChainAnalyzer(self.onchain_data).analyze()

        outlook = "Bullish" if any(
            "breakout" in i or "boost" in i or "favor" in i for i in (macro_insights + onchain_insights)
        ) else "Neutral"

        trade_signals = self.trading_agent.generate_trade_signals()

        return {
            "assets_summary": assets_summary,
            "market_highlights": market_highlights,
            "technical": technical,
            "crypto_headlines": crypto_headlines,
            "macro_headlines": macro_headlines,
            "macro_insights": macro_insights,
            "onchain_insights": onchain_insights,
            "trade_signals": trade_signals,
            "outlook": outlook,
        }

    def print_report(self):
        """Print a comprehensive market report to the console."""
        data = self.gather_data()

        print("\n--- Market Data Summary ---")
        for asset, info in data["assets_summary"].items():
            price = info["price"]
            market_cap = info["market_cap"]
            ch24 = info["change_24h"]
            wk = info["change_7d"]
            print(f"{asset}:")
            if price is not None:
                print(f"  - Current Price: ${price:,.2f}")
                print(f"  - Market Cap: ${market_cap:,.0f}")
                ch24_str = f"{ch24:.2f}%" if ch24 is not None else "N/A"
                wk_str = f"{wk:.2f}%" if wk is not None else "N/A"
                print(f"  - 24h Change: {ch24_str}")
                print(f"  - 7d Change: {wk_str}")
            else:
                print("  - Data unavailable")
            if info["ecosystem"]:
                print(f"  - Top Ecosystem Coins: {', '.join(info['ecosystem'])}")
            print("-" * 50)

        print("\n--- Market Highlights ---")
        for line in data["market_highlights"]:
            print(line)

        print("\n--- Technical Analysis ---")
        for asset, lines in data["technical"].items():
            print(f"{asset}:")
            for t in lines:
                print(f"  * {t}")

        if data["crypto_headlines"]:
            print("\n--- Crypto Headlines ---")
            for h in data["crypto_headlines"]:
                print(f"* {h}")
        if data["macro_headlines"]:
            print("\n--- Macro Headlines ---")
            for h in data["macro_headlines"]:
                print(f"* {h}")

        print("\n--- Macro Insights ---")
        for i in data["macro_insights"]:
            print(f"* {i}")

        print("\n--- On-chain Insights ---")
        for i in data["onchain_insights"]:
            print(f"* {i}")

        print("\n--- Market Outlook ---")
        print(data["outlook"])

        print("\n--- Trading Signals ---")
        for key, info in data["trade_signals"].items():
            print(f"{key}: {info['signal']} @ ${info['current_price']:.2f} -> {info['predicted_price']:.2f} (RR {info['risk_reward']:.2f})")
            print(f"  Insight: {info['insight']}")
            print("-" * 50)
