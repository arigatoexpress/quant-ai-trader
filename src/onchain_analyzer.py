class OnChainAnalyzer:
    def __init__(self, onchain_data):
        self.onchain_data = onchain_data

    def analyze(self):
        """Analyze on-chain data and provide descriptive insights."""
        insights = []
        if self.onchain_data["bitcoin_dominance"] > 60:
            insights.append("High BTC dominance (above 60%) suggests altcoins may underperform.")
        if self.onchain_data["sui_dominance"] > 0.5:
            insights.append("SUI dominance rising (above 0.5%) indicates potential breakout or increased interest.")
        returninsights if insights else ["No significant on-chain insights at this time."]

