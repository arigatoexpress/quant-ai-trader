from .sui_client import SuiClient


class OnChainAnalyzer:
    def __init__(self, onchain_data, sui_client: SuiClient | None = None):
        self.onchain_data = onchain_data
        self.sui_client = sui_client or SuiClient()

    def analyze(self):
        """Analyze on-chain data and provide descriptive insights."""
        insights = []
        if self.onchain_data["bitcoin_dominance"] > 60:
            insights.append(
                "High BTC dominance (above 60%) suggests altcoins may underperform."
            )
        if self.onchain_data["sui_dominance"] > 0.5:
            insights.append(
                "SUI dominance rising (above 0.5%) indicates potential breakout or increased interest."
            )
        gas_price = self.sui_client.current_gas_price()
        if gas_price:
            insights.append(f"Sui gas price {gas_price:,} MIST")
        chain_id = self.sui_client.chain_id()
        if chain_id:
            insights.append(f"Connected to Sui chain {chain_id}")
        return (
            insights if insights else ["No significant on-chain insights at this time."]
        )
