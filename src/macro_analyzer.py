class MacroAnalyzer:
    def __init__(self, macro_data):
        self.macro_data = macro_data

    def analyze(self):
        """Analyze macro data and provide descriptive insights."""
        insights = []
        if self.macro_data["ten_year_treasury"] > 4.5:
            insights.append("High treasury yields (above 4.5%) may signal risk-off sentiment. Consider favoring BTC over stocks.")
        if self.macro_data["inflation"] > 3.5:
            insights.append("Rising inflation (above 3.5%) could favor BTC as an inflation hedge.")
        if self.macro_data["global_m2"] > 105e12:
            insights.append("Global liquidity surge (M2 > $105T) may boost risk assets like crypto and stocks.")
        return insights if insights else ["No significant macro insights at this time."]