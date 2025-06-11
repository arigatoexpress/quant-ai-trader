import requests
import yaml
import os

class DataFetcher:
    def __init__(self, config_path=None):
        # Default to config.yaml in the config directory if no path is provided
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            # Fallback to a default asset list if config file is missing
            self.config = {"assets": ["BTC", "ETH", "SOL", "SUI", "USDT", "USDC"]}

    def fetch_price_and_market_cap(self, asset):
        """Fetch price and market cap for a given asset using CoinGecko API."""
        try:
            # CoinGecko API endpoint (asset IDs must be lowercase)
            url = f"https://api.coingecko.com/api/v3/coins/{asset.lower()}"
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad HTTP responses
            data = response.json()
            price = data["market_data"]["current_price"]["usd"]
            market_cap = data["market_data"]["market_cap"]["usd"]
            return price, market_cap
        except Exception as e:
            print(f"Error fetching data for {asset}: {e}")
            return None, None

