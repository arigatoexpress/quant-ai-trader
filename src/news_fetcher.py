import os
import requests

class NewsFetcher:
    """Fetch macro and cryptocurrency news via NewsAPI."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("NEWS_API_KEY")

    def fetch_headlines(self, query="crypto", limit=5):
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not provided")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": self.api_key,
            "pageSize": limit,
            "sortBy": "publishedAt",
            "language": "en",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a.get("title") for a in articles]

    def fetch_crypto_headlines(self, assets, limit=5):
        query = " OR ".join(assets)
        return self.fetch_headlines(query, limit)

    def fetch_macro_headlines(self, limit=5):
        query = "inflation OR interest rates OR Federal Reserve OR macroeconomy"
        return self.fetch_headlines(query, limit)
