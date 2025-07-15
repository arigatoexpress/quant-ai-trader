import os
import asyncio
import yaml
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
import tweepy
from tweepy.errors import TweepyException
import praw
from prawcore.exceptions import PrawcoreException
from dotenv import load_dotenv

from openai import OpenAI  # Note: This is for compatibility, but we'll use Grok API as per repo

# Load from repo's config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

load_dotenv()

class SocialDataFetcher:
    def __init__(self):
        self.telegram_api_id = os.getenv('TELEGRAM_API_ID')
        self.telegram_api_hash = os.getenv('TELEGRAM_API_HASH')
        self.telegram_phone = os.getenv('TELEGRAM_PHONE')
        self.twitter_bearer = os.getenv('TWITTER_BEARER_TOKEN')
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.grok_api_key = CONFIG.get('grok_api_key', os.getenv('GROK_API_KEY'))  # Tie into repo's Grok config
        self.grok_client = OpenAI(api_key=self.grok_api_key, base_url="https://api.x.ai/v1")  # Use Grok endpoint

    async def fetch_telegram_signals(self, channel='cryptosignals', limit=5):
        try:
            client = TelegramClient('session', self.telegram_api_id, self.telegram_api_hash)
            await client.start(phone=self.telegram_phone)
            messages = await client.get_messages(channel, limit=limit)
            return [msg.text for msg in messages if msg.text]
        except SessionPasswordNeededError:
            return "Telegram 2FA needed - configure manually"
        except Exception as e:
            return f"Error: {str(e)}"

    def fetch_twitter_signals(self, username='CryptoWhale', count=5):
        try:
            client = tweepy.Client(bearer_token=self.twitter_bearer)
            user = client.get_user(username=username)
            tweets = client.get_users_tweets(user.data.id, max_results=count)
            return [tweet.text for tweet in tweets.data]
        except TweepyException as e:
            return f"Twitter Error: {str(e)}"

    def fetch_reddit_signals(self, subreddit='wallstreetbets', limit=5):
        try:
            reddit = praw.Reddit(client_id=self.reddit_client_id, client_secret=self.reddit_client_secret, user_agent='quant-ai-trader')
            posts = reddit.subreddit(subreddit).hot(limit=limit)
            return [post.title + ': ' + post.selftext for post in posts]
        except PrawcoreException as e:
            return f"Reddit Error: {str(e)}"

    async def aggregate_and_analyze(self):
        tg = await self.fetch_telegram_signals()
        tw = self.fetch_twitter_signals()
        rd = self.fetch_reddit_signals()
        all_data = (tg if isinstance(tg, list) else []) + (tw if isinstance(tw, list) else []) + (rd if isinstance(rd, list) else [])
        
        # AI Analysis using Grok (integrated with repo's AI setup)
        if self.grok_api_key and all_data:
            prompt = f"Analyze these social signals for trading insights, sentiment, and risk: {all_data[:10]}"  # Limit for cost
            try:
                response = self.grok_client.chat.completions.create(
                    model="grok-3-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Grok Analysis Error: {str(e)}"
        return all_data

# Example usage
if __name__ == '__main__':
    fetcher = SocialDataFetcher()
    result = asyncio.run(fetcher.aggregate_and_analyze())
    print(result) 