"""
Advanced Sentiment Analysis Engine for Trading
Analyzes news, social media, and market data for trading signals
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import time
import threading
import re
import asyncio
import aiohttp
from dataclasses import dataclass
from textblob import TextBlob
import yfinance as yf
from pycoingecko import CoinGeckoAPI
import tweepy
import feedparser
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SentimentSignal:
    """Sentiment signal with confidence metrics"""
    asset: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # 'BEARISH', 'NEUTRAL', 'BULLISH'
    confidence: float  # 0 to 1
    sources: List[str]  # ['news', 'social', 'market']
    timestamp: datetime
    key_phrases: List[str]
    volume: int  # Number of mentions
    momentum: float  # Sentiment momentum
    supporting_data: Dict[str, Any]

@dataclass
class NewsArticle:
    """News article structure"""
    title: str
    content: str
    source: str
    url: str
    published_date: datetime
    sentiment_score: float
    relevance_score: float
    asset_mentions: List[str]

@dataclass
class SocialPost:
    """Social media post structure"""
    content: str
    source: str  # 'twitter', 'reddit', etc.
    author: str
    post_id: str
    timestamp: datetime
    sentiment_score: float
    engagement_score: float
    asset_mentions: List[str]

class SentimentAnalysisEngine:
    """Advanced sentiment analysis engine for trading signals"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                import yaml
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Initialize APIs
        self.cg = CoinGeckoAPI()
        self.news_sources = self._initialize_news_sources()
        self.twitter_api = self._initialize_twitter_api()
        
        # Sentiment models
        self.sentiment_models = {}
        self.trained_models = {}
        
        # Data storage
        self.news_data = []
        self.social_data = []
        self.market_data = {}
        self.sentiment_history = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Asset mappings
        self.asset_mappings = {
            'BTC': ['bitcoin', 'btc', '$btc', 'bitcoin', 'cryptocurrency'],
            'ETH': ['ethereum', 'eth', '$eth', 'ether', 'ethereum'],
            'SUI': ['sui', '$sui', 'sui network', 'sui blockchain'],
            'SOL': ['solana', 'sol', '$sol', 'solana blockchain'],
            'SEI': ['sei', '$sei', 'sei network', 'sei protocol']
        }
        
        print("📊 Sentiment Analysis Engine initialized")
        print(f"   News Sources: {len(self.news_sources)}")
        print(f"   Assets Monitored: {len(self.asset_mappings)}")
        
    def _default_config(self):
        return {
            'assets': ['BTC', 'ETH', 'SUI', 'SOL', 'SEI'],
            'sentiment_threshold': 0.6,
            'confidence_threshold': 0.7,
            'news_sources': [
                'coindesk', 'cointelegraph', 'decrypt', 'theblock',
                'bitcoinist', 'cryptonews', 'newsbtc'
            ],
            'social_sources': ['twitter', 'reddit'],
            'update_interval': 300,  # 5 minutes
            'sentiment_weights': {
                'news': 0.4,
                'social': 0.3,
                'market': 0.3
            },
            'time_windows': {
                'short': 1,  # 1 hour
                'medium': 6,  # 6 hours
                'long': 24  # 24 hours
            }
        }
    
    def _initialize_news_sources(self) -> Dict[str, str]:
        """Initialize news source RSS feeds"""
        return {
            'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
            'cointelegraph': 'https://cointelegraph.com/rss',
            'decrypt': 'https://decrypt.co/feed',
            'theblock': 'https://www.theblock.co/rss.xml',
            'bitcoinist': 'https://bitcoinist.com/feed/',
            'cryptonews': 'https://cryptonews.com/news/feed/',
            'newsbtc': 'https://www.newsbtc.com/feed/',
            'coinmarketcap': 'https://coinmarketcap.com/headlines/rss',
            'cryptoslate': 'https://cryptoslate.com/feed/'
        }
    
    def _initialize_twitter_api(self) -> Optional[tweepy.API]:
        """Initialize Twitter API (if credentials available)"""
        try:
            # These would need to be set in environment variables
            consumer_key = os.getenv('TWITTER_CONSUMER_KEY')
            consumer_secret = os.getenv('TWITTER_CONSUMER_SECRET')
            access_token = os.getenv('TWITTER_ACCESS_TOKEN')
            access_token_secret = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
            
            if all([consumer_key, consumer_secret, access_token, access_token_secret]):
                auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
                auth.set_access_token(access_token, access_token_secret)
                api = tweepy.API(auth, wait_on_rate_limit=True)
                return api
            else:
                print("⚠️  Twitter API credentials not found - social sentiment disabled")
                return None
                
        except Exception as e:
            print(f"❌ Error initializing Twitter API: {e}")
            return None
    
    def fetch_news_sentiment(self, hours_back: int = 24) -> List[NewsArticle]:
        """Fetch and analyze news sentiment"""
        print(f"📰 Fetching news sentiment (last {hours_back} hours)...")
        
        news_articles = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for source_name, rss_url in self.news_sources.items():
            try:
                # Parse RSS feed
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:20]:  # Limit to recent articles
                    try:
                        # Parse publication date
                        pub_date = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                        
                        if pub_date < cutoff_time:
                            continue
                        
                        # Extract content
                        title = entry.title
                        content = entry.summary if hasattr(entry, 'summary') else title
                        
                        # Clean content
                        content = re.sub(r'<[^>]+>', '', content)  # Remove HTML
                        content = re.sub(r'http\S+', '', content)  # Remove URLs
                        
                        # Analyze sentiment
                        sentiment_score = self._analyze_text_sentiment(title + " " + content)
                        
                        # Check asset relevance
                        asset_mentions = self._extract_asset_mentions(title + " " + content)
                        
                        if asset_mentions:  # Only include if relevant to tracked assets
                            article = NewsArticle(
                                title=title,
                                content=content,
                                source=source_name,
                                url=entry.link,
                                published_date=pub_date,
                                sentiment_score=sentiment_score,
                                relevance_score=self._calculate_relevance_score(content, asset_mentions),
                                asset_mentions=asset_mentions
                            )
                            
                            news_articles.append(article)
                            
                    except Exception as e:
                        print(f"⚠️  Error processing article from {source_name}: {e}")
                        continue
                        
            except Exception as e:
                print(f"❌ Error fetching from {source_name}: {e}")
                continue
        
        self.news_data.extend(news_articles)
        print(f"✅ Fetched {len(news_articles)} relevant news articles")
        
        return news_articles
    
    def fetch_social_sentiment(self, hours_back: int = 24) -> List[SocialPost]:
        """Fetch and analyze social media sentiment"""
        print(f"🐦 Fetching social sentiment (last {hours_back} hours)...")
        
        social_posts = []
        
        # Twitter sentiment
        if self.twitter_api:
            social_posts.extend(self._fetch_twitter_sentiment(hours_back))
        
        # Reddit sentiment (would need PRAW setup)
        # social_posts.extend(self._fetch_reddit_sentiment(hours_back))
        
        self.social_data.extend(social_posts)
        print(f"✅ Fetched {len(social_posts)} social media posts")
        
        return social_posts
    
    def _fetch_twitter_sentiment(self, hours_back: int) -> List[SocialPost]:
        """Fetch Twitter sentiment data"""
        posts = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for asset, keywords in self.asset_mappings.items():
            for keyword in keywords[:2]:  # Limit to avoid rate limits
                try:
                    # Search tweets
                    tweets = tweepy.Cursor(
                        self.twitter_api.search_tweets,
                        q=keyword,
                        lang="en",
                        result_type="recent",
                        tweet_mode="extended"
                    ).items(100)
                    
                    for tweet in tweets:
                        try:
                            tweet_time = tweet.created_at
                            
                            if tweet_time < cutoff_time:
                                continue
                            
                            # Skip retweets
                            if hasattr(tweet, 'retweeted_status'):
                                continue
                            
                            content = tweet.full_text
                            
                            # Clean content
                            content = re.sub(r'RT @\w+:', '', content)
                            content = re.sub(r'@\w+', '', content)
                            content = re.sub(r'#\w+', '', content)
                            content = re.sub(r'http\S+', '', content)
                            
                            # Analyze sentiment
                            sentiment_score = self._analyze_text_sentiment(content)
                            
                            # Calculate engagement score
                            engagement_score = (
                                tweet.retweet_count * 2 + 
                                tweet.favorite_count + 
                                tweet.reply_count * 3
                            ) / 100
                            
                            # Extract asset mentions
                            asset_mentions = self._extract_asset_mentions(content)
                            
                            if asset_mentions:
                                post = SocialPost(
                                    content=content,
                                    source='twitter',
                                    author=tweet.user.screen_name,
                                    post_id=str(tweet.id),
                                    timestamp=tweet_time,
                                    sentiment_score=sentiment_score,
                                    engagement_score=engagement_score,
                                    asset_mentions=asset_mentions
                                )
                                
                                posts.append(post)
                                
                        except Exception as e:
                            print(f"⚠️  Error processing tweet: {e}")
                            continue
                            
                except Exception as e:
                    print(f"❌ Error fetching tweets for {keyword}: {e}")
                    continue
        
        return posts
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using TextBlob"""
        try:
            # Clean text
            text = re.sub(r'[^\w\s]', '', text.lower())
            
            # Basic sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Enhance with crypto-specific sentiment
            crypto_sentiment = self._crypto_specific_sentiment(text)
            
            # Weighted combination
            final_sentiment = (polarity * 0.7) + (crypto_sentiment * 0.3)
            
            return np.clip(final_sentiment, -1, 1)
            
        except Exception as e:
            print(f"⚠️  Error analyzing sentiment: {e}")
            return 0.0
    
    def _crypto_specific_sentiment(self, text: str) -> float:
        """Analyze crypto-specific sentiment indicators"""
        # Positive indicators
        positive_terms = [
            'moon', 'bullish', 'pump', 'rally', 'surge', 'breakout',
            'adoption', 'partnership', 'upgrade', 'innovation', 'hodl',
            'diamond hands', 'to the moon', 'green candles', 'ath'
        ]
        
        # Negative indicators
        negative_terms = [
            'dump', 'crash', 'bearish', 'fud', 'panic', 'correction',
            'regulation', 'ban', 'hack', 'scam', 'rug pull', 'paper hands',
            'red candles', 'liquidation', 'selloff'
        ]
        
        # Neutral but important terms
        neutral_terms = [
            'stable', 'sideways', 'consolidation', 'range', 'support',
            'resistance', 'volume', 'analysis', 'technical', 'fundamental'
        ]
        
        positive_score = sum(1 for term in positive_terms if term in text)
        negative_score = sum(1 for term in negative_terms if term in text)
        
        if positive_score + negative_score == 0:
            return 0.0
        
        return (positive_score - negative_score) / (positive_score + negative_score)
    
    def _extract_asset_mentions(self, text: str) -> List[str]:
        """Extract asset mentions from text"""
        mentions = []
        text_lower = text.lower()
        
        for asset, keywords in self.asset_mappings.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    mentions.append(asset)
                    break
        
        return list(set(mentions))
    
    def _calculate_relevance_score(self, content: str, asset_mentions: List[str]) -> float:
        """Calculate relevance score for content"""
        # Base score from asset mentions
        base_score = min(len(asset_mentions) * 0.3, 1.0)
        
        # Boost for specific trading terms
        trading_terms = [
            'price', 'trading', 'market', 'buy', 'sell', 'investment',
            'analysis', 'forecast', 'prediction', 'signal', 'chart'
        ]
        
        trading_score = sum(1 for term in trading_terms if term in content.lower())
        trading_boost = min(trading_score * 0.1, 0.5)
        
        return min(base_score + trading_boost, 1.0)
    
    def fetch_market_sentiment(self, asset: str) -> Dict[str, float]:
        """Fetch market-based sentiment indicators"""
        try:
            # Get market data
            if asset == 'BTC':
                ticker = yf.Ticker('BTC-USD')
            elif asset == 'ETH':
                ticker = yf.Ticker('ETH-USD')
            elif asset == 'SOL':
                ticker = yf.Ticker('SOL-USD')
            else:
                # Use CoinGecko for other assets
                return self._fetch_coingecko_sentiment(asset)
            
            # Get recent data
            data = ticker.history(period='7d', interval='1h')
            
            if data.empty:
                return {'fear_greed': 0.5, 'momentum': 0.0, 'volume': 0.0}
            
            # Calculate sentiment indicators
            sentiment_indicators = {}
            
            # Price momentum
            price_change = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
            sentiment_indicators['momentum'] = np.tanh(price_change * 10)  # Normalize
            
            # Volume sentiment
            avg_volume = data['Volume'].mean()
            recent_volume = data['Volume'].iloc[-24:].mean()  # Last 24 hours
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            sentiment_indicators['volume'] = np.tanh((volume_ratio - 1) * 2)
            
            # Volatility (inverse sentiment)
            volatility = data['Close'].pct_change().std()
            sentiment_indicators['volatility'] = -np.tanh(volatility * 50)
            
            # Fear & Greed approximation
            fear_greed = (sentiment_indicators['momentum'] - sentiment_indicators['volatility']) / 2
            sentiment_indicators['fear_greed'] = np.clip(fear_greed, -1, 1)
            
            return sentiment_indicators
            
        except Exception as e:
            print(f"❌ Error fetching market sentiment for {asset}: {e}")
            return {'fear_greed': 0.0, 'momentum': 0.0, 'volume': 0.0}
    
    def _fetch_coingecko_sentiment(self, asset: str) -> Dict[str, float]:
        """Fetch sentiment from CoinGecko API"""
        try:
            # Map asset to CoinGecko ID
            cg_id_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'SUI': 'sui',
                'SOL': 'solana',
                'SEI': 'sei-network'
            }
            
            if asset not in cg_id_map:
                return {'fear_greed': 0.0, 'momentum': 0.0, 'volume': 0.0}
            
            coin_id = cg_id_map[asset]
            
            # Get coin data
            coin_data = self.cg.get_coin_by_id(coin_id)
            
            # Extract sentiment indicators
            sentiment_indicators = {}
            
            # Price change sentiment
            price_change_24h = coin_data.get('market_data', {}).get('price_change_percentage_24h', 0)
            sentiment_indicators['momentum'] = np.tanh(price_change_24h / 10)
            
            # Volume sentiment
            volume_24h = coin_data.get('market_data', {}).get('total_volume', {}).get('usd', 0)
            market_cap = coin_data.get('market_data', {}).get('market_cap', {}).get('usd', 1)
            volume_ratio = volume_24h / market_cap if market_cap > 0 else 0
            sentiment_indicators['volume'] = np.tanh(volume_ratio * 100)
            
            # Sentiment score (if available)
            sentiment_score = coin_data.get('sentiment_votes_up_percentage', 50)
            sentiment_indicators['fear_greed'] = (sentiment_score - 50) / 50
            
            return sentiment_indicators
            
        except Exception as e:
            print(f"❌ Error fetching CoinGecko sentiment: {e}")
            return {'fear_greed': 0.0, 'momentum': 0.0, 'volume': 0.0}
    
    def generate_sentiment_signal(self, asset: str, hours_back: int = 6) -> SentimentSignal:
        """Generate comprehensive sentiment signal for asset"""
        print(f"🎯 Generating sentiment signal for {asset}...")
        
        # Collect all sentiment data
        news_articles = [a for a in self.news_data if asset in a.asset_mentions]
        social_posts = [p for p in self.social_data if asset in p.asset_mentions]
        market_sentiment = self.fetch_market_sentiment(asset)
        
        # Calculate weighted sentiment scores
        news_sentiment = self._calculate_news_sentiment(news_articles, hours_back)
        social_sentiment = self._calculate_social_sentiment(social_posts, hours_back)
        market_sentiment_score = market_sentiment.get('fear_greed', 0)
        
        # Combine sentiments with weights
        weights = self.config['sentiment_weights']
        combined_sentiment = (
            news_sentiment * weights['news'] +
            social_sentiment * weights['social'] +
            market_sentiment_score * weights['market']
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(news_articles, social_posts, market_sentiment)
        
        # Determine sentiment label
        if combined_sentiment > self.config['sentiment_threshold']:
            sentiment_label = 'BULLISH'
        elif combined_sentiment < -self.config['sentiment_threshold']:
            sentiment_label = 'BEARISH'
        else:
            sentiment_label = 'NEUTRAL'
        
        # Calculate momentum
        momentum = self._calculate_sentiment_momentum(asset, hours_back)
        
        # Extract key phrases
        key_phrases = self._extract_key_phrases(news_articles, social_posts)
        
        # Count mentions
        total_mentions = len(news_articles) + len(social_posts)
        
        signal = SentimentSignal(
            asset=asset,
            sentiment_score=combined_sentiment,
            sentiment_label=sentiment_label,
            confidence=confidence,
            sources=['news', 'social', 'market'],
            timestamp=datetime.now(),
            key_phrases=key_phrases,
            volume=total_mentions,
            momentum=momentum,
            supporting_data={
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'market_sentiment': market_sentiment_score,
                'news_count': len(news_articles),
                'social_count': len(social_posts),
                'market_indicators': market_sentiment
            }
        )
        
        # Store in history
        if asset not in self.sentiment_history:
            self.sentiment_history[asset] = []
        self.sentiment_history[asset].append(signal)
        
        return signal
    
    def _calculate_news_sentiment(self, articles: List[NewsArticle], hours_back: int) -> float:
        """Calculate weighted news sentiment"""
        if not articles:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_articles = [a for a in articles if a.published_date >= cutoff_time]
        
        if not recent_articles:
            return 0.0
        
        # Weight by recency and relevance
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        for article in recent_articles:
            # Time decay weight
            time_weight = 1.0 - (datetime.now() - article.published_date).seconds / (hours_back * 3600)
            
            # Relevance weight
            relevance_weight = article.relevance_score
            
            # Combined weight
            weight = time_weight * relevance_weight
            
            weighted_sentiment += article.sentiment_score * weight
            total_weight += weight
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    def _calculate_social_sentiment(self, posts: List[SocialPost], hours_back: int) -> float:
        """Calculate weighted social sentiment"""
        if not posts:
            return 0.0
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_posts = [p for p in posts if p.timestamp >= cutoff_time]
        
        if not recent_posts:
            return 0.0
        
        # Weight by engagement and recency
        weighted_sentiment = 0.0
        total_weight = 0.0
        
        for post in recent_posts:
            # Time decay weight
            time_weight = 1.0 - (datetime.now() - post.timestamp).seconds / (hours_back * 3600)
            
            # Engagement weight
            engagement_weight = min(post.engagement_score / 10, 1.0)
            
            # Combined weight
            weight = time_weight * (1 + engagement_weight)
            
            weighted_sentiment += post.sentiment_score * weight
            total_weight += weight
        
        return weighted_sentiment / total_weight if total_weight > 0 else 0.0
    
    def _calculate_confidence(self, articles: List[NewsArticle], posts: List[SocialPost], 
                            market_sentiment: Dict[str, float]) -> float:
        """Calculate confidence score for sentiment signal"""
        # Base confidence from data volume
        total_sources = len(articles) + len(posts)
        volume_confidence = min(total_sources / 20, 1.0)  # Max at 20 sources
        
        # Consistency confidence (how aligned are the sentiments)
        sentiments = []
        
        for article in articles:
            sentiments.append(article.sentiment_score)
        
        for post in posts:
            sentiments.append(post.sentiment_score)
        
        if market_sentiment:
            sentiments.append(market_sentiment.get('fear_greed', 0))
        
        if len(sentiments) < 2:
            consistency_confidence = 0.5
        else:
            # Calculate standard deviation (lower = more consistent)
            std_dev = np.std(sentiments)
            consistency_confidence = max(0, 1 - std_dev)
        
        # Recency confidence
        if articles:
            avg_age_hours = sum(
                (datetime.now() - a.published_date).seconds / 3600 
                for a in articles
            ) / len(articles)
            recency_confidence = max(0, 1 - avg_age_hours / 24)
        else:
            recency_confidence = 0.5
        
        # Combined confidence
        confidence = (
            volume_confidence * 0.4 +
            consistency_confidence * 0.4 +
            recency_confidence * 0.2
        )
        
        return np.clip(confidence, 0, 1)
    
    def _calculate_sentiment_momentum(self, asset: str, hours_back: int) -> float:
        """Calculate sentiment momentum (rate of change)"""
        if asset not in self.sentiment_history or len(self.sentiment_history[asset]) < 2:
            return 0.0
        
        # Get recent sentiment signals
        recent_signals = [
            s for s in self.sentiment_history[asset] 
            if s.timestamp >= datetime.now() - timedelta(hours=hours_back)
        ]
        
        if len(recent_signals) < 2:
            return 0.0
        
        # Calculate momentum as rate of change
        recent_signals.sort(key=lambda x: x.timestamp)
        
        old_sentiment = recent_signals[0].sentiment_score
        new_sentiment = recent_signals[-1].sentiment_score
        
        momentum = new_sentiment - old_sentiment
        
        return np.clip(momentum, -1, 1)
    
    def _extract_key_phrases(self, articles: List[NewsArticle], posts: List[SocialPost]) -> List[str]:
        """Extract key phrases from sentiment data"""
        # Combine all text
        all_text = []
        
        for article in articles:
            all_text.append(article.title + " " + article.content)
        
        for post in posts:
            all_text.append(post.content)
        
        if not all_text:
            return []
        
        try:
            # Use TF-IDF to find key phrases
            vectorizer = TfidfVectorizer(
                max_features=20,
                ngram_range=(1, 3),
                stop_words='english'
            )
            
            tfidf_matrix = vectorizer.fit_transform(all_text)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top features
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-10:][::-1]
            
            key_phrases = [feature_names[i] for i in top_indices]
            
            return key_phrases
            
        except Exception as e:
            print(f"⚠️  Error extracting key phrases: {e}")
            return []
    
    def start_real_time_monitoring(self):
        """Start real-time sentiment monitoring"""
        if self.monitoring_active:
            print("⚠️  Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        print("🚀 Real-time sentiment monitoring started")
    
    def stop_real_time_monitoring(self):
        """Stop real-time sentiment monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        print("⏹️  Real-time sentiment monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Fetch fresh data
                self.fetch_news_sentiment(hours_back=1)
                self.fetch_social_sentiment(hours_back=1)
                
                # Generate signals for all assets
                for asset in self.config['assets']:
                    signal = self.generate_sentiment_signal(asset, hours_back=1)
                    
                    # Check for significant sentiment changes
                    if signal.confidence > self.config['confidence_threshold']:
                        if abs(signal.sentiment_score) > self.config['sentiment_threshold']:
                            print(f"🚨 SENTIMENT ALERT: {asset} - {signal.sentiment_label} "
                                  f"(Score: {signal.sentiment_score:.3f}, Confidence: {signal.confidence:.3f})")
                
                # Wait before next update
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def get_sentiment_dashboard(self, asset: str) -> Dict[str, Any]:
        """Generate sentiment dashboard data"""
        if asset not in self.sentiment_history:
            return {}
        
        recent_signals = [
            s for s in self.sentiment_history[asset] 
            if s.timestamp >= datetime.now() - timedelta(hours=24)
        ]
        
        if not recent_signals:
            return {}
        
        # Calculate trends
        sentiment_trend = [s.sentiment_score for s in recent_signals]
        confidence_trend = [s.confidence for s in recent_signals]
        volume_trend = [s.volume for s in recent_signals]
        
        # Current signal
        current_signal = recent_signals[-1]
        
        dashboard = {
            'asset': asset,
            'current_sentiment': {
                'score': current_signal.sentiment_score,
                'label': current_signal.sentiment_label,
                'confidence': current_signal.confidence,
                'momentum': current_signal.momentum
            },
            'trends': {
                'sentiment': sentiment_trend,
                'confidence': confidence_trend,
                'volume': volume_trend,
                'timestamps': [s.timestamp.isoformat() for s in recent_signals]
            },
            'key_phrases': current_signal.key_phrases,
            'sources': current_signal.supporting_data,
            'analysis': {
                'trend_direction': 'UP' if sentiment_trend[-1] > sentiment_trend[0] else 'DOWN',
                'volatility': np.std(sentiment_trend),
                'average_confidence': np.mean(confidence_trend),
                'total_mentions': sum(volume_trend)
            }
        }
        
        return dashboard
    
    def generate_sentiment_report(self, asset: str, hours_back: int = 24) -> str:
        """Generate comprehensive sentiment report"""
        signal = self.generate_sentiment_signal(asset, hours_back)
        dashboard = self.get_sentiment_dashboard(asset)
        
        report = f"""
📊 SENTIMENT ANALYSIS REPORT
============================
Asset: {asset}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Time Period: Last {hours_back} hours

🎯 CURRENT SENTIMENT
Sentiment Score: {signal.sentiment_score:.3f}
Sentiment Label: {signal.sentiment_label}
Confidence: {signal.confidence:.3f}
Momentum: {signal.momentum:.3f}

📈 SENTIMENT BREAKDOWN
News Sentiment: {signal.supporting_data.get('news_sentiment', 0):.3f}
Social Sentiment: {signal.supporting_data.get('social_sentiment', 0):.3f}
Market Sentiment: {signal.supporting_data.get('market_sentiment', 0):.3f}

📊 DATA SOURCES
News Articles: {signal.supporting_data.get('news_count', 0)}
Social Posts: {signal.supporting_data.get('social_count', 0)}
Total Mentions: {signal.volume}

🔑 KEY PHRASES
{', '.join(signal.key_phrases[:10])}

💡 INTERPRETATION
"""
        
        if signal.sentiment_label == 'BULLISH':
            report += f"Strong positive sentiment detected with {signal.confidence:.1%} confidence.\n"
            report += "Market participants are showing optimism.\n"
        elif signal.sentiment_label == 'BEARISH':
            report += f"Strong negative sentiment detected with {signal.confidence:.1%} confidence.\n"
            report += "Market participants are showing pessimism.\n"
        else:
            report += f"Neutral sentiment with {signal.confidence:.1%} confidence.\n"
            report += "Market sentiment is mixed or unclear.\n"
        
        if signal.momentum > 0.1:
            report += "Sentiment momentum is positive - bullish trend building.\n"
        elif signal.momentum < -0.1:
            report += "Sentiment momentum is negative - bearish trend building.\n"
        else:
            report += "Sentiment momentum is neutral - no clear trend.\n"
        
        report += f"""
⚠️  TRADING CONSIDERATIONS
• High confidence signals (>70%) are more reliable
• Consider sentiment alongside technical analysis
• Monitor momentum changes for trend reversals
• Be aware of sentiment manipulation and fake news

🔄 RECOMMENDATION
"""
        
        if signal.confidence > 0.7:
            if signal.sentiment_label == 'BULLISH':
                report += "Consider LONG position with proper risk management"
            elif signal.sentiment_label == 'BEARISH':
                report += "Consider SHORT position with proper risk management"
            else:
                report += "NEUTRAL - wait for clearer signals"
        else:
            report += "LOW CONFIDENCE - avoid trading on sentiment alone"
        
        return report

def main():
    """Demo function for testing sentiment analysis"""
    print("📊 SENTIMENT ANALYSIS ENGINE DEMO")
    print("=" * 60)
    
    # Initialize engine
    engine = SentimentAnalysisEngine()
    
    # Test assets
    test_assets = ['BTC', 'ETH', 'SUI']
    
    for asset in test_assets:
        print(f"\n🎯 Analyzing sentiment for {asset}...")
        
        # Fetch data
        engine.fetch_news_sentiment(hours_back=12)
        engine.fetch_social_sentiment(hours_back=12)
        
        # Generate signal
        signal = engine.generate_sentiment_signal(asset, hours_back=6)
        
        print(f"✅ Signal generated:")
        print(f"   Sentiment: {signal.sentiment_label}")
        print(f"   Score: {signal.sentiment_score:.3f}")
        print(f"   Confidence: {signal.confidence:.3f}")
        print(f"   Momentum: {signal.momentum:.3f}")
        print(f"   Sources: {signal.volume} mentions")
        
        # Generate report
        report = engine.generate_sentiment_report(asset, hours_back=6)
        print(f"\n📋 Report generated ({len(report)} characters)")
        
        # Get dashboard
        dashboard = engine.get_sentiment_dashboard(asset)
        if dashboard:
            print(f"📊 Dashboard: {len(dashboard)} metrics available")
    
    print("\n" + "=" * 60)
    print("🎯 Sentiment Analysis Demo Complete!")

if __name__ == "__main__":
    main() 