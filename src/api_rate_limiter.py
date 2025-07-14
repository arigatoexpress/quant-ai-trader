"""
Advanced API Rate Limiting System
Ensures all external API calls respect rate limits and handle failures gracefully
"""

import time
import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from collections import deque
import json
import os

@dataclass
class RateLimit:
    """Rate limit configuration for an API"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Max requests in burst
    cooldown_seconds: int = 1  # Minimum time between requests
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    max_retries: int = 3

@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    name: str
    base_url: str
    rate_limit: RateLimit
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 10
    priority: int = 1  # 1=high, 2=medium, 3=low

class APIRateLimiter:
    """Intelligent API rate limiter with fallback and caching"""
    
    def __init__(self):
        self.endpoints: Dict[str, APIEndpoint] = {}
        self.request_history: Dict[str, deque] = {}
        self.retry_counts: Dict[str, int] = {}
        self.locks: Dict[str, threading.Lock] = {}
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        
        # Configure default endpoints
        self._setup_default_endpoints()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def _setup_default_endpoints(self):
        """Setup default API endpoints with appropriate rate limits"""
        
        # CoinGecko API (Free tier: 10-50 calls/minute)
        coingecko_rate_limit = RateLimit(
            requests_per_minute=30,  # Conservative for free tier
            requests_per_hour=1800,
            requests_per_day=10000,
            burst_limit=5,
            cooldown_seconds=2,
            backoff_factor=2.0,
            max_retries=3
        )
        
        self.add_endpoint(APIEndpoint(
            name="coingecko",
            base_url="https://api.coingecko.com/api/v3",
            rate_limit=coingecko_rate_limit,
            api_key=os.getenv("COINGECKO_API_KEY"),
            headers={"Accept": "application/json"},
            priority=1
        ))
        
        # Yahoo Finance (More permissive but still limited)
        yahoo_rate_limit = RateLimit(
            requests_per_minute=60,
            requests_per_hour=2000,
            requests_per_day=10000,
            burst_limit=10,
            cooldown_seconds=1,
            backoff_factor=1.5,
            max_retries=2
        )
        
        self.add_endpoint(APIEndpoint(
            name="yahoo_finance",
            base_url="https://query1.finance.yahoo.com/v8",
            rate_limit=yahoo_rate_limit,
            priority=2
        ))
        
        # NewsAPI (Free tier: 1000 requests/day)
        news_rate_limit = RateLimit(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000,
            burst_limit=3,
            cooldown_seconds=6,
            backoff_factor=3.0,
            max_retries=2
        )
        
        self.add_endpoint(APIEndpoint(
            name="newsapi",
            base_url="https://newsapi.org/v2",
            rate_limit=news_rate_limit,
            api_key=os.getenv("NEWS_API_KEY"),
            headers={"User-Agent": "TradingBot/1.0"},
            priority=3
        ))
        
        # X/Twitter API (Rate limits vary by endpoint)
        twitter_rate_limit = RateLimit(
            requests_per_minute=15,  # Conservative for most endpoints
            requests_per_hour=300,
            requests_per_day=2000,
            burst_limit=2,
            cooldown_seconds=4,
            backoff_factor=2.5,
            max_retries=2
        )
        
        self.add_endpoint(APIEndpoint(
            name="twitter",
            base_url="https://api.twitter.com/2",
            rate_limit=twitter_rate_limit,
            api_key=os.getenv("TWITTER_BEARER_TOKEN"),
            headers={
                "Authorization": f"Bearer {os.getenv('TWITTER_BEARER_TOKEN')}",
                "Content-Type": "application/json"
            },
            priority=3
        ))
        
        # Blockchain RPCs (Generally more permissive)
        rpc_rate_limit = RateLimit(
            requests_per_minute=120,
            requests_per_hour=5000,
            requests_per_day=50000,
            burst_limit=20,
            cooldown_seconds=0.5,
            backoff_factor=1.5,
            max_retries=3
        )
        
        # SUI RPC
        self.add_endpoint(APIEndpoint(
            name="sui_rpc",
            base_url="https://fullnode.mainnet.sui.io",
            rate_limit=rpc_rate_limit,
            headers={"Content-Type": "application/json"},
            priority=1
        ))
        
        # Solana RPC
        self.add_endpoint(APIEndpoint(
            name="solana_rpc",
            base_url="https://api.mainnet-beta.solana.com",
            rate_limit=rpc_rate_limit,
            headers={"Content-Type": "application/json"},
            priority=1
        ))
        
    def add_endpoint(self, endpoint: APIEndpoint):
        """Add an API endpoint to the rate limiter"""
        self.endpoints[endpoint.name] = endpoint
        self.request_history[endpoint.name] = deque()
        self.retry_counts[endpoint.name] = 0
        self.locks[endpoint.name] = threading.Lock()
        
        self.logger.info(f"Added API endpoint: {endpoint.name}")
        
    def _check_rate_limit(self, endpoint_name: str) -> bool:
        """Check if we can make a request without exceeding rate limits"""
        if endpoint_name not in self.endpoints:
            return False
            
        endpoint = self.endpoints[endpoint_name]
        rate_limit = endpoint.rate_limit
        history = self.request_history[endpoint_name]
        
        now = time.time()
        
        # Clean old requests from history
        minute_ago = now - 60
        hour_ago = now - 3600
        day_ago = now - 86400
        
        # Remove requests older than a day
        while history and history[0] < day_ago:
            history.popleft()
        
        # Count requests in different time windows
        requests_last_minute = sum(1 for t in history if t > minute_ago)
        requests_last_hour = sum(1 for t in history if t > hour_ago)
        requests_last_day = len(history)
        
        # Check against limits
        if requests_last_minute >= rate_limit.requests_per_minute:
            return False
        if requests_last_hour >= rate_limit.requests_per_hour:
            return False
        if requests_last_day >= rate_limit.requests_per_day:
            return False
            
        return True
    
    def _wait_for_rate_limit(self, endpoint_name: str) -> float:
        """Calculate how long to wait before next request"""
        if endpoint_name not in self.endpoints:
            return 0
            
        endpoint = self.endpoints[endpoint_name]
        rate_limit = endpoint.rate_limit
        history = self.request_history[endpoint_name]
        
        if not history:
            return rate_limit.cooldown_seconds
            
        now = time.time()
        last_request = history[-1] if history else 0
        
        # Ensure minimum cooldown
        cooldown_wait = max(0, rate_limit.cooldown_seconds - (now - last_request))
        
        # Check if we need to wait for rate limit windows
        minute_ago = now - 60
        requests_last_minute = sum(1 for t in history if t > minute_ago)
        
        if requests_last_minute >= rate_limit.requests_per_minute:
            # Wait until the oldest request in the last minute expires
            oldest_in_minute = min(t for t in history if t > minute_ago)
            minute_wait = max(0, 60 - (now - oldest_in_minute))
            return max(cooldown_wait, minute_wait)
            
        return cooldown_wait
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if available and not expired"""
        if cache_key not in self.cache:
            return None
            
        if cache_key in self.cache_ttl:
            if datetime.now() > self.cache_ttl[cache_key]:
                # Cache expired
                del self.cache[cache_key]
                del self.cache_ttl[cache_key]
                return None
                
        return self.cache[cache_key]
    
    def _set_cache(self, cache_key: str, data: Any, ttl_minutes: int = 5):
        """Set data in cache with TTL"""
        self.cache[cache_key] = data
        self.cache_ttl[cache_key] = datetime.now() + timedelta(minutes=ttl_minutes)
    
    async def make_request(self, endpoint_name: str, url_path: str, params: Dict = None, 
                          cache_ttl_minutes: int = 5, priority_override: int = None) -> Optional[Dict]:
        """Make a rate-limited API request with caching and fallback"""
        
        if endpoint_name not in self.endpoints:
            self.logger.error(f"Unknown endpoint: {endpoint_name}")
            return None
            
        # Check cache first
        cache_key = f"{endpoint_name}_{url_path}_{json.dumps(params or {}, sort_keys=True)}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            self.logger.debug(f"Cache hit for {endpoint_name}: {url_path}")
            return cached_data
        
        endpoint = self.endpoints[endpoint_name]
        
        # Acquire lock for this endpoint
        with self.locks[endpoint_name]:
            # Check rate limit
            if not self._check_rate_limit(endpoint_name):
                wait_time = self._wait_for_rate_limit(endpoint_name)
                if wait_time > 0:
                    self.logger.info(f"Rate limit hit for {endpoint_name}, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            # Record request time
            self.request_history[endpoint_name].append(time.time())
            
        # Make the request
        try:
            import aiohttp
            
            full_url = f"{endpoint.base_url}{url_path}"
            headers = endpoint.headers.copy()
            
            if endpoint.api_key and "Authorization" not in headers:
                if endpoint_name == "coingecko" and endpoint.api_key:
                    headers["x-cg-demo-api-key"] = endpoint.api_key
                elif endpoint_name == "newsapi":
                    headers["X-Api-Key"] = endpoint.api_key
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    full_url, 
                    params=params, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=endpoint.timeout)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Cache successful response
                        self._set_cache(cache_key, data, cache_ttl_minutes)
                        
                        # Reset retry count on success
                        self.retry_counts[endpoint_name] = 0
                        
                        self.logger.debug(f"Successful request to {endpoint_name}: {url_path}")
                        return data
                        
                    elif response.status == 429:  # Rate limited
                        retry_after = response.headers.get("Retry-After", "60")
                        wait_time = int(retry_after)
                        
                        self.logger.warning(f"Rate limited by {endpoint_name}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        
                        # Retry once after rate limit
                        if self.retry_counts[endpoint_name] < endpoint.rate_limit.max_retries:
                            self.retry_counts[endpoint_name] += 1
                            return await self.make_request(endpoint_name, url_path, params, cache_ttl_minutes)
                            
                    else:
                        self.logger.error(f"API error {response.status} for {endpoint_name}: {url_path}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Request failed for {endpoint_name}: {e}")
            
            # Exponential backoff for retries
            if self.retry_counts[endpoint_name] < endpoint.rate_limit.max_retries:
                self.retry_counts[endpoint_name] += 1
                wait_time = endpoint.rate_limit.cooldown_seconds * (endpoint.rate_limit.backoff_factor ** self.retry_counts[endpoint_name])
                
                self.logger.info(f"Retrying {endpoint_name} in {wait_time:.2f}s (attempt {self.retry_counts[endpoint_name]})")
                await asyncio.sleep(wait_time)
                
                return await self.make_request(endpoint_name, url_path, params, cache_ttl_minutes)
            
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all endpoints"""
        status = {}
        
        for name, endpoint in self.endpoints.items():
            history = self.request_history[name]
            now = time.time()
            
            minute_ago = now - 60
            hour_ago = now - 3600
            
            requests_last_minute = sum(1 for t in history if t > minute_ago)
            requests_last_hour = sum(1 for t in history if t > hour_ago)
            
            status[name] = {
                "requests_last_minute": requests_last_minute,
                "requests_last_hour": requests_last_hour,
                "rate_limit_minute": endpoint.rate_limit.requests_per_minute,
                "rate_limit_hour": endpoint.rate_limit.requests_per_hour,
                "utilization_minute": requests_last_minute / endpoint.rate_limit.requests_per_minute,
                "utilization_hour": requests_last_hour / endpoint.rate_limit.requests_per_hour,
                "retry_count": self.retry_counts[name],
                "cache_entries": len([k for k in self.cache.keys() if k.startswith(name)]),
                "can_make_request": self._check_rate_limit(name)
            }
            
        return status
    
    def clear_cache(self, endpoint_name: str = None):
        """Clear cache for specific endpoint or all endpoints"""
        if endpoint_name:
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(endpoint_name)]
            for key in keys_to_remove:
                del self.cache[key]
                if key in self.cache_ttl:
                    del self.cache_ttl[key]
        else:
            self.cache.clear()
            self.cache_ttl.clear()
            
        self.logger.info(f"Cache cleared for {'all endpoints' if not endpoint_name else endpoint_name}")

# Global rate limiter instance
rate_limiter = APIRateLimiter() 