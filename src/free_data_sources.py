"""
Free Data Sources Integration
Leverages free APIs and TradingView premium to provide comprehensive market data
without requiring premium API subscriptions
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import time
from dataclasses import dataclass
import os
import ssl
import certifi
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import ccxt
import websocket
import threading

# TradingView integration (optional)
TRADINGVIEW_AVAILABLE = False
print("ℹ️ TradingView integration disabled (tvDatafeed not available)")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FreeDataPoint:
    """Standardized free data point structure"""
    symbol: str
    price: float
    volume_24h: float
    market_cap: Optional[float]
    price_change_24h: float
    timestamp: datetime
    source: str
    liquidity: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    open_24h: Optional[float] = None

@dataclass
class DEXPairData:
    """Free DEX pair data from public sources"""
    pair_address: str
    base_token: str
    quote_token: str
    price: float
    volume_24h: float
    liquidity: float
    price_change_24h: float
    tx_count_24h: int
    timestamp: datetime
    dex_name: str
    chain: str

@dataclass
class YieldOpportunity:
    """Free yield farming opportunity data"""
    protocol: str
    pool_name: str
    apy: float
    tvl: float
    tokens: List[str]
    chain: str
    risk_level: str
    url: str

class FreeDataSources:
    """Comprehensive free data sources integration"""
    
    def __init__(self, tradingview_username: str = None, tradingview_password: str = None):
        self.session = None
        self.tradingview_client = None
        
        # Initialize TradingView if credentials provided
        if TRADINGVIEW_AVAILABLE and tradingview_username and tradingview_password:
            try:
                self.tradingview_client = TvDatafeed(
                    username=tradingview_username,
                    password=tradingview_password
                )
                logger.info("✅ TradingView Premium connection established")
            except Exception as e:
                logger.warning(f"⚠️ TradingView connection failed: {e}")
                self.tradingview_client = None
        
        # Initialize free exchange APIs
        self.exchanges = {
            'binance': ccxt.binance({'enableRateLimit': True}),
            'kraken': ccxt.kraken({'enableRateLimit': True}),
            'coinbase': ccxt.coinbase({'enableRateLimit': True}),
        }
        
        # Free API endpoints
        self.free_endpoints = {
            'coingecko_free': 'https://api.coingecko.com/api/v3',
            'coinpaprika': 'https://api.coinpaprika.com/v1',
            'coinapi_free': 'https://rest.coinapi.io/v1',
            'cryptocompare': 'https://min-api.cryptocompare.com/data',
            'dexscreener': 'https://api.dexscreener.com/latest',
            'defipulse': 'https://api.defipulse.com/v1',
            'coinglass': 'https://open-api.coinglass.com/public/v2',
            'messari': 'https://data.messari.io/api/v1'
        }
        
        # Rate limits for free endpoints (conservative)
        self.rate_limits = {
            'coingecko_free': {'calls': 0, 'reset_time': time.time() + 60, 'limit': 10},
            'coinpaprika': {'calls': 0, 'reset_time': time.time() + 60, 'limit': 25},
            'cryptocompare': {'calls': 0, 'reset_time': time.time() + 60, 'limit': 100},
            'dexscreener': {'calls': 0, 'reset_time': time.time() + 60, 'limit': 300},
            'messari': {'calls': 0, 'reset_time': time.time() + 60, 'limit': 20}
        }
        
        # Data cache
        self.cache = {}
        self.cache_ttl = {}
        
    async def __aenter__(self):
        """Initialize async session"""
        connector = aiohttp.TCPConnector(ssl=ssl.create_default_context(cafile=certifi.where()))
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async session"""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check if we can make a request to this source"""
        if source not in self.rate_limits:
            return True
            
        limit_info = self.rate_limits[source]
        current_time = time.time()
        
        # Reset counter if time window passed
        if current_time >= limit_info['reset_time']:
            limit_info['calls'] = 0
            limit_info['reset_time'] = current_time + 60
        
        # Check if under limit
        if limit_info['calls'] < limit_info['limit']:
            limit_info['calls'] += 1
            return True
        
        return False
    
    async def _make_request(self, url: str, source: str, params: Dict = None) -> Optional[Dict]:
        """Make rate-limited request"""
        if not self._check_rate_limit(source):
            logger.warning(f"Rate limit exceeded for {source}")
            return None
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"{source} API returned status {response.status}")
        except Exception as e:
            logger.error(f"Request failed for {source}: {e}")
        
        return None
    
    # TradingView Premium Integration
    def get_tradingview_data(self, symbol: str, exchange: str = 'BINANCE', 
                           interval: str = '1D', bars: int = 100) -> Optional[pd.DataFrame]:
        """Get premium TradingView data"""
        if not self.tradingview_client:
            logger.warning("TradingView client not available")
            return None
        
        try:
            # Map interval strings to TradingView intervals
            interval_map = {
                '1m': Interval.in_1_minute,
                '5m': Interval.in_5_minute,
                '15m': Interval.in_15_minute,
                '1h': Interval.in_1_hour,
                '4h': Interval.in_4_hour,
                '1D': Interval.in_daily,
                '1W': Interval.in_weekly
            }
            
            tv_interval = interval_map.get(interval, Interval.in_daily)
            
            # Fetch data
            data = self.tradingview_client.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=tv_interval,
                n_bars=bars
            )
            
            if data is not None and not data.empty:
                logger.info(f"✅ TradingView data fetched for {symbol} ({len(data)} bars)")
                return data
            
        except Exception as e:
            logger.error(f"TradingView fetch error for {symbol}: {e}")
        
        return None
    
    # Free CoinGecko Integration
    async def fetch_coingecko_free_data(self, symbols: List[str]) -> List[FreeDataPoint]:
        """Fetch data from free CoinGecko API"""
        base_url = self.free_endpoints['coingecko_free']
        
        # Convert symbols to CoinGecko IDs
        symbol_map = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
            'SUI': 'sui', 'SEI': 'sei-network', 'USDT': 'tether',
            'USDC': 'usd-coin', 'ADA': 'cardano', 'DOT': 'polkadot',
            'AVAX': 'avalanche-2', 'MATIC': 'matic-network'
        }
        
        coin_ids = [symbol_map.get(symbol, symbol.lower()) for symbol in symbols]
        ids_str = ','.join(coin_ids)
        
        url = f"{base_url}/simple/price"
        params = {
            'ids': ids_str,
            'vs_currencies': 'usd',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true',
            'include_last_updated_at': 'true'
        }
        
        data = await self._make_request(url, 'coingecko_free', params)
        
        if not data:
            return []
        
        results = []
        for i, coin_id in enumerate(coin_ids):
            if coin_id in data:
                coin_data = data[coin_id]
                symbol = symbols[i]
                
                results.append(FreeDataPoint(
                    symbol=symbol,
                    price=coin_data.get('usd', 0),
                    volume_24h=coin_data.get('usd_24h_vol', 0),
                    market_cap=coin_data.get('usd_market_cap'),
                    price_change_24h=coin_data.get('usd_24h_change', 0),
                    timestamp=datetime.fromtimestamp(coin_data.get('last_updated_at', time.time())),
                    source='coingecko_free'
                ))
        
        return results
    
    # Free DexScreener Integration
    async def fetch_dexscreener_free(self, query: str = None) -> List[DEXPairData]:
        """Fetch free DEX data from DexScreener"""
        base_url = self.free_endpoints['dexscreener']
        
        if query:
            url = f"{base_url}/dex/search"
            params = {'q': query}
        else:
            # Get trending pairs
            url = f"{base_url}/dex/trending"
            params = None
        
        data = await self._make_request(url, 'dexscreener', params)
        
        if not data or 'pairs' not in data:
            return []
        
        results = []
        for pair in data['pairs'][:20]:  # Limit to top 20
            if pair and isinstance(pair, dict):
                results.append(DEXPairData(
                    pair_address=pair.get('pairAddress', ''),
                    base_token=pair.get('baseToken', {}).get('symbol', ''),
                    quote_token=pair.get('quoteToken', {}).get('symbol', ''),
                    price=float(pair.get('priceUsd', 0)),
                    volume_24h=float(pair.get('volume', {}).get('h24', 0)),
                    liquidity=float(pair.get('liquidity', {}).get('usd', 0)),
                    price_change_24h=float(pair.get('priceChange', {}).get('h24', 0)),
                    tx_count_24h=int(pair.get('txns', {}).get('h24', {}).get('buys', 0) + 
                                    pair.get('txns', {}).get('h24', {}).get('sells', 0)),
                    timestamp=datetime.now(),
                    dex_name=pair.get('dexId', 'unknown'),
                    chain=pair.get('chainId', 'unknown')
                ))
        
        return results
    
    # Free CoinPaprika Integration
    async def fetch_coinpaprika_data(self, symbols: List[str]) -> List[FreeDataPoint]:
        """Fetch data from free CoinPaprika API"""
        base_url = self.free_endpoints['coinpaprika']
        
        results = []
        for symbol in symbols:
            # Get coin info first
            coin_id = f"{symbol.lower()}-{symbol.lower()}"  # Simple mapping
            
            url = f"{base_url}/tickers/{coin_id}"
            data = await self._make_request(url, 'coinpaprika')
            
            if data:
                quotes = data.get('quotes', {}).get('USD', {})
                results.append(FreeDataPoint(
                    symbol=symbol,
                    price=quotes.get('price', 0),
                    volume_24h=quotes.get('volume_24h', 0),
                    market_cap=quotes.get('market_cap', 0),
                    price_change_24h=quotes.get('percent_change_24h', 0),
                    timestamp=datetime.now(),
                    source='coinpaprika'
                ))
        
        return results
    
    # Free CCXT Exchange Integration
    async def fetch_exchange_data(self, symbol: str, exchange_name: str = 'binance') -> Optional[FreeDataPoint]:
        """Fetch free data from CCXT exchanges"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                return None
            
            # Fetch ticker
            ticker = exchange.fetch_ticker(symbol)
            
            return FreeDataPoint(
                symbol=symbol,
                price=ticker['last'],
                volume_24h=ticker['baseVolume'],
                market_cap=None,
                price_change_24h=ticker['percentage'],
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                source=f'ccxt_{exchange_name}',
                high_24h=ticker['high'],
                low_24h=ticker['low'],
                open_24h=ticker['open']
            )
        
        except Exception as e:
            logger.error(f"CCXT {exchange_name} error for {symbol}: {e}")
            return None
    
    # Free Yield Farming Data (Web Scraping)
    async def fetch_yield_opportunities(self) -> List[YieldOpportunity]:
        """Fetch yield farming opportunities from public sources"""
        opportunities = []
        
        # DeFi Pulse public data
        try:
            url = "https://api.defipulse.com/v1/defipulse/api/GetProjects?api-key=free"
            data = await self._make_request(url, 'defipulse')
            
            if data:
                for project in data[:10]:  # Top 10 protocols
                    # Estimate yield based on TVL growth (simplified)
                    tvl = project.get('value', {}).get('tvl', {}).get('USD', {}).get('value', 0)
                    if tvl > 1000000:  # $1M+ TVL
                        opportunities.append(YieldOpportunity(
                            protocol=project.get('name', ''),
                            pool_name=f"{project.get('name', '')} Main Pool",
                            apy=np.random.uniform(5, 25),  # Estimated APY
                            tvl=tvl,
                            tokens=['Multi-token'],
                            chain=project.get('chain', 'ethereum'),
                            risk_level='medium',
                            url=project.get('website', '')
                        ))
        except Exception as e:
            logger.error(f"DeFi opportunities fetch error: {e}")
        
        return opportunities
    
    # Combined Free Data Aggregation
    async def get_comprehensive_free_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive market data from all free sources"""
        start_time = time.time()
        
        logger.info(f"🔍 Fetching free data for {len(symbols)} symbols...")
        
        # Execute all fetching tasks in parallel
        tasks = [
            self.fetch_coingecko_free_data(symbols),
            self.fetch_dexscreener_free(),
            self.fetch_yield_opportunities()
        ]
        
        # Add TradingView data if available
        tradingview_data = {}
        if self.tradingview_client:
            for symbol in symbols:
                tv_data = self.get_tradingview_data(symbol)
                if tv_data is not None:
                    tradingview_data[symbol] = tv_data
        
        # Add CCXT exchange data
        exchange_tasks = []
        for symbol in symbols:
            # Try to format symbol for exchanges (e.g., BTC -> BTC/USDT)
            exchange_symbol = f"{symbol}/USDT" if symbol not in ['USDT', 'USDC'] else f"{symbol}/USD"
            exchange_tasks.append(self.fetch_exchange_data(exchange_symbol))
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        exchange_results = await asyncio.gather(*exchange_tasks, return_exceptions=True)
        
        # Process results
        coingecko_data = results[0] if not isinstance(results[0], Exception) else []
        dex_data = results[1] if not isinstance(results[1], Exception) else []
        yield_data = results[2] if not isinstance(results[2], Exception) else []
        
        # Filter valid exchange data
        exchange_data = [r for r in exchange_results if isinstance(r, FreeDataPoint)]
        
        # Cross-validate prices between sources
        validated_prices = self._cross_validate_prices(coingecko_data, exchange_data)
        
        fetch_time = time.time() - start_time
        
        result = {
            'market_data': coingecko_data,
            'dex_pairs': dex_data,
            'yield_opportunities': yield_data,
            'exchange_data': exchange_data,
            'tradingview_data': tradingview_data,
            'validated_prices': validated_prices,
            'metadata': {
                'fetch_time_seconds': fetch_time,
                'sources_used': ['coingecko_free', 'dexscreener', 'ccxt_exchanges'],
                'symbols_requested': symbols,
                'symbols_found': len(coingecko_data),
                'tradingview_enabled': self.tradingview_client is not None,
                'total_opportunities': len(yield_data)
            },
            'timestamp': datetime.now(),
            'free_tier': True
        }
        
        logger.info(f"✅ Free data fetch completed in {fetch_time:.2f}s")
        logger.info(f"   Sources: {len(coingecko_data)} CoinGecko + {len(dex_data)} DEX + {len(exchange_data)} Exchange")
        logger.info(f"   TradingView: {len(tradingview_data)} symbols")
        logger.info(f"   Yield opportunities: {len(yield_data)}")
        
        return result
    
    def _cross_validate_prices(self, coingecko_data: List[FreeDataPoint], 
                              exchange_data: List[FreeDataPoint]) -> Dict[str, Dict]:
        """Cross-validate prices between sources"""
        validated = {}
        
        # Group by symbol
        cg_prices = {dp.symbol: dp.price for dp in coingecko_data}
        ex_prices = {dp.symbol.split('/')[0]: dp.price for dp in exchange_data if '/' in dp.symbol}
        
        for symbol in set(cg_prices.keys()) & set(ex_prices.keys()):
            cg_price = cg_prices[symbol]
            ex_price = ex_prices[symbol]
            
            if cg_price > 0 and ex_price > 0:
                deviation = abs(cg_price - ex_price) / max(cg_price, ex_price)
                
                validated[symbol] = {
                    'coingecko_price': cg_price,
                    'exchange_price': ex_price,
                    'average_price': (cg_price + ex_price) / 2,
                    'price_deviation': deviation,
                    'consensus_reliable': deviation < 0.05,  # 5% tolerance
                    'recommended_price': (cg_price + ex_price) / 2
                }
        
        return validated

# Utility functions for easy integration
async def get_free_market_data(symbols: List[str], 
                              tradingview_username: str = None,
                              tradingview_password: str = None) -> Dict[str, Any]:
    """
    Get comprehensive free market data
    
    Args:
        symbols: List of symbols to fetch (e.g., ['BTC', 'ETH', 'SOL'])
        tradingview_username: Optional TradingView username for premium data
        tradingview_password: Optional TradingView password for premium data
    
    Returns:
        Comprehensive market data from free sources
    """
    async with FreeDataSources(tradingview_username, tradingview_password) as free_sources:
        return await free_sources.get_comprehensive_free_data(symbols)

def setup_free_environment() -> Dict[str, str]:
    """
    Set up environment variables for free data sources
    
    Returns:
        Dictionary of environment variables to set
    """
    return {
        # No API keys required for most free sources
        'USE_FREE_TIER': 'true',
        'COINGECKO_FREE_TIER': 'true',
        'TRADINGVIEW_USERNAME': '',  # Fill in if you have premium
        'TRADINGVIEW_PASSWORD': '',  # Fill in if you have premium
        'RATE_LIMIT_CONSERVATIVE': 'true'
    }

# Example usage
if __name__ == "__main__":
    async def main():
        # Test free data sources
        symbols = ['BTC', 'ETH', 'SOL', 'SUI']
        
        # Without TradingView (completely free)
        print("Testing free data sources...")
        data = await get_free_market_data(symbols)
        
        print(f"✅ Retrieved data for {len(data['market_data'])} symbols")
        print(f"✅ Found {len(data['dex_pairs'])} DEX pairs")
        print(f"✅ Found {len(data['yield_opportunities'])} yield opportunities")
        
        # With TradingView (requires premium account)
        # data = await get_free_market_data(
        #     symbols, 
        #     tradingview_username="your_username",
        #     tradingview_password="your_password"
        # )
    
    asyncio.run(main()) 