"""
Simple Free Data Sources Integration
Uses only basic free APIs without complex dependencies
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import time
from dataclasses import dataclass
import os
import yfinance as yf
import ccxt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleDataPoint:
    """Simple data point structure"""
    symbol: str
    price: float
    volume_24h: float
    market_cap: Optional[float]
    price_change_24h: float
    timestamp: datetime
    source: str

@dataclass
class DEXData:
    """DEX trading pair data"""
    pair_address: str
    base_token: str
    quote_token: str
    price: float
    volume_24h: float
    liquidity: float
    price_change_24h: float
    dex_name: str

@dataclass
class YieldData:
    """Yield farming opportunity"""
    protocol: str
    pool_name: str
    apy: float
    tvl: float
    tokens: List[str]
    chain: str

class SimpleFreeDataSources:
    """Simple free data sources without complex dependencies"""
    
    def __init__(self):
        self.session = None
        
        # Free API endpoints
        self.endpoints = {
            'coingecko': 'https://api.coingecko.com/api/v3',
            'dexscreener': 'https://api.dexscreener.com/latest',
            'coinpaprika': 'https://api.coinpaprika.com/v1'
        }
        
        # Simple rate limiting
        self.last_request = {}
        self.min_interval = 2  # seconds between requests
        
        # Initialize CCXT exchanges (free)
        try:
            self.exchanges = {
                'binance': ccxt.binance({'enableRateLimit': True, 'sandbox': False}),
                'kraken': ccxt.kraken({'enableRateLimit': True}),
            }
        except Exception as e:
            logger.warning(f"CCXT initialization warning: {e}")
            self.exchanges = {}
    
    async def __aenter__(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async session"""
        if self.session:
            await self.session.close()
    
    def _rate_limit(self, source: str):
        """Simple rate limiting"""
        now = time.time()
        last = self.last_request.get(source, 0)
        
        if now - last < self.min_interval:
            time.sleep(self.min_interval - (now - last))
        
        self.last_request[source] = time.time()
    
    async def _make_request(self, url: str, source: str, params: Dict = None) -> Optional[Dict]:
        """Make rate-limited request"""
        self._rate_limit(source)
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"{source} returned status {response.status}")
        except Exception as e:
            logger.error(f"Request failed for {source}: {e}")
        
        return None
    
    # Free CoinGecko API
    async def get_coingecko_prices(self, symbols: List[str]) -> List[SimpleDataPoint]:
        """Get prices from free CoinGecko API"""
        symbol_map = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
            'SUI': 'sui', 'SEI': 'sei-network', 'USDT': 'tether',
            'USDC': 'usd-coin', 'ADA': 'cardano', 'DOT': 'polkadot'
        }
        
        coin_ids = [symbol_map.get(symbol, symbol.lower()) for symbol in symbols]
        ids_str = ','.join(coin_ids)
        
        url = f"{self.endpoints['coingecko']}/simple/price"
        params = {
            'ids': ids_str,
            'vs_currencies': 'usd',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_24hr_change': 'true'
        }
        
        data = await self._make_request(url, 'coingecko', params)
        
        if not data:
            return []
        
        results = []
        for i, coin_id in enumerate(coin_ids):
            if coin_id in data:
                coin_data = data[coin_id]
                symbol = symbols[i]
                
                results.append(SimpleDataPoint(
                    symbol=symbol,
                    price=coin_data.get('usd', 0),
                    volume_24h=coin_data.get('usd_24h_vol', 0),
                    market_cap=coin_data.get('usd_market_cap'),
                    price_change_24h=coin_data.get('usd_24h_change', 0),
                    timestamp=datetime.now(),
                    source='coingecko_free'
                ))
        
        return results
    
    # Free DexScreener API
    async def get_dexscreener_trending(self) -> List[DEXData]:
        """Get trending pairs from DexScreener"""
        url = f"{self.endpoints['dexscreener']}/dex/trending"
        data = await self._make_request(url, 'dexscreener')
        
        if not data or 'pairs' not in data:
            return []
        
        results = []
        for pair in data['pairs'][:10]:  # Top 10 pairs
            if pair:
                results.append(DEXData(
                    pair_address=pair.get('pairAddress', ''),
                    base_token=pair.get('baseToken', {}).get('symbol', ''),
                    quote_token=pair.get('quoteToken', {}).get('symbol', ''),
                    price=float(pair.get('priceUsd', 0)),
                    volume_24h=float(pair.get('volume', {}).get('h24', 0)),
                    liquidity=float(pair.get('liquidity', {}).get('usd', 0)),
                    price_change_24h=float(pair.get('priceChange', {}).get('h24', 0)),
                    dex_name=pair.get('dexId', 'unknown')
                ))
        
        return results
    
    # Yahoo Finance (free)
    def get_yahoo_finance_data(self, symbols: List[str]) -> List[SimpleDataPoint]:
        """Get data from Yahoo Finance using yfinance"""
        results = []
        
        for symbol in symbols:
            try:
                # Add -USD suffix for crypto symbols
                yahoo_symbol = f"{symbol}-USD" if symbol not in ['USD', 'USDT', 'USDC'] else symbol
                
                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info
                hist = ticker.history(period="2d")
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    price_change = ((current_price - prev_price) / prev_price) * 100
                    
                    results.append(SimpleDataPoint(
                        symbol=symbol,
                        price=float(current_price),
                        volume_24h=float(hist['Volume'].iloc[-1]),
                        market_cap=info.get('marketCap'),
                        price_change_24h=float(price_change),
                        timestamp=datetime.now(),
                        source='yahoo_finance'
                    ))
            
            except Exception as e:
                logger.warning(f"Yahoo Finance error for {symbol}: {e}")
        
        return results
    
    # CCXT Exchange Data (free)
    async def get_exchange_prices(self, symbols: List[str]) -> List[SimpleDataPoint]:
        """Get prices from exchanges via CCXT"""
        results = []
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                markets = exchange.load_markets()
                
                for symbol in symbols:
                    # Try common trading pairs
                    for quote in ['USDT', 'USD', 'USDC']:
                        trading_pair = f"{symbol}/{quote}"
                        
                        if trading_pair in markets:
                            try:
                                ticker = exchange.fetch_ticker(trading_pair)
                                
                                results.append(SimpleDataPoint(
                                    symbol=symbol,
                                    price=float(ticker['last']),
                                    volume_24h=float(ticker['baseVolume'] or 0),
                                    market_cap=None,
                                    price_change_24h=float(ticker['percentage'] or 0),
                                    timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                                    source=f'ccxt_{exchange_name}'
                                ))
                                break  # Found a pair, move to next symbol
                            
                            except Exception as e:
                                logger.warning(f"Error fetching {trading_pair} from {exchange_name}: {e}")
                
            except Exception as e:
                logger.warning(f"Exchange {exchange_name} error: {e}")
        
        return results
    
    # Mock yield opportunities (since we can't access premium DeFi APIs)
    def get_mock_yield_opportunities(self) -> List[YieldData]:
        """Generate mock yield opportunities for demonstration"""
        protocols = [
            {'name': 'Uniswap V3', 'chain': 'ethereum', 'base_apy': 15},
            {'name': 'PancakeSwap', 'chain': 'bsc', 'base_apy': 25},
            {'name': 'SushiSwap', 'chain': 'ethereum', 'base_apy': 18},
            {'name': 'Raydium', 'chain': 'solana', 'base_apy': 30},
            {'name': 'Jupiter', 'chain': 'solana', 'base_apy': 22},
        ]
        
        opportunities = []
        for protocol in protocols:
            # Generate random but realistic yield data
            apy = protocol['base_apy'] + np.random.uniform(-5, 10)
            tvl = np.random.uniform(100000, 5000000)
            
            opportunities.append(YieldData(
                protocol=protocol['name'],
                pool_name=f"{protocol['name']} Liquidity Pool",
                apy=apy,
                tvl=tvl,
                tokens=['Multi-token'],
                chain=protocol['chain']
            ))
        
        return opportunities
    
    # Comprehensive data aggregation
    async def get_comprehensive_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive market data from all free sources"""
        start_time = time.time()
        
        logger.info(f"🔍 Fetching free data for {symbols}")
        
        # Fetch data from multiple sources
        coingecko_data = await self.get_coingecko_prices(symbols)
        dex_data = await self.get_dexscreener_trending()
        yahoo_data = self.get_yahoo_finance_data(symbols)
        exchange_data = await self.get_exchange_prices(symbols)
        yield_data = self.get_mock_yield_opportunities()
        
        # Cross-validate prices
        price_consensus = self._get_price_consensus(symbols, coingecko_data, yahoo_data, exchange_data)
        
        fetch_time = time.time() - start_time
        
        result = {
            'market_data': coingecko_data,
            'dex_pairs': dex_data,
            'yahoo_data': yahoo_data,
            'exchange_data': exchange_data,
            'yield_opportunities': yield_data,
            'price_consensus': price_consensus,
            'metadata': {
                'fetch_time_seconds': fetch_time,
                'sources_used': ['coingecko_free', 'dexscreener', 'yahoo_finance', 'ccxt_exchanges'],
                'symbols_requested': symbols,
                'symbols_found': len(coingecko_data),
                'total_data_points': len(coingecko_data) + len(yahoo_data) + len(exchange_data)
            },
            'timestamp': datetime.now(),
            'free_tier_only': True
        }
        
        logger.info(f"✅ Free data fetch completed in {fetch_time:.2f}s")
        logger.info(f"   CoinGecko: {len(coingecko_data)} symbols")
        logger.info(f"   Yahoo Finance: {len(yahoo_data)} symbols")
        logger.info(f"   Exchange data: {len(exchange_data)} symbols")
        logger.info(f"   DEX pairs: {len(dex_data)}")
        logger.info(f"   Yield opportunities: {len(yield_data)}")
        
        return result
    
    def _get_price_consensus(self, symbols: List[str], *data_sources) -> Dict[str, Dict]:
        """Get price consensus across multiple sources"""
        consensus = {}
        
        for symbol in symbols:
            prices = []
            sources = []
            
            # Collect prices from all sources
            for source_data in data_sources:
                for data_point in source_data:
                    if data_point.symbol == symbol and data_point.price > 0:
                        prices.append(data_point.price)
                        sources.append(data_point.source)
            
            if len(prices) >= 2:
                avg_price = np.mean(prices)
                price_std = np.std(prices)
                price_range = (min(prices), max(prices))
                
                consensus[symbol] = {
                    'average_price': avg_price,
                    'price_range': price_range,
                    'price_std': price_std,
                    'num_sources': len(prices),
                    'sources': sources,
                    'reliable': price_std / avg_price < 0.05  # 5% tolerance
                }
        
        return consensus

# Utility functions
async def get_simple_market_data(symbols: List[str] = None) -> Dict[str, Any]:
    """
    Get market data using only free sources
    
    Args:
        symbols: List of symbols (default: ['BTC', 'ETH', 'SOL', 'SUI'])
    
    Returns:
        Comprehensive market data from free sources only
    """
    if symbols is None:
        symbols = ['BTC', 'ETH', 'SOL', 'SUI']
    
    async with SimpleFreeDataSources() as data_source:
        return await data_source.get_comprehensive_data(symbols)

def print_data_summary(data: Dict[str, Any]):
    """Print a summary of the retrieved data"""
    metadata = data.get('metadata', {})
    
    print(f"\n📊 Free Data Summary:")
    print(f"   • Fetch time: {metadata.get('fetch_time_seconds', 0):.2f}s")
    print(f"   • Sources: {', '.join(metadata.get('sources_used', []))}")
    print(f"   • Symbols found: {metadata.get('symbols_found', 0)}")
    print(f"   • Total data points: {metadata.get('total_data_points', 0)}")
    
    # Price consensus
    consensus = data.get('price_consensus', {})
    print(f"   • Price consensus: {len(consensus)} symbols")
    
    for symbol, info in consensus.items():
        reliability = "✅" if info['reliable'] else "⚠️"
        print(f"     {symbol}: ${info['average_price']:,.2f} {reliability} ({info['num_sources']} sources)")

# Demo function
async def demo_free_data():
    """Demo the free data sources"""
    print("🚀 Free Data Sources Demo")
    print("=" * 50)
    print("Using ONLY free APIs - no premium subscriptions required!")
    
    symbols = ['BTC', 'ETH', 'SOL', 'SUI']
    
    try:
        data = await get_simple_market_data(symbols)
        print_data_summary(data)
        
        # Show some yield opportunities
        yields = data.get('yield_opportunities', [])
        if yields:
            print(f"\n💰 Yield Opportunities:")
            for opp in yields[:3]:
                print(f"   • {opp.protocol}: {opp.apy:.1f}% APY (${opp.tvl:,.0f} TVL)")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(demo_free_data()) 