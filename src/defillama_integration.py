"""
DeFi Llama API Integration
Real DeFi data respecting free tier rate limits
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import time
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Protocol:
    """DeFi protocol data"""
    id: str
    name: str
    address: Optional[str]
    symbol: str
    url: str
    description: str
    chain: str
    logo: str
    audits: str
    audit_note: Optional[str]
    gecko_id: Optional[str]
    cmcId: Optional[str]
    category: str
    chains: List[str]
    module: str
    twitter: Optional[str]
    forkedFrom: List[str]
    oracles: List[str]
    listedAt: int
    methodology: str
    slug: str
    tvl: float
    chainTvls: Dict[str, float]
    change_1h: Optional[float]
    change_1d: Optional[float]
    change_7d: Optional[float]
    tokenBreakdowns: Optional[Dict]
    mcap: Optional[float]

@dataclass
class YieldPool:
    """Yield farming pool data"""
    pool: str
    chain: str
    project: str
    symbol: str
    tvlUsd: float
    apyBase: Optional[float]
    apyReward: Optional[float] 
    apy: float
    rewardTokens: List[str]
    count: int
    outlier: bool
    mu: float
    sigma: float
    il7d: Optional[float]
    apyBase7d: Optional[float]
    apyMean30d: float
    volumeUsd1d: Optional[float]
    volumeUsd7d: Optional[float]
    stablecoin: bool
    ilRisk: str
    exposure: str
    predictions: Dict[str, Any]
    poolMeta: Optional[str]
    underlyingTokens: List[str]
    url: str

@dataclass
class TVLData:
    """TVL (Total Value Locked) data"""
    date: str
    totalLiquidityUSD: float

class DeFiLlamaAPI:
    """DeFi Llama API client with free tier rate limiting"""
    
    def __init__(self):
        self.session = None
        self.base_url = "https://api.llama.fi"
        self.yields_url = "https://yields.llama.fi"
        
        # Conservative rate limiting for free tier
        # DeFi Llama doesn't publish specific limits but we'll be conservative
        self.rate_limit = {
            'requests_per_minute': 30,  # Conservative estimate
            'min_interval': 2,  # 2 seconds between requests
            'last_request': 0
        }
        
        # Cache to reduce API calls
        self.cache = {}
        self.cache_ttl = {}
        self.cache_duration = 300  # 5 minutes
        
    async def __aenter__(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'QuantAITrader/1.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async session"""
        if self.session:
            await self.session.close()
    
    def _should_rate_limit(self) -> bool:
        """Check if we should rate limit"""
        now = time.time()
        time_since_last = now - self.rate_limit['last_request']
        
        if time_since_last < self.rate_limit['min_interval']:
            return True
        return False
    
    async def _rate_limited_request(self, url: str, cache_key: str = None) -> Optional[Dict]:
        """Make rate-limited request with caching"""
        # Check cache first
        if cache_key and cache_key in self.cache:
            cached_time = self.cache_ttl.get(cache_key, 0)
            if time.time() - cached_time < self.cache_duration:
                logger.debug(f"Using cached data for {cache_key}")
                return self.cache[cache_key]
        
        # Rate limiting
        if self._should_rate_limit():
            wait_time = self.rate_limit['min_interval'] - (time.time() - self.rate_limit['last_request'])
            await asyncio.sleep(wait_time)
        
        try:
            async with self.session.get(url) as response:
                self.rate_limit['last_request'] = time.time()
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache successful responses
                    if cache_key:
                        self.cache[cache_key] = data
                        self.cache_ttl[cache_key] = time.time()
                    
                    return data
                else:
                    logger.warning(f"DeFi Llama API returned status {response.status} for {url}")
                    
        except Exception as e:
            logger.error(f"DeFi Llama API request failed: {e}")
        
        return None
    
    async def get_protocols(self) -> List[Protocol]:
        """Get all DeFi protocols with TVL data"""
        url = f"{self.base_url}/protocols"
        data = await self._rate_limited_request(url, "protocols")
        
        if not data:
            return []
        
        protocols = []
        for protocol_data in data:
            try:
                protocol = Protocol(
                    id=protocol_data.get('id', ''),
                    name=protocol_data.get('name', ''),
                    address=protocol_data.get('address'),
                    symbol=protocol_data.get('symbol', ''),
                    url=protocol_data.get('url', ''),
                    description=protocol_data.get('description', ''),
                    chain=protocol_data.get('chain', ''),
                    logo=protocol_data.get('logo', ''),
                    audits=protocol_data.get('audits', ''),
                    audit_note=protocol_data.get('audit_note'),
                    gecko_id=protocol_data.get('gecko_id'),
                    cmcId=protocol_data.get('cmcId'),
                    category=protocol_data.get('category', ''),
                    chains=protocol_data.get('chains', []),
                    module=protocol_data.get('module', ''),
                    twitter=protocol_data.get('twitter'),
                    forkedFrom=protocol_data.get('forkedFrom', []),
                    oracles=protocol_data.get('oracles', []),
                    listedAt=protocol_data.get('listedAt', 0),
                    methodology=protocol_data.get('methodology', ''),
                    slug=protocol_data.get('slug', ''),
                    tvl=float(protocol_data.get('tvl', 0)),
                    chainTvls=protocol_data.get('chainTvls', {}),
                    change_1h=protocol_data.get('change_1h'),
                    change_1d=protocol_data.get('change_1d'),
                    change_7d=protocol_data.get('change_7d'),
                    tokenBreakdowns=protocol_data.get('tokenBreakdowns'),
                    mcap=protocol_data.get('mcap')
                )
                protocols.append(protocol)
            except Exception as e:
                logger.warning(f"Error parsing protocol data: {e}")
        
        logger.info(f"Retrieved {len(protocols)} protocols from DeFi Llama")
        return protocols
    
    async def get_protocol_tvl(self, protocol_slug: str) -> List[TVLData]:
        """Get historical TVL data for a specific protocol"""
        url = f"{self.base_url}/protocol/{protocol_slug}"
        data = await self._rate_limited_request(url, f"protocol_tvl_{protocol_slug}")
        
        if not data or 'chainTvls' not in data:
            return []
        
        tvl_data = []
        chain_tvls = data['chainTvls']
        
        # Get the main chain data or combined data
        main_data = chain_tvls.get(list(chain_tvls.keys())[0], {}) if chain_tvls else {}
        tvl_points = main_data.get('tvl', [])
        
        for point in tvl_points[-30:]:  # Last 30 data points
            try:
                tvl_data.append(TVLData(
                    date=datetime.fromtimestamp(point['date']).isoformat(),
                    totalLiquidityUSD=float(point['totalLiquidityUSD'])
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Error parsing TVL data point: {e}")
        
        return tvl_data
    
    async def get_yield_pools(self) -> List[YieldPool]:
        """Get all yield farming pools"""
        url = f"{self.yields_url}/pools"
        data = await self._rate_limited_request(url, "yield_pools")
        
        if not data or 'data' not in data:
            return []
        
        pools = []
        for pool_data in data['data']:
            try:
                pool = YieldPool(
                    pool=pool_data.get('pool', ''),
                    chain=pool_data.get('chain', ''),
                    project=pool_data.get('project', ''),
                    symbol=pool_data.get('symbol', ''),
                    tvlUsd=float(pool_data.get('tvlUsd', 0)),
                    apyBase=pool_data.get('apyBase'),
                    apyReward=pool_data.get('apyReward'),
                    apy=float(pool_data.get('apy', 0)),
                    rewardTokens=pool_data.get('rewardTokens', []),
                    count=int(pool_data.get('count', 0)),
                    outlier=bool(pool_data.get('outlier', False)),
                    mu=float(pool_data.get('mu', 0)),
                    sigma=float(pool_data.get('sigma', 0)),
                    il7d=pool_data.get('il7d'),
                    apyBase7d=pool_data.get('apyBase7d'),
                    apyMean30d=float(pool_data.get('apyMean30d', 0)),
                    volumeUsd1d=pool_data.get('volumeUsd1d'),
                    volumeUsd7d=pool_data.get('volumeUsd7d'),
                    stablecoin=bool(pool_data.get('stablecoin', False)),
                    ilRisk=pool_data.get('ilRisk', 'unknown'),
                    exposure=pool_data.get('exposure', 'unknown'),
                    predictions=pool_data.get('predictions', {}),
                    poolMeta=pool_data.get('poolMeta'),
                    underlyingTokens=pool_data.get('underlyingTokens', []),
                    url=pool_data.get('url', '')
                )
                pools.append(pool)
            except Exception as e:
                logger.warning(f"Error parsing yield pool data: {e}")
        
        logger.info(f"Retrieved {len(pools)} yield pools from DeFi Llama")
        return pools
    
    async def get_high_yield_opportunities(self, 
                                         min_apy: float = 10, 
                                         min_tvl: float = 100000,
                                         max_il_risk: str = 'medium',
                                         exclude_stablecoins: bool = False) -> List[YieldPool]:
        """Get filtered high-yield opportunities"""
        all_pools = await self.get_yield_pools()
        
        filtered_pools = []
        for pool in all_pools:
            # Apply filters
            if pool.apy < min_apy:
                continue
            if pool.tvlUsd < min_tvl:
                continue
            if pool.outlier:
                continue
            if exclude_stablecoins and pool.stablecoin:
                continue
            
            # IL risk filter
            risk_levels = {'no': 0, 'low': 1, 'medium': 2, 'high': 3}
            if risk_levels.get(pool.ilRisk, 3) > risk_levels.get(max_il_risk, 2):
                continue
            
            filtered_pools.append(pool)
        
        # Sort by risk-adjusted yield
        filtered_pools.sort(key=lambda x: x.apy * (1 - 0.1 * risk_levels.get(x.ilRisk, 0)), reverse=True)
        
        logger.info(f"Found {len(filtered_pools)} high-yield opportunities")
        return filtered_pools[:50]  # Top 50
    
    async def get_top_protocols_by_tvl(self, limit: int = 20) -> List[Protocol]:
        """Get top protocols by TVL"""
        protocols = await self.get_protocols()
        
        # Filter out protocols with no TVL data
        valid_protocols = [p for p in protocols if p.tvl > 0]
        
        # Sort by TVL
        valid_protocols.sort(key=lambda x: x.tvl, reverse=True)
        
        return valid_protocols[:limit]
    
    async def get_chain_tvl_summary(self) -> Dict[str, float]:
        """Get TVL summary by blockchain"""
        url = f"{self.base_url}/chains"
        data = await self._rate_limited_request(url, "chains_tvl")
        
        if not data:
            return {}
        
        chain_tvls = {}
        for chain_data in data:
            chain_name = chain_data.get('name', '')
            tvl = float(chain_data.get('tvl', 0))
            if tvl > 0:
                chain_tvls[chain_name] = tvl
        
        return chain_tvls
    
    async def get_comprehensive_defi_data(self) -> Dict[str, Any]:
        """Get comprehensive DeFi ecosystem data"""
        start_time = time.time()
        
        logger.info("🔍 Fetching comprehensive DeFi data from DeFi Llama...")
        
        # Execute requests in sequence to respect rate limits
        top_protocols = await self.get_top_protocols_by_tvl(20)
        yield_opportunities = await self.get_high_yield_opportunities(
            min_apy=15, 
            min_tvl=500000, 
            max_il_risk='medium'
        )
        chain_tvls = await self.get_chain_tvl_summary()
        
        total_tvl = sum(p.tvl for p in top_protocols)
        
        fetch_time = time.time() - start_time
        
        result = {
            'protocols': [
                {
                    'name': p.name,
                    'tvl': p.tvl,
                    'category': p.category,
                    'chain': p.chain,
                    'chains': p.chains,
                    'change_1d': p.change_1d,
                    'change_7d': p.change_7d,
                    'url': p.url,
                    'logo': p.logo
                } for p in top_protocols
            ],
            'yield_opportunities': [
                {
                    'pool_id': y.pool,
                    'project': y.project,
                    'symbol': y.symbol,
                    'chain': y.chain,
                    'apy': y.apy,
                    'apy_base': y.apyBase,
                    'apy_reward': y.apyReward,
                    'tvl_usd': y.tvlUsd,
                    'stablecoin': y.stablecoin,
                    'il_risk': y.ilRisk,
                    'exposure': y.exposure,
                    'url': y.url,
                    'underlying_tokens': y.underlyingTokens
                } for y in yield_opportunities
            ],
            'chain_tvls': chain_tvls,
            'metadata': {
                'total_protocols': len(top_protocols),
                'total_tvl_usd': total_tvl,
                'yield_opportunities_count': len(yield_opportunities),
                'chains_count': len(chain_tvls),
                'fetch_time_seconds': fetch_time,
                'data_source': 'defillama_free_api',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"✅ DeFi Llama data fetch completed in {fetch_time:.2f}s")
        logger.info(f"   Protocols: {len(top_protocols)} (${total_tvl/1e9:.1f}B TVL)")
        logger.info(f"   Yield opportunities: {len(yield_opportunities)}")
        logger.info(f"   Chains: {len(chain_tvls)}")
        
        return result

# Utility functions
async def get_defi_market_data() -> Dict[str, Any]:
    """Get comprehensive DeFi market data using free DeFi Llama API"""
    async with DeFiLlamaAPI() as api:
        return await api.get_comprehensive_defi_data()

async def get_best_yield_opportunities(min_apy: float = 20) -> List[Dict[str, Any]]:
    """Get best yield opportunities above minimum APY"""
    async with DeFiLlamaAPI() as api:
        pools = await api.get_high_yield_opportunities(min_apy=min_apy)
        
        return [
            {
                'protocol': pool.project,
                'symbol': pool.symbol,
                'chain': pool.chain,
                'apy': pool.apy,
                'tvl': pool.tvlUsd,
                'risk': pool.ilRisk,
                'url': pool.url
            } for pool in pools[:10]
        ]

# Demo function
async def demo_defillama():
    """Demo the DeFi Llama integration"""
    print("🦙 DeFi Llama Integration Demo")
    print("=" * 50)
    print("Using FREE DeFi Llama API - real DeFi data!")
    
    try:
        # Get comprehensive data
        data = await get_defi_market_data()
        
        print(f"\n📊 DeFi Market Summary:")
        print(f"   • Total Protocols: {data['metadata']['total_protocols']}")
        print(f"   • Total TVL: ${data['metadata']['total_tvl_usd']/1e9:.1f}B")
        print(f"   • Yield Opportunities: {data['metadata']['yield_opportunities_count']}")
        print(f"   • Chains: {data['metadata']['chains_count']}")
        print(f"   • Fetch Time: {data['metadata']['fetch_time_seconds']:.2f}s")
        
        # Show top protocols
        print(f"\n🏆 Top Protocols by TVL:")
        for i, protocol in enumerate(data['protocols'][:5], 1):
            change_1d = protocol['change_1d'] or 0
            print(f"   {i}. {protocol['name']}: ${protocol['tvl']/1e9:.2f}B "
                  f"({change_1d:+.1f}% 24h) [{protocol['category']}]")
        
        # Show best yields
        print(f"\n💰 Best Yield Opportunities:")
        for i, opp in enumerate(data['yield_opportunities'][:5], 1):
            print(f"   {i}. {opp['project']} ({opp['symbol']}): {opp['apy']:.1f}% APY")
            print(f"      ${opp['tvl_usd']:,.0f} TVL on {opp['chain']} | Risk: {opp['il_risk']}")
        
        # Show chain distribution
        print(f"\n⛓️ Chain TVL Distribution:")
        sorted_chains = sorted(data['chain_tvls'].items(), key=lambda x: x[1], reverse=True)
        for chain, tvl in sorted_chains[:5]:
            print(f"   • {chain}: ${tvl/1e9:.1f}B")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    asyncio.run(demo_defillama()) 