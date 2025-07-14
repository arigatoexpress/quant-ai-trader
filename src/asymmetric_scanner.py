"""
Asymmetric Trading Opportunity Scanner
Scans multiple data sources to identify high-value, low-risk trading opportunities
"""

import asyncio
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import yaml
import json
from dataclasses import dataclass, asdict
import time
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Local imports
try:
    from data_stream_integrations import DataStreamIntegrations, DataPoint, DEXData, DeFiMetrics
    from telegram_bot import send_quick_alert, TradingAlert
except ImportError:
    print("⚠️  Data stream integrations not available in demo mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AsymmetricOpportunity:
    """Asymmetric trading opportunity data structure"""
    opportunity_id: str
    asset: str
    opportunity_type: str  # defi_yield, price_momentum, arbitrage, new_listing
    expected_return: float  # Percentage
    risk_score: float  # 0-1 scale (0 = lowest risk)
    confidence_score: float  # 0-1 scale (1 = highest confidence)
    time_horizon: str  # short, medium, long
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    volume_requirement: Optional[float]
    liquidity_score: float  # 0-1 scale
    market_cap: Optional[float]
    data_sources: List[str]
    analysis: str
    action_plan: str
    discovered_at: datetime
    expires_at: Optional[datetime]
    raw_data: Optional[Dict[str, Any]] = None

@dataclass
class ScannerConfig:
    """Configuration for the asymmetric scanner"""
    min_expected_return: float = 15.0  # Minimum expected return %
    max_risk_score: float = 0.7  # Maximum acceptable risk
    min_confidence: float = 0.6  # Minimum confidence score
    min_liquidity: float = 0.3  # Minimum liquidity score
    max_market_cap: float = 10_000_000_000  # $10B max market cap
    min_market_cap: float = 1_000_000  # $1M min market cap
    scan_interval_minutes: int = 30
    max_opportunities: int = 50
    alert_threshold: float = 8.0  # Alert on opportunities > 8/10 score

class AsymmetricTradingScanner:
    """Advanced scanner for asymmetric trading opportunities"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_path)
        self.scanner_config = ScannerConfig(**self.config.get('scanner', {}))
        
        # Initialize data integrations
        self.data_streams = None
        
        # Opportunity storage
        self.opportunities: List[AsymmetricOpportunity] = []
        self.opportunity_history: List[AsymmetricOpportunity] = []
        
        # Scanning state
        self.scanning_active = False
        self.last_scan_time = None
        
        # Performance tracking
        self.scan_stats = {
            'total_scans': 0,
            'opportunities_found': 0,
            'avg_scan_time': 0,
            'last_scan_duration': 0
        }
        
        # Analysis models
        self.momentum_thresholds = {
            'strong_up': 20.0,
            'moderate_up': 10.0,
            'sideways': 5.0,
            'moderate_down': -10.0,
            'strong_down': -20.0
        }
        
        logger.info("🔍 Asymmetric Trading Scanner initialized")
        logger.info(f"   Min Expected Return: {self.scanner_config.min_expected_return}%")
        logger.info(f"   Max Risk Score: {self.scanner_config.max_risk_score}")
        logger.info(f"   Scan Interval: {self.scanner_config.scan_interval_minutes} minutes")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        return {
            'scanner': {
                'min_expected_return': 15.0,
                'max_risk_score': 0.7,
                'min_confidence': 0.6,
                'min_liquidity': 0.3,
                'scan_interval_minutes': 30,
                'max_opportunities': 50,
                'alert_threshold': 8.0
            },
            'assets': ['BTC', 'ETH', 'SOL', 'SUI', 'SEI'],
            'defi_chains': ['ethereum', 'sui', 'solana', 'polygon', 'arbitrum'],
            'notifications': {
                'telegram_enabled': True,
                'min_score_for_alert': 8.0
            }
        }
    
    async def initialize_data_streams(self):
        """Initialize data stream connections"""
        try:
            self.data_streams = DataStreamIntegrations()
            await self.data_streams.__aenter__()
            logger.info("✅ Data streams initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize data streams: {e}")
            self.data_streams = None
    
    async def scan_for_opportunities(self) -> List[AsymmetricOpportunity]:
        """Main scanning function to find asymmetric opportunities"""
        start_time = time.time()
        all_opportunities = []
        
        try:
            logger.info("🔍 Starting asymmetric opportunity scan...")
            
            if not self.data_streams:
                await self.initialize_data_streams()
            
            if not self.data_streams:
                logger.warning("⚠️  No data streams available, using mock data")
                return self._generate_mock_opportunities()
            
            # Get comprehensive market data
            market_data = await self.data_streams.get_comprehensive_market_data(
                self.config.get('assets', ['BTC', 'ETH', 'SOL', 'SUI', 'SEI'])
            )
            
            # Scan different opportunity types in parallel
            opportunity_tasks = [
                self._scan_defi_yields(market_data),
                self._scan_price_momentum(market_data),
                self._scan_arbitrage_opportunities(market_data),
                self._scan_new_listings(market_data),
                self._scan_volume_anomalies(market_data)
            ]
            
            results = await asyncio.gather(*opportunity_tasks, return_exceptions=True)
            
            # Collect all opportunities
            for result in results:
                if not isinstance(result, Exception) and result:
                    all_opportunities.extend(result)
            
            # Filter and rank opportunities
            filtered_opportunities = self._filter_opportunities(all_opportunities)
            ranked_opportunities = self._rank_opportunities(filtered_opportunities)
            
            # Update statistics
            scan_duration = time.time() - start_time
            self.scan_stats['total_scans'] += 1
            self.scan_stats['opportunities_found'] += len(ranked_opportunities)
            self.scan_stats['last_scan_duration'] = scan_duration
            self.scan_stats['avg_scan_time'] = (
                (self.scan_stats['avg_scan_time'] * (self.scan_stats['total_scans'] - 1) + scan_duration) /
                self.scan_stats['total_scans']
            )
            
            self.last_scan_time = datetime.now()
            self.opportunities = ranked_opportunities[:self.scanner_config.max_opportunities]
            
            logger.info(f"✅ Scan completed in {scan_duration:.2f}s")
            logger.info(f"   Found {len(all_opportunities)} raw opportunities")
            logger.info(f"   Filtered to {len(ranked_opportunities)} qualified opportunities")
            
            # Send alerts for high-score opportunities
            await self._send_opportunity_alerts(ranked_opportunities)
            
            return ranked_opportunities
            
        except Exception as e:
            logger.error(f"❌ Error during opportunity scan: {e}")
            return []
    
    async def _scan_defi_yields(self, market_data: Dict[str, Any]) -> List[AsymmetricOpportunity]:
        """Scan for high-yield DeFi opportunities"""
        opportunities = []
        
        try:
            defi_yields = market_data.get('defi_yields', [])
            
            for yield_data in defi_yields[:20]:  # Top 20 yields
                if isinstance(yield_data, dict):
                    apy = yield_data.get('apy', 0)
                    tvl = yield_data.get('tvl', 0)
                    risk_score = yield_data.get('risk_score', 0.5)
                    
                    if apy > self.scanner_config.min_expected_return and risk_score < self.scanner_config.max_risk_score:
                        
                        # Calculate scores
                        confidence_score = self._calculate_defi_confidence(yield_data)
                        liquidity_score = min(1.0, np.log(tvl) / 20) if tvl > 0 else 0
                        
                        opportunity = AsymmetricOpportunity(
                            opportunity_id=f"defi_{yield_data.get('pool', 'unknown')}_{int(time.time())}",
                            asset=yield_data.get('symbol', 'UNKNOWN'),
                            opportunity_type='defi_yield',
                            expected_return=apy,
                            risk_score=risk_score,
                            confidence_score=confidence_score,
                            time_horizon='medium',
                            entry_price=None,
                            target_price=None,
                            stop_loss=None,
                            volume_requirement=None,
                            liquidity_score=liquidity_score,
                            market_cap=None,
                            data_sources=['defillama'],
                            analysis=f"High-yield DeFi opportunity: {apy:.1f}% APY with {risk_score:.2f} risk score",
                            action_plan=f"Consider LP position in {yield_data.get('project', 'Unknown')} with ${yield_data.get('tvl', 0):,.0f} TVL",
                            discovered_at=datetime.now(),
                            expires_at=datetime.now() + timedelta(hours=24),
                            raw_data=yield_data
                        )
                        
                        opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"❌ Error scanning DeFi yields: {e}")
        
        return opportunities
    
    async def _scan_price_momentum(self, market_data: Dict[str, Any]) -> List[AsymmetricOpportunity]:
        """Scan for price momentum opportunities"""
        opportunities = []
        
        try:
            coingecko_data = market_data.get('coingecko_data', [])
            
            for data_point in coingecko_data:
                if isinstance(data_point, DataPoint):
                    change_24h = data_point.price_change_24h
                    volume = data_point.volume_24h
                    market_cap = data_point.market_cap
                    
                    # Look for strong momentum with good volume
                    if (abs(change_24h) > 10 and volume > 1_000_000 and 
                        market_cap and self.scanner_config.min_market_cap < market_cap < self.scanner_config.max_market_cap):
                        
                        # Determine if bullish or bearish momentum
                        is_bullish = change_24h > 0
                        momentum_strength = abs(change_24h)
                        
                        # Calculate scores
                        confidence_score = self._calculate_momentum_confidence(data_point)
                        risk_score = self._calculate_momentum_risk(data_point)
                        liquidity_score = min(1.0, volume / 10_000_000)  # $10M volume = 1.0 score
                        
                        expected_return = momentum_strength * 0.8 if is_bullish else momentum_strength * 0.6
                        
                        if expected_return > self.scanner_config.min_expected_return:
                            opportunity = AsymmetricOpportunity(
                                opportunity_id=f"momentum_{data_point.asset}_{int(time.time())}",
                                asset=data_point.symbol,
                                opportunity_type='price_momentum',
                                expected_return=expected_return,
                                risk_score=risk_score,
                                confidence_score=confidence_score,
                                time_horizon='short' if momentum_strength > 20 else 'medium',
                                entry_price=data_point.price,
                                target_price=data_point.price * (1 + expected_return/100) if is_bullish else data_point.price * (1 - expected_return/100),
                                stop_loss=data_point.price * 0.95 if is_bullish else data_point.price * 1.05,
                                volume_requirement=volume * 0.1,  # 10% of daily volume
                                liquidity_score=liquidity_score,
                                market_cap=market_cap,
                                data_sources=['coingecko'],
                                analysis=f"Strong {'bullish' if is_bullish else 'bearish'} momentum: {change_24h:+.1f}% with ${volume:,.0f} volume",
                                action_plan=f"{'Long' if is_bullish else 'Short'} position with tight stop-loss",
                                discovered_at=datetime.now(),
                                expires_at=datetime.now() + timedelta(hours=6),
                                raw_data=asdict(data_point)
                            )
                            
                            opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"❌ Error scanning price momentum: {e}")
        
        return opportunities
    
    async def _scan_arbitrage_opportunities(self, market_data: Dict[str, Any]) -> List[AsymmetricOpportunity]:
        """Scan for arbitrage opportunities between DEXs"""
        opportunities = []
        
        try:
            dex_data = market_data.get('dex_data', [])
            
            # Group by base token
            token_prices = {}
            for dex_pair in dex_data:
                if isinstance(dex_pair, DEXData):
                    token = dex_pair.base_token
                    if token not in token_prices:
                        token_prices[token] = []
                    
                    token_prices[token].append({
                        'price': dex_pair.price,
                        'dex': dex_pair.dex_name,
                        'liquidity': dex_pair.liquidity,
                        'volume': dex_pair.volume_24h
                    })
            
            # Find arbitrage opportunities
            for token, prices in token_prices.items():
                if len(prices) >= 2:
                    prices.sort(key=lambda x: x['price'])
                    lowest = prices[0]
                    highest = prices[-1]
                    
                    if lowest['price'] > 0 and highest['liquidity'] > 50000 and lowest['liquidity'] > 50000:
                        price_diff = (highest['price'] - lowest['price']) / lowest['price'] * 100
                        
                        if price_diff > 3:  # At least 3% arbitrage opportunity
                            
                            # Calculate scores
                            confidence_score = min(1.0, min(lowest['liquidity'], highest['liquidity']) / 500000)
                            risk_score = 1 - confidence_score  # Higher liquidity = lower risk
                            liquidity_score = min(1.0, min(lowest['liquidity'], highest['liquidity']) / 1000000)
                            
                            opportunity = AsymmetricOpportunity(
                                opportunity_id=f"arbitrage_{token}_{int(time.time())}",
                                asset=token,
                                opportunity_type='arbitrage',
                                expected_return=price_diff * 0.8,  # Account for fees
                                risk_score=risk_score,
                                confidence_score=confidence_score,
                                time_horizon='short',
                                entry_price=lowest['price'],
                                target_price=highest['price'],
                                stop_loss=lowest['price'] * 0.98,
                                volume_requirement=min(lowest['liquidity'], highest['liquidity']) * 0.05,
                                liquidity_score=liquidity_score,
                                market_cap=None,
                                data_sources=['dexscreener'],
                                analysis=f"Arbitrage opportunity: {price_diff:.2f}% spread between {lowest['dex']} and {highest['dex']}",
                                action_plan=f"Buy on {lowest['dex']} at ${lowest['price']:.4f}, sell on {highest['dex']} at ${highest['price']:.4f}",
                                discovered_at=datetime.now(),
                                expires_at=datetime.now() + timedelta(minutes=30),
                                raw_data={'lowest': lowest, 'highest': highest}
                            )
                            
                            opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"❌ Error scanning arbitrage opportunities: {e}")
        
        return opportunities
    
    async def _scan_new_listings(self, market_data: Dict[str, Any]) -> List[AsymmetricOpportunity]:
        """Scan for new listing opportunities with high potential"""
        opportunities = []
        
        try:
            # This would typically involve checking new DEX pairs with recent creation
            dex_data = market_data.get('dex_data', [])
            
            for dex_pair in dex_data[:10]:  # Check top 10 pairs
                if isinstance(dex_pair, DEXData):
                    # Look for pairs with high volume but relatively low market cap (potential new gems)
                    if (dex_pair.volume_24h > 100000 and  # Good volume
                        dex_pair.tx_count_24h > 100 and  # Active trading
                        dex_pair.liquidity > 50000):  # Minimum liquidity
                        
                        # Estimate market cap based on liquidity (rough approximation)
                        estimated_market_cap = dex_pair.liquidity * 10  # Rough estimate
                        
                        if estimated_market_cap < 50_000_000:  # Under $50M estimated market cap
                            
                            # Calculate scores based on activity and fundamentals
                            volume_score = min(1.0, dex_pair.volume_24h / 1_000_000)
                            liquidity_score = min(1.0, dex_pair.liquidity / 1_000_000)
                            activity_score = min(1.0, dex_pair.tx_count_24h / 1000)
                            
                            confidence_score = (volume_score + liquidity_score + activity_score) / 3
                            risk_score = 0.8  # High risk for new listings
                            
                            if confidence_score > self.scanner_config.min_confidence:
                                expected_return = 50 + (confidence_score * 50)  # 50-100% potential
                                
                                opportunity = AsymmetricOpportunity(
                                    opportunity_id=f"new_listing_{dex_pair.base_token}_{int(time.time())}",
                                    asset=dex_pair.base_token,
                                    opportunity_type='new_listing',
                                    expected_return=expected_return,
                                    risk_score=risk_score,
                                    confidence_score=confidence_score,
                                    time_horizon='medium',
                                    entry_price=dex_pair.price,
                                    target_price=dex_pair.price * (1 + expected_return/100),
                                    stop_loss=dex_pair.price * 0.8,  # 20% stop loss
                                    volume_requirement=dex_pair.volume_24h * 0.05,
                                    liquidity_score=liquidity_score,
                                    market_cap=estimated_market_cap,
                                    data_sources=['dexscreener'],
                                    analysis=f"Active new token: ${dex_pair.volume_24h:,.0f} volume, {dex_pair.tx_count_24h} txns",
                                    action_plan=f"Small position entry with tight stop-loss, monitor for breakout",
                                    discovered_at=datetime.now(),
                                    expires_at=datetime.now() + timedelta(hours=12),
                                    raw_data=asdict(dex_pair)
                                )
                                
                                opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"❌ Error scanning new listings: {e}")
        
        return opportunities
    
    async def _scan_volume_anomalies(self, market_data: Dict[str, Any]) -> List[AsymmetricOpportunity]:
        """Scan for unusual volume spikes that might indicate accumulation"""
        opportunities = []
        
        try:
            coingecko_data = market_data.get('coingecko_data', [])
            
            for data_point in coingecko_data:
                if isinstance(data_point, DataPoint):
                    # Look for high volume with moderate price movement (accumulation pattern)
                    volume = data_point.volume_24h
                    price_change = abs(data_point.price_change_24h)
                    
                    if volume > 5_000_000 and price_change < 10:  # High volume, low volatility
                        
                        # Calculate volume anomaly score
                        volume_score = min(1.0, volume / 50_000_000)  # $50M = perfect score
                        stability_score = max(0, 1 - (price_change / 20))  # Lower volatility = higher score
                        
                        confidence_score = (volume_score + stability_score) / 2
                        risk_score = 0.4  # Medium risk for accumulation plays
                        
                        if confidence_score > self.scanner_config.min_confidence:
                            expected_return = 25 + (confidence_score * 25)  # 25-50% potential
                            
                            opportunity = AsymmetricOpportunity(
                                opportunity_id=f"volume_anomaly_{data_point.symbol}_{int(time.time())}",
                                asset=data_point.symbol,
                                opportunity_type='volume_anomaly',
                                expected_return=expected_return,
                                risk_score=risk_score,
                                confidence_score=confidence_score,
                                time_horizon='medium',
                                entry_price=data_point.price,
                                target_price=data_point.price * (1 + expected_return/100),
                                stop_loss=data_point.price * 0.9,
                                volume_requirement=volume * 0.02,  # 2% of daily volume
                                liquidity_score=min(1.0, volume / 10_000_000),
                                market_cap=data_point.market_cap,
                                data_sources=['coingecko'],
                                analysis=f"Volume anomaly: ${volume:,.0f} volume with only {price_change:.1f}% price change",
                                action_plan=f"Accumulation pattern detected, gradual position building recommended",
                                discovered_at=datetime.now(),
                                expires_at=datetime.now() + timedelta(hours=24),
                                raw_data=asdict(data_point)
                            )
                            
                            opportunities.append(opportunity)
            
        except Exception as e:
            logger.error(f"❌ Error scanning volume anomalies: {e}")
        
        return opportunities
    
    def _calculate_defi_confidence(self, yield_data: Dict[str, Any]) -> float:
        """Calculate confidence score for DeFi yield opportunities"""
        score = 0.5  # Base score
        
        # TVL factor
        tvl = yield_data.get('tvl', 0)
        if tvl > 10_000_000:
            score += 0.3
        elif tvl > 1_000_000:
            score += 0.2
        elif tvl > 100_000:
            score += 0.1
        
        # APY reasonableness
        apy = yield_data.get('apy', 0)
        if 15 <= apy <= 100:  # Reasonable range
            score += 0.2
        elif apy > 100:
            score -= 0.1  # Too good to be true
        
        return min(1.0, max(0.0, score))
    
    def _calculate_momentum_confidence(self, data_point: DataPoint) -> float:
        """Calculate confidence score for momentum opportunities"""
        score = 0.5  # Base score
        
        # Volume factor
        if data_point.volume_24h > 10_000_000:
            score += 0.3
        elif data_point.volume_24h > 1_000_000:
            score += 0.2
        
        # Market cap factor
        if data_point.market_cap:
            if 100_000_000 < data_point.market_cap < 10_000_000_000:  # Sweet spot
                score += 0.2
            elif data_point.market_cap > 10_000_000_000:  # Too large
                score -= 0.1
        
        return min(1.0, max(0.0, score))
    
    def _calculate_momentum_risk(self, data_point: DataPoint) -> float:
        """Calculate risk score for momentum opportunities"""
        risk = 0.5  # Base risk
        
        # Volatility factor
        volatility = abs(data_point.price_change_24h)
        if volatility > 30:
            risk += 0.3
        elif volatility > 20:
            risk += 0.2
        elif volatility < 5:
            risk -= 0.1
        
        # Market cap factor
        if data_point.market_cap:
            if data_point.market_cap < 10_000_000:  # Small cap = higher risk
                risk += 0.3
            elif data_point.market_cap > 1_000_000_000:  # Large cap = lower risk
                risk -= 0.2
        
        return min(1.0, max(0.0, risk))
    
    def _filter_opportunities(self, opportunities: List[AsymmetricOpportunity]) -> List[AsymmetricOpportunity]:
        """Filter opportunities based on configuration thresholds"""
        filtered = []
        
        for opp in opportunities:
            if (opp.expected_return >= self.scanner_config.min_expected_return and
                opp.risk_score <= self.scanner_config.max_risk_score and
                opp.confidence_score >= self.scanner_config.min_confidence and
                opp.liquidity_score >= self.scanner_config.min_liquidity):
                
                # Additional market cap filters
                if opp.market_cap:
                    if not (self.scanner_config.min_market_cap <= opp.market_cap <= self.scanner_config.max_market_cap):
                        continue
                
                filtered.append(opp)
        
        return filtered
    
    def _rank_opportunities(self, opportunities: List[AsymmetricOpportunity]) -> List[AsymmetricOpportunity]:
        """Rank opportunities by overall score"""
        
        for opp in opportunities:
            # Calculate composite score (0-10 scale)
            return_score = min(10, opp.expected_return / 10)  # 100% return = 10 points
            risk_score = (1 - opp.risk_score) * 10  # Lower risk = higher score
            confidence_score = opp.confidence_score * 10
            liquidity_score = opp.liquidity_score * 10
            
            # Weight the scores
            composite_score = (
                return_score * 0.3 +
                risk_score * 0.25 +
                confidence_score * 0.25 +
                liquidity_score * 0.2
            )
            
            # Store score in raw_data for sorting
            if not opp.raw_data:
                opp.raw_data = {}
            opp.raw_data['composite_score'] = composite_score
        
        # Sort by composite score (descending)
        return sorted(opportunities, key=lambda x: x.raw_data.get('composite_score', 0), reverse=True)
    
    async def _send_opportunity_alerts(self, opportunities: List[AsymmetricOpportunity]):
        """Send alerts for high-scoring opportunities"""
        if not self.config.get('notifications', {}).get('telegram_enabled', False):
            return
        
        alert_threshold = self.config.get('notifications', {}).get('min_score_for_alert', 8.0)
        
        for opp in opportunities[:5]:  # Top 5 opportunities
            score = opp.raw_data.get('composite_score', 0) if opp.raw_data else 0
            
            if score >= alert_threshold:
                urgency = 'high' if score >= 9 else 'medium'
                
                alert_message = f"""
🎯 **ASYMMETRIC OPPORTUNITY**

**Asset:** {opp.asset}
**Type:** {opp.opportunity_type.replace('_', ' ').title()}
**Expected Return:** {opp.expected_return:.1f}%
**Risk Score:** {opp.risk_score:.2f}/1.0
**Confidence:** {opp.confidence_score:.2f}/1.0
**Score:** {score:.1f}/10

**Analysis:** {opp.analysis}

**Action:** {opp.action_plan}
                """
                
                try:
                    await send_quick_alert(alert_message, urgency)
                except Exception as e:
                    logger.error(f"❌ Failed to send opportunity alert: {e}")
    
    def _generate_mock_opportunities(self) -> List[AsymmetricOpportunity]:
        """Generate mock opportunities for testing"""
        mock_opportunities = [
            AsymmetricOpportunity(
                opportunity_id="mock_defi_1",
                asset="SUI",
                opportunity_type="defi_yield",
                expected_return=45.0,
                risk_score=0.6,
                confidence_score=0.8,
                time_horizon="medium",
                entry_price=4.25,
                target_price=None,
                stop_loss=None,
                volume_requirement=None,
                liquidity_score=0.7,
                market_cap=12_000_000_000,
                data_sources=["mock"],
                analysis="High-yield liquidity pool on Sui network",
                action_plan="Provide liquidity to SUI/USDC pool",
                discovered_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=24),
                raw_data={"composite_score": 8.5}
            ),
            AsymmetricOpportunity(
                opportunity_id="mock_momentum_1",
                asset="SOL",
                opportunity_type="price_momentum",
                expected_return=25.0,
                risk_score=0.5,
                confidence_score=0.9,
                time_horizon="short",
                entry_price=245.50,
                target_price=306.88,
                stop_loss=233.23,
                volume_requirement=1_000_000,
                liquidity_score=0.9,
                market_cap=115_000_000_000,
                data_sources=["mock"],
                analysis="Strong bullish momentum with high volume",
                action_plan="Long position with tight stop-loss",
                discovered_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=6),
                raw_data={"composite_score": 8.2}
            )
        ]
        
        return mock_opportunities
    
    async def start_continuous_scanning(self):
        """Start continuous scanning for opportunities"""
        self.scanning_active = True
        logger.info("🔄 Starting continuous asymmetric opportunity scanning...")
        
        while self.scanning_active:
            try:
                opportunities = await self.scan_for_opportunities()
                
                if opportunities:
                    logger.info(f"✅ Found {len(opportunities)} opportunities")
                    self._save_opportunities_to_file(opportunities)
                
                # Wait for next scan
                await asyncio.sleep(self.scanner_config.scan_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"❌ Error in continuous scanning: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    def stop_scanning(self):
        """Stop continuous scanning"""
        self.scanning_active = False
        logger.info("⏹️ Stopped asymmetric opportunity scanning")
    
    def _save_opportunities_to_file(self, opportunities: List[AsymmetricOpportunity]):
        """Save opportunities to JSON file for persistence"""
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'scan_stats': self.scan_stats,
                'opportunities': [asdict(opp) for opp in opportunities]
            }
            
            with open('asymmetric_opportunities.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"❌ Failed to save opportunities: {e}")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Get summary report of scanning activity"""
        return {
            'scanning_active': self.scanning_active,
            'last_scan_time': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'total_opportunities': len(self.opportunities),
            'scan_stats': self.scan_stats,
            'top_opportunities': [
                {
                    'asset': opp.asset,
                    'type': opp.opportunity_type,
                    'expected_return': opp.expected_return,
                    'risk_score': opp.risk_score,
                    'score': opp.raw_data.get('composite_score', 0) if opp.raw_data else 0
                }
                for opp in self.opportunities[:5]
            ]
        }

# Utility functions
async def run_single_scan(config_path: Optional[str] = None) -> List[AsymmetricOpportunity]:
    """Run a single opportunity scan"""
    scanner = AsymmetricTradingScanner(config_path)
    return await scanner.scan_for_opportunities()

async def start_scanner_service(config_path: Optional[str] = None):
    """Start the scanner as a service"""
    scanner = AsymmetricTradingScanner(config_path)
    await scanner.start_continuous_scanning()

# Main execution
if __name__ == "__main__":
    async def main():
        scanner = AsymmetricTradingScanner()
        
        print("🔍 Asymmetric Trading Opportunity Scanner")
        print("1. Single scan")
        print("2. Continuous scanning")
        
        choice = input("Choose option (1 or 2): ").strip()
        
        if choice == "1":
            opportunities = await scanner.scan_for_opportunities()
            
            print(f"\n📊 Found {len(opportunities)} opportunities:")
            for i, opp in enumerate(opportunities[:10], 1):
                score = opp.raw_data.get('composite_score', 0) if opp.raw_data else 0
                print(f"{i}. {opp.asset} - {opp.opportunity_type} - {opp.expected_return:.1f}% return - Score: {score:.1f}/10")
        
        elif choice == "2":
            await scanner.start_continuous_scanning()
        
        else:
            print("Invalid choice")
    
    asyncio.run(main()) 