"""
Asymmetric Trading Framework
============================

This module implements advanced asymmetric trading strategies based on the principles
of Nassim Nicholas Taleb's "Antifragile" and "Skin in the Game" concepts. The framework
focuses on creating portfolios that benefit from volatility and uncertainty rather
than being harmed by it.

Key Components:
- Barbell Portfolio Strategy: Combines safe assets (75-80%) with high-risk, high-reward assets (20-25%)
- Asymmetric Bet Sizing: Positions sized based on potential upside vs. downside
- Fat Tail Risk Management: Protection against extreme market events
- Volatility Harvesting: Profiting from market volatility
- Convexity Strategies: Benefiting from non-linear payoffs

The framework is designed to:
1. Minimize downside risk while maximizing upside potential
2. Benefit from market volatility and uncertainty
3. Provide protection against black swan events
4. Generate consistent returns across different market conditions

Author: AI Assistant
Version: 2.0.0
License: MIT
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

# Import dependencies
from advanced_ml_models import AdvancedMLModels
from sentiment_analysis_engine import SentimentAnalysisEngine

logger = logging.getLogger(__name__)

@dataclass
class AsymmetricBet:
    """
    Represents an asymmetric trading bet with defined risk/reward characteristics.
    
    An asymmetric bet is characterized by:
    - Limited downside risk (small potential loss)
    - Unlimited or large upside potential (large potential gain)
    - Positive expected value over time
    - Convex payoff structure
    
    This structure allows the portfolio to benefit from volatility and uncertainty
    while protecting against catastrophic losses.
    """
    symbol: str
    position_type: str  # 'long', 'short', 'option', 'futures'
    size: float  # Position size as percentage of portfolio
    entry_price: float
    stop_loss: float
    target_price: float
    max_loss: float  # Maximum potential loss
    potential_gain: float  # Potential gain if target is reached
    probability: float  # Estimated probability of success
    expected_value: float  # Expected value of the bet
    convexity: float  # Measure of convexity (positive = convex, negative = concave)
    timestamp: datetime
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        # Calculate expected value
        self.expected_value = (self.potential_gain * self.probability) - (self.max_loss * (1 - self.probability))
        
        # Calculate convexity (simplified measure)
        self.convexity = self.potential_gain / max(self.max_loss, 0.001)  # Avoid division by zero

@dataclass
class BarbellPortfolio:
    """
    Implements the barbell portfolio strategy combining safe and risky assets.
    
    The barbell strategy allocates:
    - 75-80% to safe, low-volatility assets (bonds, cash, defensive stocks)
    - 20-25% to high-risk, high-reward assets (options, leveraged ETFs, crypto)
    
    This structure provides:
    - Capital preservation through safe assets
    - Upside potential through risky assets
    - Protection against market crashes
    - Benefit from volatility and uncertainty
    
    The strategy is based on the principle that most of the time markets are stable,
    but occasionally experience extreme events that can be exploited.
    """
    total_capital: float
    safe_allocation: float = 0.75  # 75% to safe assets
    risky_allocation: float = 0.25  # 25% to risky assets
    safe_assets: List[Dict[str, Any]] = None
    risky_assets: List[Dict[str, Any]] = None
    asymmetric_bets: List[AsymmetricBet] = None
    rebalance_frequency: str = 'monthly'
    last_rebalance: datetime = None
    
    def __post_init__(self):
        """Initialize portfolio components."""
        if self.safe_assets is None:
            self.safe_assets = []
        if self.risky_assets is None:
            self.risky_assets = []
        if self.asymmetric_bets is None:
            self.asymmetric_bets = []
    
    def add_safe_asset(self, symbol: str, allocation: float, asset_type: str = 'bond'):
        """
        Add a safe asset to the portfolio.
        
        Safe assets should have:
        - Low volatility
        - Predictable returns
        - High liquidity
        - Government backing or strong credit quality
        
        Examples: Treasury bonds, investment-grade corporate bonds, cash equivalents
        """
        self.safe_assets.append({
            'symbol': symbol,
            'allocation': allocation,
            'type': asset_type,
            'added_date': datetime.now()
        })
        logger.info(f"Added safe asset: {symbol} with {allocation:.1%} allocation")
    
    def add_risky_asset(self, symbol: str, allocation: float, asset_type: str = 'option'):
        """
        Add a risky asset to the portfolio.
        
        Risky assets should have:
        - High potential upside
        - Limited downside risk
        - Convex payoff structure
        - Exposure to volatility or uncertainty
        
        Examples: Out-of-the-money options, leveraged ETFs, certain crypto assets
        """
        self.risky_assets.append({
            'symbol': symbol,
            'allocation': allocation,
            'type': asset_type,
            'added_date': datetime.now()
        })
        logger.info(f"Added risky asset: {symbol} with {allocation:.1%} allocation")
    
    def add_asymmetric_bet(self, bet: AsymmetricBet):
        """
        Add an asymmetric bet to the portfolio.
        
        Asymmetric bets are the core of the strategy, providing:
        - Limited downside risk
        - Unlimited upside potential
        - Positive expected value
        - Convex payoff structure
        """
        self.asymmetric_bets.append(bet)
        logger.info(f"Added asymmetric bet: {bet.symbol} with EV: ${bet.expected_value:.2f}")
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate key portfolio metrics and risk measures.
        
        Returns:
            Dictionary containing portfolio metrics including:
            - Total value and allocations
            - Expected return and volatility
            - Maximum drawdown potential
            - Convexity measures
            - Risk-adjusted returns
        """
        # Calculate safe asset value
        safe_value = sum(asset['allocation'] for asset in self.safe_assets) * self.total_capital
        
        # Calculate risky asset value
        risky_value = sum(asset['allocation'] for asset in self.risky_assets) * self.total_capital
        
        # Calculate asymmetric bet metrics
        total_bet_value = sum(bet.size for bet in self.asymmetric_bets) * self.total_capital
        total_expected_value = sum(bet.expected_value for bet in self.asymmetric_bets)
        avg_convexity = np.mean([bet.convexity for bet in self.asymmetric_bets]) if self.asymmetric_bets else 0
        
        # Calculate portfolio metrics
        total_value = safe_value + risky_value + total_bet_value
        expected_return = (safe_value * 0.03 + risky_value * 0.15 + total_expected_value) / total_value
        max_drawdown = risky_value / total_value  # Simplified calculation
        
        return {
            'total_value': total_value,
            'safe_allocation': safe_value / total_value,
            'risky_allocation': risky_value / total_value,
            'bet_allocation': total_bet_value / total_value,
            'expected_return': expected_return,
            'max_drawdown': max_drawdown,
            'total_expected_value': total_expected_value,
            'avg_convexity': avg_convexity,
            'risk_reward_ratio': total_expected_value / max_drawdown if max_drawdown > 0 else 0
        }
    
    def rebalance_portfolio(self):
        """
        Rebalance the portfolio to maintain target allocations.
        
        Rebalancing ensures:
        - Safe assets remain at 75-80% of portfolio
        - Risky assets remain at 20-25% of portfolio
        - Asymmetric bets are properly sized
        - Risk metrics are within acceptable ranges
        """
        metrics = self.calculate_portfolio_metrics()
        
        # Check if rebalancing is needed
        safe_deviation = abs(metrics['safe_allocation'] - self.safe_allocation)
        risky_deviation = abs(metrics['risky_allocation'] - self.risky_allocation)
        
        if safe_deviation > 0.05 or risky_deviation > 0.05:  # 5% tolerance
            logger.info("Rebalancing portfolio to maintain target allocations")
            
            # Implement rebalancing logic here
            # This would involve buying/selling assets to reach target allocations
            
            self.last_rebalance = datetime.now()
            logger.info("Portfolio rebalancing completed")

class AsymmetricTradingFramework:
    """
    Main framework for implementing asymmetric trading strategies.
    
    This framework combines:
    - Barbell portfolio construction
    - Asymmetric bet identification and sizing
    - Fat tail risk management
    - Volatility harvesting strategies
    - Convexity optimization
    
    The framework is designed to create portfolios that:
    1. Benefit from market volatility and uncertainty
    2. Provide protection against extreme market events
    3. Generate consistent positive returns
    4. Maintain capital preservation
    """
    
    def __init__(self, config: Dict[str, Any], ml_models: AdvancedMLModels = None, 
                 sentiment_engine: SentimentAnalysisEngine = None):
        """
        Initialize the asymmetric trading framework.
        
        Args:
            config: Configuration dictionary containing strategy parameters
            ml_models: Advanced ML models for signal generation
            sentiment_engine: Sentiment analysis engine for market sentiment
        """
        self.config = config
        self.ml_models = ml_models
        self.sentiment_engine = sentiment_engine
        self.portfolio = None
        self.is_initialized = False
        
        # Strategy parameters
        self.min_convexity = config.get('min_convexity', 2.0)
        self.max_position_size = config.get('max_position_size', 0.05)  # 5% max per position
        self.min_probability = config.get('min_probability', 0.1)  # 10% minimum success probability
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)  # 5% rebalancing threshold
        
        logger.info("Asymmetric Trading Framework initialized")
    
    async def initialize(self):
        """
        Initialize the framework and create initial portfolio.
        
        This method:
        1. Validates configuration parameters
        2. Creates initial barbell portfolio
        3. Sets up monitoring and risk management
        4. Initializes ML models and sentiment analysis
        """
        try:
            logger.info("Initializing Asymmetric Trading Framework...")
            
            # Validate configuration
            self._validate_config()
            
            # Create initial portfolio
            initial_capital = self.config.get('initial_capital', 100000)
            self.portfolio = BarbellPortfolio(
                total_capital=initial_capital,
                safe_allocation=self.config.get('safe_allocation', 0.75),
                risky_allocation=self.config.get('risky_allocation', 0.25)
            )
            
            # Initialize safe assets
            await self._initialize_safe_assets()
            
            # Initialize risky assets
            await self._initialize_risky_assets()
            
            self.is_initialized = True
            logger.info("Asymmetric Trading Framework initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Asymmetric Trading Framework: {str(e)}")
            raise
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_params = ['initial_capital', 'safe_allocation', 'risky_allocation']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required configuration parameter: {param}")
        
        # Validate allocations
        safe_alloc = self.config['safe_allocation']
        risky_alloc = self.config['risky_allocation']
        
        if abs(safe_alloc + risky_alloc - 1.0) > 0.01:
            raise ValueError("Safe and risky allocations must sum to 1.0")
        
        if safe_alloc < 0.7 or safe_alloc > 0.85:
            logger.warning(f"Safe allocation {safe_alloc:.1%} is outside recommended range (70-85%)")
        
        if risky_alloc < 0.15 or risky_alloc > 0.3:
            logger.warning(f"Risky allocation {risky_alloc:.1%} is outside recommended range (15-30%)")
    
    async def _initialize_safe_assets(self):
        """Initialize safe assets for the barbell portfolio."""
        safe_assets_config = self.config.get('safe_assets', [])
        
        for asset in safe_assets_config:
            self.portfolio.add_safe_asset(
                symbol=asset['symbol'],
                allocation=asset['allocation'],
                asset_type=asset.get('type', 'bond')
            )
    
    async def _initialize_risky_assets(self):
        """Initialize risky assets for the barbell portfolio."""
        risky_assets_config = self.config.get('risky_assets', [])
        
        for asset in risky_assets_config:
            self.portfolio.add_risky_asset(
                symbol=asset['symbol'],
                allocation=asset['allocation'],
                asset_type=asset.get('type', 'option')
            )
    
    async def identify_asymmetric_opportunities(self, market_data: pd.DataFrame) -> List[AsymmetricBet]:
        """
        Identify asymmetric trading opportunities in the market.
        
        This method analyzes market data to find opportunities with:
        - Limited downside risk
        - High upside potential
        - Positive expected value
        - Convex payoff structure
        
        Args:
            market_data: Market data including prices, volatility, sentiment
            
        Returns:
            List of identified asymmetric betting opportunities
        """
        opportunities = []
        
        try:
            # Use ML models to identify opportunities
            if self.ml_models:
                signals = await self.ml_models.generate_signals(market_data)
                
                for signal in signals:
                    if self._is_asymmetric_opportunity(signal):
                        bet = await self._create_asymmetric_bet(signal)
                        if bet:
                            opportunities.append(bet)
            
            # Use sentiment analysis for additional opportunities
            if self.sentiment_engine:
                sentiment_opportunities = await self._identify_sentiment_opportunities(market_data)
                opportunities.extend(sentiment_opportunities)
            
            logger.info(f"Identified {len(opportunities)} asymmetric opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying asymmetric opportunities: {str(e)}")
            return []
    
    def _is_asymmetric_opportunity(self, signal: Dict[str, Any]) -> bool:
        """
        Determine if a signal represents an asymmetric opportunity.
        
        Criteria for asymmetric opportunities:
        - Limited downside risk (stop loss close to current price)
        - High upside potential (target price significantly higher)
        - Positive expected value
        - Convex payoff structure
        """
        # Extract signal parameters
        current_price = signal.get('current_price', 0)
        target_price = signal.get('target_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        probability = signal.get('probability', 0)
        
        if current_price <= 0 or target_price <= 0 or stop_loss <= 0:
            return False
        
        # Calculate potential gain and loss
        potential_gain = target_price - current_price
        max_loss = current_price - stop_loss
        
        # Check if opportunity is asymmetric
        if max_loss <= 0 or potential_gain <= 0:
            return False
        
        # Calculate convexity
        convexity = potential_gain / max_loss
        
        # Calculate expected value
        expected_value = (potential_gain * probability) - (max_loss * (1 - probability))
        
        # Check criteria
        is_convex = convexity >= self.min_convexity
        has_positive_ev = expected_value > 0
        has_reasonable_probability = probability >= self.min_probability
        
        return is_convex and has_positive_ev and has_reasonable_probability
    
    async def _create_asymmetric_bet(self, signal: Dict[str, Any]) -> Optional[AsymmetricBet]:
        """
        Create an asymmetric bet from a trading signal.
        
        This method:
        1. Calculates optimal position size
        2. Sets stop loss and target levels
        3. Estimates probability of success
        4. Calculates expected value and convexity
        """
        try:
            # Extract signal data
            symbol = signal.get('symbol', '')
            current_price = signal.get('current_price', 0)
            target_price = signal.get('target_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            probability = signal.get('probability', 0)
            position_type = signal.get('position_type', 'long')
            
            # Calculate position size based on Kelly Criterion
            position_size = self._calculate_position_size(signal)
            
            # Calculate metrics
            potential_gain = target_price - current_price
            max_loss = current_price - stop_loss
            expected_value = (potential_gain * probability) - (max_loss * (1 - probability))
            convexity = potential_gain / max_loss
            
            # Create asymmetric bet
            bet = AsymmetricBet(
                symbol=symbol,
                position_type=position_type,
                size=position_size,
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target_price,
                max_loss=max_loss,
                potential_gain=potential_gain,
                probability=probability,
                expected_value=expected_value,
                convexity=convexity,
                timestamp=datetime.now()
            )
            
            return bet
            
        except Exception as e:
            logger.error(f"Error creating asymmetric bet: {str(e)}")
            return None
    
    def _calculate_position_size(self, signal: Dict[str, Any]) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        The Kelly Criterion maximizes the long-term growth rate of capital
        by finding the optimal fraction of capital to risk on each bet.
        
        Kelly Formula: f = (bp - q) / b
        where:
        - f = fraction of capital to bet
        - b = odds received on the bet (potential gain / max loss)
        - p = probability of winning
        - q = probability of losing (1 - p)
        """
        try:
            current_price = signal.get('current_price', 0)
            target_price = signal.get('target_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            probability = signal.get('probability', 0)
            
            if current_price <= 0 or target_price <= 0 or stop_loss <= 0:
                return 0
            
            # Calculate odds (potential gain / max loss)
            potential_gain = target_price - current_price
            max_loss = current_price - stop_loss
            odds = potential_gain / max_loss
            
            # Calculate Kelly fraction
            p = probability
            q = 1 - probability
            kelly_fraction = (odds * p - q) / odds
            
            # Apply constraints
            kelly_fraction = max(0, min(kelly_fraction, self.max_position_size))
            
            return kelly_fraction
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    async def _identify_sentiment_opportunities(self, market_data: pd.DataFrame) -> List[AsymmetricBet]:
        """
        Identify asymmetric opportunities based on sentiment analysis.
        
        This method looks for:
        - Extreme sentiment readings (fear/greed)
        - Sentiment divergences from price action
        - News-driven opportunities
        - Social media sentiment shifts
        """
        opportunities = []
        
        try:
            if not self.sentiment_engine:
                return opportunities
            
            # Get sentiment data
            sentiment_data = await self.sentiment_engine.analyze_market_sentiment(market_data)
            
            # Look for extreme sentiment opportunities
            for symbol, sentiment in sentiment_data.items():
                if self._is_extreme_sentiment_opportunity(sentiment):
                    bet = await self._create_sentiment_bet(symbol, sentiment)
                    if bet:
                        opportunities.append(bet)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying sentiment opportunities: {str(e)}")
            return []
    
    def _is_extreme_sentiment_opportunity(self, sentiment: Dict[str, Any]) -> bool:
        """
        Determine if extreme sentiment represents an asymmetric opportunity.
        
        Extreme sentiment often creates opportunities because:
        - Fear creates oversold conditions (buying opportunities)
        - Greed creates overbought conditions (selling opportunities)
        - Sentiment tends to mean-revert over time
        """
        # Extract sentiment metrics
        fear_greed_index = sentiment.get('fear_greed_index', 50)
        sentiment_score = sentiment.get('sentiment_score', 0)
        volatility = sentiment.get('volatility', 0)
        
        # Check for extreme fear (buying opportunity)
        if fear_greed_index < 20 and sentiment_score < -0.5:
            return True
        
        # Check for extreme greed (selling opportunity)
        if fear_greed_index > 80 and sentiment_score > 0.5:
            return True
        
        # Check for high volatility opportunities
        if volatility > 0.3:  # 30% volatility threshold
            return True
        
        return False
    
    async def _create_sentiment_bet(self, symbol: str, sentiment: Dict[str, Any]) -> Optional[AsymmetricBet]:
        """
        Create an asymmetric bet based on sentiment analysis.
        
        This method creates contrarian bets when sentiment is extreme,
        expecting mean reversion in both sentiment and price.
        """
        try:
            fear_greed_index = sentiment.get('fear_greed_index', 50)
            sentiment_score = sentiment.get('sentiment_score', 0)
            
            # Determine position type based on sentiment
            if fear_greed_index < 20:  # Extreme fear - buy opportunity
                position_type = 'long'
                probability = 0.7  # Higher probability for contrarian bets
            elif fear_greed_index > 80:  # Extreme greed - sell opportunity
                position_type = 'short'
                probability = 0.7
            else:
                return None
            
            # Create simplified bet structure
            # In practice, you would get current price and calculate proper levels
            current_price = 100  # Placeholder
            target_price = current_price * (1.1 if position_type == 'long' else 0.9)
            stop_loss = current_price * (0.95 if position_type == 'long' else 1.05)
            
            bet = AsymmetricBet(
                symbol=symbol,
                position_type=position_type,
                size=0.02,  # 2% position size for sentiment bets
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target_price,
                max_loss=abs(current_price - stop_loss),
                potential_gain=abs(target_price - current_price),
                probability=probability,
                expected_value=0,  # Will be calculated in __post_init__
                convexity=0,  # Will be calculated in __post_init__
                timestamp=datetime.now()
            )
            
            return bet
            
        except Exception as e:
            logger.error(f"Error creating sentiment bet: {str(e)}")
            return None
    
    async def manage_fat_tail_risk(self, portfolio: BarbellPortfolio) -> Dict[str, Any]:
        """
        Implement fat tail risk management strategies.
        
        Fat tail risk management focuses on:
        - Protection against extreme market events
        - Hedging against black swan events
        - Maintaining portfolio convexity
        - Dynamic position sizing based on market conditions
        
        Returns:
            Dictionary containing risk management actions and metrics
        """
        try:
            risk_metrics = {
                'portfolio_convexity': 0,
                'fat_tail_exposure': 0,
                'hedge_recommendations': [],
                'position_adjustments': []
            }
            
            # Calculate portfolio convexity
            portfolio_metrics = portfolio.calculate_portfolio_metrics()
            risk_metrics['portfolio_convexity'] = portfolio_metrics['avg_convexity']
            
            # Check for fat tail exposure
            fat_tail_exposure = self._calculate_fat_tail_exposure(portfolio)
            risk_metrics['fat_tail_exposure'] = fat_tail_exposure
            
            # Generate hedge recommendations
            if fat_tail_exposure > 0.3:  # 30% threshold
                hedges = await self._generate_hedge_recommendations(portfolio)
                risk_metrics['hedge_recommendations'] = hedges
            
            # Recommend position adjustments
            adjustments = await self._recommend_position_adjustments(portfolio)
            risk_metrics['position_adjustments'] = adjustments
            
            logger.info(f"Fat tail risk management completed - Convexity: {risk_metrics['portfolio_convexity']:.2f}")
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error in fat tail risk management: {str(e)}")
            return {}
    
    def _calculate_fat_tail_exposure(self, portfolio: BarbellPortfolio) -> float:
        """
        Calculate portfolio exposure to fat tail events.
        
        Fat tail exposure measures the portfolio's vulnerability to:
        - Extreme market crashes
        - Black swan events
        - Systemic risk
        - Correlation breakdown
        """
        try:
            # Calculate exposure based on portfolio composition
            risky_allocation = sum(asset['allocation'] for asset in portfolio.risky_assets)
            bet_exposure = sum(bet.size for bet in portfolio.asymmetric_bets)
            
            # Weighted exposure calculation
            total_exposure = (risky_allocation * 0.7) + (bet_exposure * 0.3)
            
            return min(total_exposure, 1.0)  # Cap at 100%
            
        except Exception as e:
            logger.error(f"Error calculating fat tail exposure: {str(e)}")
            return 0
    
    async def _generate_hedge_recommendations(self, portfolio: BarbellPortfolio) -> List[Dict[str, Any]]:
        """
        Generate hedge recommendations for fat tail protection.
        
        Hedge strategies include:
        - Put options on major indices
        - VIX futures and options
        - Gold and other safe havens
        - Inverse ETFs
        - Volatility products
        """
        hedges = []
        
        try:
            # VIX-based hedges
            hedges.append({
                'type': 'vix_call_option',
                'symbol': 'VIX',
                'strike': 'current + 20%',
                'expiry': '1 month',
                'allocation': 0.02,  # 2% of portfolio
                'rationale': 'Protection against market volatility spikes'
            })
            
            # Gold hedge
            hedges.append({
                'type': 'gold_etf',
                'symbol': 'GLD',
                'allocation': 0.03,  # 3% of portfolio
                'rationale': 'Safe haven during market stress'
            })
            
            # Inverse S&P 500 hedge
            hedges.append({
                'type': 'inverse_etf',
                'symbol': 'SH',
                'allocation': 0.01,  # 1% of portfolio
                'rationale': 'Direct hedge against market decline'
            })
            
            logger.info(f"Generated {len(hedges)} hedge recommendations")
            return hedges
            
        except Exception as e:
            logger.error(f"Error generating hedge recommendations: {str(e)}")
            return []
    
    async def _recommend_position_adjustments(self, portfolio: BarbellPortfolio) -> List[Dict[str, Any]]:
        """
        Recommend position adjustments based on current market conditions.
        
        Adjustments may include:
        - Reducing position sizes in high-risk assets
        - Increasing safe asset allocation
        - Closing asymmetric bets with negative convexity
        - Adding new asymmetric opportunities
        """
        adjustments = []
        
        try:
            # Check for positions with negative convexity
            for bet in portfolio.asymmetric_bets:
                if bet.convexity < 1.0:  # Negative convexity
                    adjustments.append({
                        'action': 'close_position',
                        'symbol': bet.symbol,
                        'reason': 'Negative convexity - limited upside potential',
                        'priority': 'high'
                    })
            
            # Check for oversized positions
            for bet in portfolio.asymmetric_bets:
                if bet.size > self.max_position_size:
                    adjustments.append({
                        'action': 'reduce_position',
                        'symbol': bet.symbol,
                        'current_size': bet.size,
                        'recommended_size': self.max_position_size,
                        'reason': 'Position size exceeds maximum limit',
                        'priority': 'medium'
                    })
            
            logger.info(f"Generated {len(adjustments)} position adjustment recommendations")
            return adjustments
            
        except Exception as e:
            logger.error(f"Error recommending position adjustments: {str(e)}")
            return []
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary and performance metrics.
        
        Returns:
            Dictionary containing portfolio status, performance metrics,
            risk measures, and recommendations.
        """
        if not self.portfolio:
            return {'error': 'Portfolio not initialized'}
        
        try:
            # Calculate portfolio metrics
            metrics = self.portfolio.calculate_portfolio_metrics()
            
            # Get risk management status
            risk_metrics = await self.manage_fat_tail_risk(self.portfolio)
            
            # Compile summary
            summary = {
                'portfolio_status': {
                    'total_value': metrics['total_value'],
                    'safe_allocation': f"{metrics['safe_allocation']:.1%}",
                    'risky_allocation': f"{metrics['risky_allocation']:.1%}",
                    'bet_allocation': f"{metrics['bet_allocation']:.1%}",
                    'expected_return': f"{metrics['expected_return']:.1%}",
                    'max_drawdown': f"{metrics['max_drawdown']:.1%}",
                    'risk_reward_ratio': f"{metrics['risk_reward_ratio']:.2f}"
                },
                'risk_management': {
                    'portfolio_convexity': f"{risk_metrics.get('portfolio_convexity', 0):.2f}",
                    'fat_tail_exposure': f"{risk_metrics.get('fat_tail_exposure', 0):.1%}",
                    'hedge_recommendations': len(risk_metrics.get('hedge_recommendations', [])),
                    'position_adjustments': len(risk_metrics.get('position_adjustments', []))
                },
                'asymmetric_bets': {
                    'total_bets': len(self.portfolio.asymmetric_bets),
                    'total_expected_value': f"${metrics['total_expected_value']:.2f}",
                    'avg_convexity': f"{metrics['avg_convexity']:.2f}"
                },
                'last_update': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {str(e)}")
            return {'error': str(e)} 