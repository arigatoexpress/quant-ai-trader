"""
Grok 4 Data Integration Framework
Ensures all data sources are properly integrated and optimized for Grok 4 analysis
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import aiohttp
import websockets
from dataclasses import dataclass, asdict
import hashlib
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Represents a data source with metadata"""
    name: str
    type: str  # market, sentiment, onchain, news, social
    url: str
    api_key: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    last_update: Optional[datetime] = None
    reliability_score: float = 0.8
    latency_ms: int = 100

@dataclass
class IntegratedDataPoint:
    """Represents an integrated data point from multiple sources"""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    market_cap: float
    sentiment_score: float
    sentiment_confidence: float
    social_volume: float
    news_sentiment: float
    onchain_metrics: Dict[str, float]
    technical_indicators: Dict[str, float]
    macro_indicators: Dict[str, float]
    data_quality_score: float
    source_count: int
    metadata: Dict[str, Any]

class DataQualityAssessor:
    """Assesses and scores data quality from various sources"""
    
    def __init__(self):
        self.quality_thresholds = {
            'price_consistency': 0.95,
            'volume_reliability': 0.8,
            'sentiment_confidence': 0.7,
            'onchain_freshness': 300,  # seconds
            'news_relevance': 0.6
        }
    
    def assess_price_quality(self, prices: List[float]) -> float:
        """Assess price data quality"""
        if len(prices) < 2:
            return 0.0
        
        # Check for price consistency
        price_changes = np.diff(prices)
        extreme_changes = np.abs(price_changes) > np.std(price_changes) * 3
        
        # Check for zero or negative prices
        invalid_prices = sum(1 for p in prices if p <= 0)
        
        quality_score = 1.0
        quality_score -= extreme_changes.mean() * 0.3
        quality_score -= invalid_prices / len(prices) * 0.5
        
        return max(0.0, quality_score)
    
    def assess_volume_quality(self, volumes: List[float]) -> float:
        """Assess volume data quality"""
        if len(volumes) < 2:
            return 0.0
        
        # Check for volume consistency
        volume_changes = np.diff(volumes)
        zero_volumes = sum(1 for v in volumes if v <= 0)
        
        quality_score = 1.0
        quality_score -= zero_volumes / len(volumes) * 0.7
        quality_score -= (np.std(volume_changes) / np.mean(volumes)) * 0.3
        
        return max(0.0, quality_score)
    
    def assess_sentiment_quality(self, sentiments: List[float], confidences: List[float]) -> float:
        """Assess sentiment data quality"""
        if len(sentiments) == 0:
            return 0.0
        
        # Check sentiment range
        valid_sentiments = [s for s in sentiments if -1 <= s <= 1]
        range_score = len(valid_sentiments) / len(sentiments)
        
        # Check confidence scores
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        quality_score = range_score * 0.6 + avg_confidence * 0.4
        return quality_score
    
    def assess_onchain_quality(self, metrics: Dict[str, List[float]], timestamps: List[datetime]) -> float:
        """Assess onchain data quality"""
        if not metrics or not timestamps:
            return 0.0
        
        # Check data freshness
        now = datetime.now()
        time_diffs = [(now - ts).total_seconds() for ts in timestamps]
        freshness_score = 1.0 - min(max(time_diffs) / 3600, 1.0)  # 1 hour max
        
        # Check data completeness
        completeness_score = 0.0
        for metric_name, values in metrics.items():
            if values and len(values) > 0:
                completeness_score += 1.0
        
        completeness_score /= len(metrics) if metrics else 1.0
        
        quality_score = freshness_score * 0.7 + completeness_score * 0.3
        return quality_score
    
    def calculate_overall_quality(self, 
                                price_quality: float,
                                volume_quality: float,
                                sentiment_quality: float,
                                onchain_quality: float) -> float:
        """Calculate overall data quality score"""
        weights = {
            'price': 0.4,
            'volume': 0.2,
            'sentiment': 0.2,
            'onchain': 0.2
        }
        
        overall_quality = (
            price_quality * weights['price'] +
            volume_quality * weights['volume'] +
            sentiment_quality * weights['sentiment'] +
            onchain_quality * weights['onchain']
        )
        
        return overall_quality

class DataNormalizer:
    """Normalizes and standardizes data from different sources"""
    
    def __init__(self):
        self.scalers = {}
        self.normalization_methods = {
            'z_score': self.z_score_normalize,
            'min_max': self.min_max_normalize,
            'robust': self.robust_normalize,
            'log': self.log_normalize
        }
    
    def z_score_normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    
    def min_max_normalize(self, data: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    
    def robust_normalize(self, data: np.ndarray) -> np.ndarray:
        """Robust normalization using median and IQR"""
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return np.zeros_like(data)
        return (data - median) / iqr
    
    def log_normalize(self, data: np.ndarray) -> np.ndarray:
        """Log normalization for skewed data"""
        # Add small constant to avoid log(0)
        data_positive = data - np.min(data) + 1e-8
        return np.log(data_positive)
    
    def normalize_feature(self, 
                         data: np.ndarray, 
                         feature_name: str, 
                         method: str = 'z_score') -> np.ndarray:
        """Normalize a specific feature"""
        if method not in self.normalization_methods:
            method = 'z_score'
        
        normalized = self.normalization_methods[method](data)
        
        # Store scaler information for later use
        self.scalers[feature_name] = {
            'method': method,
            'original_data': data,
            'normalized_data': normalized
        }
        
        return normalized
    
    def denormalize_feature(self, 
                           normalized_data: np.ndarray, 
                           feature_name: str) -> np.ndarray:
        """Denormalize a feature back to original scale"""
        if feature_name not in self.scalers:
            return normalized_data
        
        scaler_info = self.scalers[feature_name]
        method = scaler_info['method']
        original_data = scaler_info['original_data']
        
        if method == 'z_score':
            mean = np.mean(original_data)
            std = np.std(original_data)
            return normalized_data * std + mean
        elif method == 'min_max':
            min_val = np.min(original_data)
            max_val = np.max(original_data)
            return normalized_data * (max_val - min_val) + min_val
        elif method == 'robust':
            median = np.median(original_data)
            q75, q25 = np.percentile(original_data, [75, 25])
            iqr = q75 - q25
            return normalized_data * iqr + median
        elif method == 'log':
            min_val = np.min(original_data)
            return np.exp(normalized_data) + min_val - 1e-8
        
        return normalized_data

class Grok4DataFormatter:
    """Formats data specifically for optimal Grok 4 analysis"""
    
    def __init__(self):
        self.required_features = [
            'price', 'volume', 'market_cap', 'sentiment_score', 'social_volume',
            'news_sentiment', 'active_addresses', 'transaction_count', 'rsi', 'macd',
            'bollinger_upper', 'bollinger_lower', 'volatility', 'momentum'
        ]
        
        self.feature_importance = {
            'price': 1.0,
            'volume': 0.9,
            'sentiment_score': 0.8,
            'market_cap': 0.7,
            'social_volume': 0.6,
            'news_sentiment': 0.6,
            'active_addresses': 0.5,
            'transaction_count': 0.5,
            'rsi': 0.4,
            'macd': 0.4,
            'volatility': 0.3,
            'momentum': 0.3
        }
    
    def format_for_grok4(self, 
                        integrated_data: List[IntegratedDataPoint],
                        include_metadata: bool = True) -> Dict[str, Any]:
        """Format integrated data for Grok 4 analysis"""
        
        if not integrated_data:
            return {}
        
        # Extract features
        features = {}
        for feature in self.required_features:
            values = []
            for data_point in integrated_data:
                if hasattr(data_point, feature):
                    values.append(getattr(data_point, feature))
                elif feature in data_point.onchain_metrics:
                    values.append(data_point.onchain_metrics[feature])
                elif feature in data_point.technical_indicators:
                    values.append(data_point.technical_indicators[feature])
                elif feature in data_point.macro_indicators:
                    values.append(data_point.macro_indicators[feature])
                else:
                    values.append(0.0)  # Default value
            
            features[feature] = values
        
        # Create feature matrix
        feature_matrix = []
        for i in range(len(integrated_data)):
            row = []
            for feature in self.required_features:
                if feature in features and i < len(features[feature]):
                    row.append(features[feature][i])
                else:
                    row.append(0.0)
            feature_matrix.append(row)
        
        # Calculate feature weights based on importance
        feature_weights = [self.feature_importance.get(feature, 0.5) for feature in self.required_features]
        
        # Prepare metadata
        metadata = {}
        if include_metadata:
            metadata = {
                'symbols': list(set(dp.symbol for dp in integrated_data)),
                'time_range': {
                    'start': min(dp.timestamp for dp in integrated_data).isoformat(),
                    'end': max(dp.timestamp for dp in integrated_data).isoformat()
                },
                'data_points': len(integrated_data),
                'avg_quality_score': np.mean([dp.data_quality_score for dp in integrated_data]),
                'feature_weights': dict(zip(self.required_features, feature_weights))
            }
        
        # Format for Grok 4
        grok4_format = {
            'features': {
                'matrix': feature_matrix,
                'names': self.required_features,
                'weights': feature_weights
            },
            'metadata': metadata,
            'timestamps': [dp.timestamp.isoformat() for dp in integrated_data],
            'symbols': [dp.symbol for dp in integrated_data],
            'quality_scores': [dp.data_quality_score for dp in integrated_data]
        }
        
        return grok4_format
    
    def create_grok4_prompt(self, 
                           formatted_data: Dict[str, Any],
                           analysis_type: str = 'trading_decision') -> str:
        """Create optimized prompt for Grok 4 analysis"""
        
        if not formatted_data:
            return ""
        
        # Extract key information
        features = formatted_data.get('features', {})
        metadata = formatted_data.get('metadata', {})
        quality_scores = formatted_data.get('quality_scores', [])
        
        # Calculate data quality
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        # Create context
        context = f"""
        You are analyzing cryptocurrency market data for asymmetric trading opportunities.
        
        Data Quality: {avg_quality:.2f}/1.0
        Symbols: {', '.join(metadata.get('symbols', []))}
        Time Range: {metadata.get('time_range', {}).get('start', 'N/A')} to {metadata.get('time_range', {}).get('end', 'N/A')}
        Data Points: {metadata.get('data_points', 0)}
        
        Key Features (with importance weights):
        """
        
        for feature, weight in metadata.get('feature_weights', {}).items():
            context += f"- {feature}: {weight:.2f}\n"
        
        # Add analysis instructions
        if analysis_type == 'trading_decision':
            context += """
            
            Please analyze this data and provide:
            1. Asymmetric trading opportunities (high reward, limited risk)
            2. Barbell portfolio recommendations (90% safe, 10% risky)
            3. Fat tail risk assessment
            4. Specific entry/exit points with confidence levels
            5. Position sizing recommendations
            
            Focus on convex payoffs and optionality in market inefficiencies.
            """
        elif analysis_type == 'risk_assessment':
            context += """
            
            Please assess:
            1. Fat tail risks and black swan probabilities
            2. Portfolio vulnerability to extreme events
            3. Required hedge positions
            4. Risk-adjusted return expectations
            5. Maximum drawdown scenarios
            """
        
        return context

class Grok4DataIntegration:
    """Orchestrates data collection, integration, and formatting for Grok 4"""
    
    def __init__(self):
        self.data_sources = {}
        self.quality_assessor = DataQualityAssessor()
        self.normalizer = DataNormalizer()
        self.formatter = Grok4DataFormatter()
        self.integrated_data_cache = {}
        self.last_update = {}
        
    def add_data_source(self, source: DataSource):
        """Add a data source to the orchestrator"""
        self.data_sources[source.name] = source
        logger.info(f"Added data source: {source.name} ({source.type})")
    
    async def collect_market_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect market data from multiple sources"""
        market_data = {}
        
        for symbol in symbols:
            symbol_data = []
            
            # Collect from multiple sources for redundancy
            for source_name, source in self.data_sources.items():
                if source.type == 'market':
                    try:
                        # Simulate data collection (replace with actual API calls)
                        data = await self._fetch_market_data(source, symbol)
                        if data is not None:
                            symbol_data.append(data)
                    except Exception as e:
                        logger.warning(f"Failed to collect market data from {source_name}: {e}")
            
            # Merge and validate data
            if symbol_data:
                merged_data = self._merge_market_data(symbol_data)
                market_data[symbol] = merged_data
        
        return market_data
    
    async def collect_sentiment_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect sentiment data from multiple sources"""
        sentiment_data = {}
        
        for symbol in symbols:
            symbol_data = []
            
            for source_name, source in self.data_sources.items():
                if source.type == 'sentiment':
                    try:
                        data = await self._fetch_sentiment_data(source, symbol)
                        if data is not None:
                            symbol_data.append(data)
                    except Exception as e:
                        logger.warning(f"Failed to collect sentiment data from {source_name}: {e}")
            
            if symbol_data:
                merged_data = self._merge_sentiment_data(symbol_data)
                sentiment_data[symbol] = merged_data
        
        return sentiment_data
    
    async def collect_onchain_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect onchain data from multiple sources"""
        onchain_data = {}
        
        for symbol in symbols:
            symbol_data = []
            
            for source_name, source in self.data_sources.items():
                if source.type == 'onchain':
                    try:
                        data = await self._fetch_onchain_data(source, symbol)
                        if data is not None:
                            symbol_data.append(data)
                    except Exception as e:
                        logger.warning(f"Failed to collect onchain data from {source_name}: {e}")
            
            if symbol_data:
                merged_data = self._merge_onchain_data(symbol_data)
                onchain_data[symbol] = merged_data
        
        return onchain_data
    
    async def collect_news_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect news data from multiple sources"""
        news_data = {}
        
        for symbol in symbols:
            symbol_data = []
            
            for source_name, source in self.data_sources.items():
                if source.type == 'news':
                    try:
                        data = await self._fetch_news_data(source, symbol)
                        if data is not None:
                            symbol_data.append(data)
                    except Exception as e:
                        logger.warning(f"Failed to collect news data from {source_name}: {e}")
            
            if symbol_data:
                merged_data = self._merge_news_data(symbol_data)
                news_data[symbol] = merged_data
        
        return news_data
    
    async def collect_social_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Collect social media data from multiple sources"""
        social_data = {}
        
        for symbol in symbols:
            symbol_data = []
            
            for source_name, source in self.data_sources.items():
                if source.type == 'social':
                    try:
                        data = await self._fetch_social_data(source, symbol)
                        if data is not None:
                            symbol_data.append(data)
                    except Exception as e:
                        logger.warning(f"Failed to collect social data from {source_name}: {e}")
            
            if symbol_data:
                merged_data = self._merge_social_data(symbol_data)
                social_data[symbol] = merged_data
        
        return social_data
    
    async def integrate_all_data(self, 
                               symbols: List[str],
                               time_window: timedelta = timedelta(hours=24)) -> Dict[str, List[IntegratedDataPoint]]:
        """Integrate all data sources for comprehensive analysis"""
        
        integrated_data = {}
        
        # Collect data from all sources
        market_data = await self.collect_market_data(symbols)
        sentiment_data = await self.collect_sentiment_data(symbols)
        onchain_data = await self.collect_onchain_data(symbols)
        news_data = await self.collect_news_data(symbols)
        social_data = await self.collect_social_data(symbols)
        
        # Integrate data for each symbol
        for symbol in symbols:
            symbol_integrated = []
            
            # Get data for this symbol
            symbol_market = market_data.get(symbol, pd.DataFrame())
            symbol_sentiment = sentiment_data.get(symbol, pd.DataFrame())
            symbol_onchain = onchain_data.get(symbol, pd.DataFrame())
            symbol_news = news_data.get(symbol, pd.DataFrame())
            symbol_social = social_data.get(symbol, pd.DataFrame())
            
            # Align timestamps and integrate
            if not symbol_market.empty:
                for idx, row in symbol_market.iterrows():
                    # Ensure timestamp is a single datetime
                    if isinstance(idx, datetime):
                        timestamp = idx
                    elif isinstance(idx, (str, int, float)):
                        try:
                            timestamp = pd.Timestamp(idx).to_pydatetime()
                        except Exception:
                            timestamp = datetime.now()
                    else:
                        timestamp = datetime.now()
                    # Get corresponding data from other sources
                    sentiment_row = symbol_sentiment.loc[timestamp] if timestamp in symbol_sentiment.index else None
                    onchain_row = symbol_onchain.loc[timestamp] if timestamp in symbol_onchain.index else None
                    news_row = symbol_news.loc[timestamp] if timestamp in symbol_news.index else None
                    social_row = symbol_social.loc[timestamp] if timestamp in symbol_social.index else None
                    # Create integrated data point
                    integrated_point = IntegratedDataPoint(
                        timestamp=timestamp,
                        symbol=symbol,
                        price=float(row.get('close', 0.0) or 0.0),
                        volume=float(row.get('volume', 0.0) or 0.0),
                        market_cap=float(row.get('market_cap', 0.0) or 0.0),
                        sentiment_score=float(sentiment_row.get('sentiment_score', 0.0) if sentiment_row is not None else 0.0),
                        sentiment_confidence=float(sentiment_row.get('confidence', 0.0) if sentiment_row is not None else 0.0),
                        social_volume=float(social_row.get('volume', 0.0) if social_row is not None else 0.0),
                        news_sentiment=float(news_row.get('sentiment_score', 0.0) if news_row is not None else 0.0),
                        onchain_metrics=self._extract_onchain_metrics(onchain_row) if onchain_row is not None else {},
                        technical_indicators=self._calculate_technical_indicators(symbol_market, timestamp),
                        macro_indicators=self._get_macro_indicators(timestamp),
                        data_quality_score=float(self._calculate_data_quality(symbol_market, symbol_sentiment, symbol_onchain, timestamp)),
                        source_count=self._count_data_sources(symbol_market, symbol_sentiment, symbol_onchain, symbol_news, symbol_social, timestamp),
                        metadata={
                            'market_data': row.to_dict() if hasattr(row, 'to_dict') else {},
                            'sentiment_data': sentiment_row.to_dict() if sentiment_row is not None and hasattr(sentiment_row, 'to_dict') else {},
                            'onchain_data': onchain_row.to_dict() if onchain_row is not None and hasattr(onchain_row, 'to_dict') else {},
                            'news_data': news_row.to_dict() if news_row is not None and hasattr(news_row, 'to_dict') else {},
                            'social_data': social_row.to_dict() if social_row is not None and hasattr(social_row, 'to_dict') else {}
                        }
                    )
                    symbol_integrated.append(integrated_point)
            integrated_data[symbol] = symbol_integrated
        
        return integrated_data
    
    async def prepare_grok4_analysis(self, 
                                   integrated_data: Dict[str, List[IntegratedDataPoint]],
                                   analysis_type: str = 'trading_decision') -> Dict[str, Any]:
        """Prepare integrated data for Grok 4 analysis"""
        
        # Flatten all data points
        all_data_points = []
        for symbol_data in integrated_data.values():
            all_data_points.extend(symbol_data)
        
        # Sort by timestamp
        all_data_points.sort(key=lambda x: x.timestamp)
        
        # Format for Grok 4
        formatted_data = self.formatter.format_for_grok4(all_data_points)
        
        # Create Grok 4 prompt
        grok4_prompt = self.formatter.create_grok4_prompt(formatted_data, analysis_type)
        
        return {
            'formatted_data': formatted_data,
            'grok4_prompt': grok4_prompt,
            'analysis_type': analysis_type,
            'data_summary': {
                'total_data_points': len(all_data_points),
                'symbols': list(integrated_data.keys()),
                'time_range': {
                    'start': min(dp.timestamp for dp in all_data_points).isoformat() if all_data_points else None,
                    'end': max(dp.timestamp for dp in all_data_points).isoformat() if all_data_points else None
                },
                'avg_quality_score': np.mean([dp.data_quality_score for dp in all_data_points]) if all_data_points else 0.0
            }
        }
    
    async def scan_market_for_opportunities(self, all_symbols: List[str], min_quality: float = 0.7) -> List[Dict[str, Any]]:
        """Scan the entire market for new tokens, strategies, and asymmetric bets"""
        integrated_data = await self.integrate_all_data(all_symbols)
        all_data_points = []
        for symbol_data in integrated_data.values():
            all_data_points.extend(symbol_data)
        # Filter by data quality
        filtered = [dp for dp in all_data_points if float(dp.data_quality_score) >= min_quality]
        # Rank by convexity, sentiment, and volatility
        def safe_float(val):
            try:
                return float(val)
            except Exception:
                return 0.0
        ranked = sorted(filtered, key=lambda dp: float(abs(safe_float(dp.technical_indicators.get('momentum', 0))) * safe_float(dp.sentiment_score) * safe_float(dp.data_quality_score)), reverse=True)
        # Prepare recommendations
        recommendations = []
        for dp in ranked[:20]:  # Top 20 opportunities
            recommendations.append({
                'symbol': dp.symbol,
                'timestamp': dp.timestamp,
                'price': float(safe_float(dp.price)),
                'sentiment_score': float(safe_float(dp.sentiment_score)),
                'convexity': float(safe_float(dp.technical_indicators.get('momentum', 0))),
                'volatility': float(safe_float(dp.technical_indicators.get('volatility', 0))),
                'data_quality_score': float(safe_float(dp.data_quality_score)),
                'reason': 'High asymmetric potential',
                'details': dp.metadata
            })
        return recommendations
    
    # Helper methods for data collection (simulated)
    async def _fetch_market_data(self, source: DataSource, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch market data from a source (simulated)"""
        # Simulate API call delay
        await asyncio.sleep(0.1)
        
        # Generate sample market data
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='1H')
        data = pd.DataFrame({
            'open': np.random.normal(100, 10, len(dates)),
            'high': np.random.normal(105, 10, len(dates)),
            'low': np.random.normal(95, 10, len(dates)),
            'close': np.random.normal(100, 10, len(dates)),
            'volume': np.random.normal(1000000, 200000, len(dates)),
            'market_cap': np.random.normal(1000000000, 100000000, len(dates))
        }, index=dates)
        
        return data
    
    async def _fetch_sentiment_data(self, source: DataSource, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment data from a source (simulated)"""
        await asyncio.sleep(0.1)
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='1H')
        data = pd.DataFrame({
            'sentiment_score': np.random.normal(0.6, 0.2, len(dates)),
            'confidence': np.random.normal(0.7, 0.1, len(dates)),
            'volume': np.random.normal(1000, 200, len(dates))
        }, index=dates)
        
        return data
    
    async def _fetch_onchain_data(self, source: DataSource, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch onchain data from a source (simulated)"""
        await asyncio.sleep(0.1)
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='1H')
        data = pd.DataFrame({
            'active_addresses': np.random.normal(100000, 10000, len(dates)),
            'transaction_count': np.random.normal(50000, 5000, len(dates)),
            'gas_price': np.random.normal(20, 5, len(dates)),
            'network_hashrate': np.random.normal(1000000, 100000, len(dates))
        }, index=dates)
        
        return data
    
    async def _fetch_news_data(self, source: DataSource, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch news data from a source (simulated)"""
        await asyncio.sleep(0.1)
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='1H')
        data = pd.DataFrame({
            'sentiment_score': np.random.normal(0.5, 0.3, len(dates)),
            'article_count': np.random.poisson(5, len(dates)),
            'headline_sentiment': np.random.normal(0.4, 0.4, len(dates))
        }, index=dates)
        
        return data
    
    async def _fetch_social_data(self, source: DataSource, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch social data from a source (simulated)"""
        await asyncio.sleep(0.1)
        
        dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='1H')
        data = pd.DataFrame({
            'volume': np.random.normal(5000, 1000, len(dates)),
            'mentions': np.random.poisson(100, len(dates)),
            'sentiment_score': np.random.normal(0.6, 0.2, len(dates))
        }, index=dates)
        
        return data
    
    def _merge_market_data(self, data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge market data from multiple sources"""
        if not data_list:
            return pd.DataFrame()
        
        # Simple merge - take the first non-empty dataset
        for data in data_list:
            if not data.empty:
                return data
        
        return pd.DataFrame()
    
    def _merge_sentiment_data(self, data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge sentiment data from multiple sources"""
        if not data_list:
            return pd.DataFrame()
        
        # Weighted average based on source reliability
        merged_data = pd.DataFrame()
        total_weight = 0
        
        for data in data_list:
            if not data.empty:
                weight = 1.0  # Could be based on source reliability
                if merged_data.empty:
                    merged_data = data * weight
                else:
                    merged_data += data * weight
                total_weight += weight
        
        if total_weight > 0:
            merged_data /= total_weight
        
        return merged_data
    
    def _merge_onchain_data(self, data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge onchain data from multiple sources"""
        return self._merge_sentiment_data(data_list)  # Same logic
    
    def _merge_news_data(self, data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge news data from multiple sources"""
        return self._merge_sentiment_data(data_list)  # Same logic
    
    def _merge_social_data(self, data_list: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge social data from multiple sources"""
        return self._merge_sentiment_data(data_list)  # Same logic
    
    def _extract_onchain_metrics(self, onchain_row) -> Dict[str, float]:
        """Extract onchain metrics from a data row"""
        if onchain_row is None:
            return {}
        
        metrics = {}
        for key, value in onchain_row.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
        
        return metrics
    
    def _calculate_technical_indicators(self, market_data: pd.DataFrame, timestamp: datetime) -> Dict[str, float]:
        """Calculate technical indicators for a specific timestamp"""
        indicators = {}
        
        if market_data.empty or timestamp not in market_data.index:
            return indicators
        
        # Get historical data up to timestamp
        historical_data = market_data.loc[:timestamp]
        
        if len(historical_data) < 14:
            return indicators
        
        # Calculate RSI
        delta = historical_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = rsi.iloc[-1] if not rsi.empty else 50.0
        
        # Calculate MACD
        ema12 = historical_data['close'].ewm(span=12).mean()
        ema26 = historical_data['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        indicators['macd'] = macd.iloc[-1] if not macd.empty else 0.0
        indicators['macd_signal'] = signal.iloc[-1] if not signal.empty else 0.0
        
        # Calculate Bollinger Bands
        sma20 = historical_data['close'].rolling(window=20).mean()
        std20 = historical_data['close'].rolling(window=20).std()
        indicators['bollinger_upper'] = sma20.iloc[-1] + (std20.iloc[-1] * 2) if not sma20.empty else 0.0
        indicators['bollinger_lower'] = sma20.iloc[-1] - (std20.iloc[-1] * 2) if not sma20.empty else 0.0
        
        # Calculate volatility
        returns = historical_data['close'].pct_change().dropna()
        indicators['volatility'] = returns.std() * np.sqrt(24) if len(returns) > 0 else 0.0  # 24-hour volatility
        
        # Calculate momentum
        indicators['momentum'] = (historical_data['close'].iloc[-1] / historical_data['close'].iloc[-5] - 1) if len(historical_data) >= 5 else 0.0
        
        return indicators
    
    def _get_macro_indicators(self, timestamp: datetime) -> Dict[str, float]:
        """Get macro indicators for a specific timestamp"""
        # Simulated macro indicators
        return {
            'vix': np.random.normal(20, 5),
            'dollar_index': np.random.normal(100, 2),
            'gold_price': np.random.normal(2000, 100),
            'oil_price': np.random.normal(80, 10),
            'interest_rate': np.random.normal(5, 1)
        }
    
    def _calculate_data_quality(self, 
                              market_data: pd.DataFrame,
                              sentiment_data: pd.DataFrame,
                              onchain_data: pd.DataFrame,
                              timestamp: datetime) -> float:
        quality_scores = []
        # Market data quality
        if not market_data.empty and timestamp in market_data.index:
            prices = [float(x) for x in market_data['close'].values.tolist()]
            volumes = [float(x) for x in market_data['volume'].values.tolist()]
            quality_scores.append(self.quality_assessor.assess_price_quality(prices))
            quality_scores.append(self.quality_assessor.assess_volume_quality(volumes))
        # Sentiment data quality
        if not sentiment_data.empty and timestamp in sentiment_data.index:
            sentiments = [float(x) for x in sentiment_data['sentiment_score'].values.tolist()]
            confidences = [float(x) for x in sentiment_data['confidence'].values.tolist()]
            quality_scores.append(self.quality_assessor.assess_sentiment_quality(sentiments, confidences))
        # Onchain data quality
        if not onchain_data.empty and timestamp in onchain_data.index:
            metrics = {col: [float(x) for x in onchain_data[col].values.tolist()] for col in onchain_data.columns}
            timestamps = onchain_data.index.tolist()
            quality_scores.append(self.quality_assessor.assess_onchain_quality(metrics, timestamps))
        return float(np.mean(quality_scores)) if quality_scores else 0.0
    
    def _count_data_sources(self, 
                           market_data: pd.DataFrame,
                           sentiment_data: pd.DataFrame,
                           onchain_data: pd.DataFrame,
                           news_data: pd.DataFrame,
                           social_data: pd.DataFrame,
                           timestamp: datetime) -> int:
        """Count how many data sources have data for a specific timestamp"""
        count = 0
        
        if not market_data.empty and timestamp in market_data.index:
            count += 1
        if not sentiment_data.empty and timestamp in sentiment_data.index:
            count += 1
        if not onchain_data.empty and timestamp in onchain_data.index:
            count += 1
        if not news_data.empty and timestamp in news_data.index:
            count += 1
        if not social_data.empty and timestamp in social_data.index:
            count += 1
        
        return count

# Example usage and testing
async def test_grok4_integration():
    """Test the Grok 4 data integration framework"""
    
    # Initialize orchestrator
    orchestrator = Grok4DataIntegration()
    
    # Add data sources
    sources = [
        DataSource("binance", "market", "https://api.binance.com"),
        DataSource("coinbase", "market", "https://api.coinbase.com"),
        DataSource("twitter_sentiment", "sentiment", "https://api.twitter.com"),
        DataSource("reddit_sentiment", "sentiment", "https://api.reddit.com"),
        DataSource("etherscan", "onchain", "https://api.etherscan.io"),
        DataSource("news_api", "news", "https://newsapi.org"),
        DataSource("social_analytics", "social", "https://api.social-analytics.com")
    ]
    
    for source in sources:
        orchestrator.add_data_source(source)
    
    # Test symbols
    symbols = ["BTC", "ETH", "SOL", "SUI"]
    
    # Integrate all data
    integrated_data = await orchestrator.integrate_all_data(symbols)
    
    # Prepare for Grok 4 analysis
    grok4_analysis = await orchestrator.prepare_grok4_analysis(integrated_data, 'trading_decision')
    
    print("Grok 4 Data Integration Results:")
    print(f"Total Data Points: {grok4_analysis['data_summary']['total_data_points']}")
    print(f"Symbols: {grok4_analysis['data_summary']['symbols']}")
    print(f"Average Quality Score: {grok4_analysis['data_summary']['avg_quality_score']:.3f}")
    print(f"Analysis Type: {grok4_analysis['analysis_type']}")
    
    return grok4_analysis

if __name__ == "__main__":
    asyncio.run(test_grok4_integration()) 