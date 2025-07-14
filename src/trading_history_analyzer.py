"""
Trading History Analyzer - Comprehensive analysis of trading patterns and behaviors
"""

import pandas as pd
import numpy as np
import json
import yaml
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from data_fetcher import DataFetcher


@dataclass
class Trade:
    """Represents a single trade"""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    value: float
    fees: float
    exchange: str
    trade_type: str  # 'MARKET', 'LIMIT', 'STOP_LOSS', etc.
    pnl: Optional[float] = None
    duration: Optional[timedelta] = None


@dataclass
class TradingSession:
    """Represents a trading session"""
    start_time: datetime
    end_time: datetime
    trades_count: int
    total_volume: float
    pnl: float
    win_rate: float
    emotions: str  # 'CALM', 'FOMO', 'FEAR', 'GREED', 'REVENGE'


@dataclass
class PerformanceMetrics:
    """Comprehensive trading performance metrics"""
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_holding_period: timedelta
    risk_reward_ratio: float
    calmar_ratio: float
    sortino_ratio: float


@dataclass
class TradingBehavior:
    """Analysis of trading psychological patterns"""
    discipline_score: float  # 0-100
    risk_management_score: float  # 0-100
    emotional_control_score: float  # 0-100
    patience_score: float  # 0-100
    consistency_score: float  # 0-100
    overtrading_tendency: float  # 0-100
    fomo_susceptibility: float  # 0-100
    revenge_trading_risk: float  # 0-100
    primary_weaknesses: List[str]
    primary_strengths: List[str]


@dataclass
class StrategyRecommendation:
    """Personalized strategy recommendation"""
    strategy_type: str
    confidence: float
    reasoning: str
    improvements: List[str]
    risk_adjustments: List[str]
    behavioral_modifications: List[str]
    expected_improvement: float


class TradingHistoryAnalyzer:
    """Comprehensive trading history analysis system"""
    
    def __init__(self, config_path=None):
        # Load configuration
        if config_path is None:
            config_path = '../config/config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        self.data_fetcher = DataFetcher()
        
        # Initialize GROK client for AI analysis
        self.grok_client = OpenAI(
            api_key=config.get('grok_api_key'),
            base_url="https://api.x.ai/v1",
        )
        
        # Trading data storage
        self.trades: List[Trade] = []
        self.trading_sessions: List[TradingSession] = []
        self.performance_metrics: Optional[PerformanceMetrics] = None
        self.trading_behavior: Optional[TradingBehavior] = None
        
        print("📊 Trading History Analyzer initialized")
    
    def import_trading_data(self, data_source: str, file_path: str = None, exchange_data: Dict = None) -> bool:
        """Import trading data from various sources"""
        try:
            if data_source == "csv":
                return self._import_from_csv(file_path)
            elif data_source == "exchange_api":
                return self._import_from_exchange(exchange_data)
            elif data_source == "manual":
                return self._import_manual_data(exchange_data)
            elif data_source == "demo":
                return self._generate_demo_data()
            else:
                print(f"❌ Unsupported data source: {data_source}")
                return False
                
        except Exception as e:
            print(f"❌ Error importing trading data: {e}")
            return False
    
    def _import_from_csv(self, file_path: str) -> bool:
        """Import trades from CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            # Expected columns: timestamp, symbol, side, quantity, price
            required_columns = ['timestamp', 'symbol', 'side', 'quantity', 'price']
            
            if not all(col in df.columns for col in required_columns):
                print(f"❌ CSV missing required columns: {required_columns}")
                return False
            
            for _, row in df.iterrows():
                trade = Trade(
                    timestamp=pd.to_datetime(row['timestamp']),
                    symbol=row['symbol'],
                    side=row['side'].upper(),
                    quantity=float(row['quantity']),
                    price=float(row['price']),
                    value=float(row['quantity']) * float(row['price']),
                    fees=float(row.get('fees', 0)),
                    exchange=row.get('exchange', 'UNKNOWN'),
                    trade_type=row.get('trade_type', 'MARKET')
                )
                self.trades.append(trade)
            
            print(f"✅ Imported {len(self.trades)} trades from CSV")
            return True
            
        except Exception as e:
            print(f"❌ Error importing CSV: {e}")
            return False
    
    def _import_from_exchange(self, exchange_data: Dict) -> bool:
        """Import trades from exchange API data"""
        # Placeholder for exchange API integration
        print("🔄 Exchange API integration not yet implemented")
        return False
    
    def _import_manual_data(self, manual_data: Dict) -> bool:
        """Import manually entered trading data"""
        try:
            trades_data = manual_data.get('trades', [])
            
            for trade_data in trades_data:
                trade = Trade(
                    timestamp=datetime.fromisoformat(trade_data['timestamp']),
                    symbol=trade_data['symbol'],
                    side=trade_data['side'].upper(),
                    quantity=float(trade_data['quantity']),
                    price=float(trade_data['price']),
                    value=float(trade_data['quantity']) * float(trade_data['price']),
                    fees=float(trade_data.get('fees', 0)),
                    exchange=trade_data.get('exchange', 'MANUAL'),
                    trade_type=trade_data.get('trade_type', 'MARKET')
                )
                self.trades.append(trade)
            
            print(f"✅ Imported {len(trades_data)} manual trades")
            return True
            
        except Exception as e:
            print(f"❌ Error importing manual data: {e}")
            return False
    
    def _generate_demo_data(self) -> bool:
        """Generate realistic demo trading data for analysis"""
        try:
            # Generate 200 realistic trades over the past 6 months
            symbols = ['BTC', 'ETH', 'SOL', 'SUI', 'SEI', 'AVAX', 'DOT', 'ATOM']
            exchanges = ['BINANCE', 'COINBASE', 'KRAKEN', 'OKX']
            
            start_date = datetime.now() - timedelta(days=180)
            
            np.random.seed(42)  # For reproducible demo data
            
            for i in range(200):
                # Random timestamp
                days_offset = np.random.randint(0, 180)
                hours_offset = np.random.randint(0, 24)
                minutes_offset = np.random.randint(0, 60)
                
                timestamp = start_date + timedelta(
                    days=days_offset, 
                    hours=hours_offset, 
                    minutes=minutes_offset
                )
                
                # Random trade parameters
                symbol = np.random.choice(symbols)
                side = np.random.choice(['BUY', 'SELL'])
                quantity = np.random.uniform(0.1, 10.0)
                
                # Realistic price based on symbol
                base_prices = {
                    'BTC': 45000, 'ETH': 2500, 'SOL': 100, 'SUI': 2.5,
                    'SEI': 0.5, 'AVAX': 25, 'DOT': 8, 'ATOM': 12
                }
                price_variation = np.random.uniform(0.8, 1.2)
                price = base_prices[symbol] * price_variation
                
                # Simulate some trading psychology patterns
                if i % 15 == 0:  # FOMO trades (larger quantities, market orders)
                    quantity *= 2
                    trade_type = 'MARKET'
                elif i % 20 == 0:  # Revenge trades (after losses)
                    quantity *= 1.5
                    trade_type = 'MARKET'
                else:
                    trade_type = np.random.choice(['MARKET', 'LIMIT'], p=[0.7, 0.3])
                
                value = quantity * price
                fees = value * 0.001  # 0.1% fees
                
                trade = Trade(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    value=value,
                    fees=fees,
                    exchange=np.random.choice(exchanges),
                    trade_type=trade_type
                )
                
                self.trades.append(trade)
            
            # Sort trades by timestamp
            self.trades.sort(key=lambda x: x.timestamp)
            
            # Calculate PnL for pairs
            self._calculate_trade_pnl()
            
            print(f"✅ Generated {len(self.trades)} demo trades for analysis")
            return True
            
        except Exception as e:
            print(f"❌ Error generating demo data: {e}")
            return False
    
    def _calculate_trade_pnl(self):
        """Calculate PnL for trade pairs"""
        position_tracker = {}
        
        for trade in self.trades:
            symbol = trade.symbol
            
            if symbol not in position_tracker:
                position_tracker[symbol] = {'quantity': 0, 'avg_price': 0, 'total_cost': 0}
            
            position = position_tracker[symbol]
            
            if trade.side == 'BUY':
                # Add to position
                total_cost = position['total_cost'] + trade.value + trade.fees
                total_quantity = position['quantity'] + trade.quantity
                
                if total_quantity > 0:
                    position['avg_price'] = total_cost / total_quantity
                    position['quantity'] = total_quantity
                    position['total_cost'] = total_cost
                
            elif trade.side == 'SELL':
                # Reduce position and calculate PnL
                if position['quantity'] > 0:
                    sell_quantity = min(trade.quantity, position['quantity'])
                    cost_basis = position['avg_price'] * sell_quantity
                    sale_proceeds = trade.price * sell_quantity - trade.fees
                    
                    trade.pnl = sale_proceeds - cost_basis
                    
                    # Update position
                    position['quantity'] -= sell_quantity
                    if position['quantity'] <= 0:
                        position_tracker[symbol] = {'quantity': 0, 'avg_price': 0, 'total_cost': 0}
                    else:
                        position['total_cost'] -= cost_basis
    
    def analyze_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            print("❌ No trades to analyze")
            return None
        
        try:
            # Basic metrics
            total_trades = len(self.trades)
            trades_with_pnl = [t for t in self.trades if t.pnl is not None]
            
            if not trades_with_pnl:
                print("⚠️ No trades with PnL data available")
                return None
            
            # Win/Loss analysis
            winning_trades = [t for t in trades_with_pnl if t.pnl > 0]
            losing_trades = [t for t in trades_with_pnl if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(trades_with_pnl) * 100
            
            # PnL analysis
            total_pnl = sum(t.pnl for t in trades_with_pnl)
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
            largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
            
            # Risk metrics
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Risk-reward ratio
            risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            # Drawdown calculation
            cumulative_pnl = []
            running_total = 0
            for trade in trades_with_pnl:
                running_total += trade.pnl
                cumulative_pnl.append(running_total)
            
            if cumulative_pnl:
                peak = cumulative_pnl[0]
                max_drawdown = 0
                for value in cumulative_pnl:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak if peak != 0 else 0
                    max_drawdown = max(max_drawdown, drawdown)
                max_drawdown *= 100  # Convert to percentage
            else:
                max_drawdown = 0
            
            # Sharpe ratio (simplified)
            if cumulative_pnl:
                returns = np.diff(cumulative_pnl)
                if len(returns) > 1:
                    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else 0
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Holding period analysis
            buy_trades = {t.symbol: t for t in self.trades if t.side == 'BUY'}
            holding_periods = []
            
            for trade in self.trades:
                if trade.side == 'SELL' and trade.symbol in buy_trades:
                    buy_trade = buy_trades[trade.symbol]
                    if trade.timestamp > buy_trade.timestamp:
                        duration = trade.timestamp - buy_trade.timestamp
                        holding_periods.append(duration)
                        trade.duration = duration
            
            avg_holding_period = np.mean(holding_periods) if holding_periods else timedelta(0)
            
            # Additional ratios
            sortino_ratio = sharpe_ratio * 1.2  # Simplified
            calmar_ratio = (total_pnl / len(trades_with_pnl)) / max_drawdown if max_drawdown > 0 else 0
            
            self.performance_metrics = PerformanceMetrics(
                total_trades=total_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                total_pnl=total_pnl,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_holding_period=avg_holding_period,
                risk_reward_ratio=risk_reward_ratio,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio
            )
            
            return self.performance_metrics
            
        except Exception as e:
            print(f"❌ Error calculating performance metrics: {e}")
            return None
    
    def analyze_trading_behavior(self) -> TradingBehavior:
        """Analyze trading psychology and behavioral patterns"""
        if not self.trades:
            print("❌ No trades to analyze")
            return None
        
        try:
            # Group trades by sessions (same day trading)
            self._identify_trading_sessions()
            
            # Calculate behavioral scores
            discipline_score = self._calculate_discipline_score()
            risk_management_score = self._calculate_risk_management_score()
            emotional_control_score = self._calculate_emotional_control_score()
            patience_score = self._calculate_patience_score()
            consistency_score = self._calculate_consistency_score()
            overtrading_tendency = self._calculate_overtrading_tendency()
            fomo_susceptibility = self._calculate_fomo_susceptibility()
            revenge_trading_risk = self._calculate_revenge_trading_risk()
            
            # Identify strengths and weaknesses
            strengths = []
            weaknesses = []
            
            scores = {
                'Discipline': discipline_score,
                'Risk Management': risk_management_score,
                'Emotional Control': emotional_control_score,
                'Patience': patience_score,
                'Consistency': consistency_score
            }
            
            # Strengths are scores > 70
            for skill, score in scores.items():
                if score > 70:
                    strengths.append(skill)
                elif score < 50:
                    weaknesses.append(skill)
            
            # Add specific behavioral weaknesses
            if overtrading_tendency > 70:
                weaknesses.append('Overtrading')
            if fomo_susceptibility > 70:
                weaknesses.append('FOMO Trading')
            if revenge_trading_risk > 70:
                weaknesses.append('Revenge Trading')
            
            self.trading_behavior = TradingBehavior(
                discipline_score=discipline_score,
                risk_management_score=risk_management_score,
                emotional_control_score=emotional_control_score,
                patience_score=patience_score,
                consistency_score=consistency_score,
                overtrading_tendency=overtrading_tendency,
                fomo_susceptibility=fomo_susceptibility,
                revenge_trading_risk=revenge_trading_risk,
                primary_strengths=strengths,
                primary_weaknesses=weaknesses
            )
            
            return self.trading_behavior
            
        except Exception as e:
            print(f"❌ Error analyzing trading behavior: {e}")
            return None
    
    def _identify_trading_sessions(self):
        """Identify and group trades into trading sessions"""
        if not self.trades:
            return
        
        current_session_trades = []
        current_date = None
        
        for trade in sorted(self.trades, key=lambda x: x.timestamp):
            trade_date = trade.timestamp.date()
            
            if current_date is None or trade_date != current_date:
                # Start new session
                if current_session_trades:
                    self._create_trading_session(current_session_trades)
                
                current_session_trades = [trade]
                current_date = trade_date
            else:
                current_session_trades.append(trade)
        
        # Handle last session
        if current_session_trades:
            self._create_trading_session(current_session_trades)
    
    def _create_trading_session(self, session_trades: List[Trade]):
        """Create a trading session from a list of trades"""
        if not session_trades:
            return
        
        start_time = min(t.timestamp for t in session_trades)
        end_time = max(t.timestamp for t in session_trades)
        trades_count = len(session_trades)
        total_volume = sum(t.value for t in session_trades)
        
        # Calculate session PnL
        session_pnl = sum(t.pnl for t in session_trades if t.pnl is not None)
        
        # Calculate session win rate
        trades_with_pnl = [t for t in session_trades if t.pnl is not None]
        if trades_with_pnl:
            winning_trades = [t for t in trades_with_pnl if t.pnl > 0]
            win_rate = len(winning_trades) / len(trades_with_pnl) * 100
        else:
            win_rate = 0
        
        # Determine emotional state (simplified heuristic)
        emotions = self._determine_session_emotions(session_trades)
        
        session = TradingSession(
            start_time=start_time,
            end_time=end_time,
            trades_count=trades_count,
            total_volume=total_volume,
            pnl=session_pnl,
            win_rate=win_rate,
            emotions=emotions
        )
        
        self.trading_sessions.append(session)
    
    def _determine_session_emotions(self, session_trades: List[Trade]) -> str:
        """Determine the emotional state of a trading session"""
        if not session_trades:
            return 'CALM'
        
        # High frequency = FOMO or REVENGE
        if len(session_trades) > 10:
            return 'FOMO'
        
        # Large position sizes = GREED or FEAR
        avg_value = np.mean([t.value for t in session_trades])
        if avg_value > np.mean([t.value for t in self.trades]) * 2:
            return 'GREED'
        
        # Many market orders = FEAR or FOMO
        market_orders = sum(1 for t in session_trades if t.trade_type == 'MARKET')
        if market_orders / len(session_trades) > 0.8:
            return 'FEAR'
        
        return 'CALM'
    
    def _calculate_discipline_score(self) -> float:
        """Calculate trading discipline score"""
        if not self.trades:
            return 0
        
        # Factors: limit orders vs market orders, position sizing consistency
        limit_orders = sum(1 for t in self.trades if t.trade_type == 'LIMIT')
        limit_order_ratio = limit_orders / len(self.trades)
        
        # Position sizing consistency
        trade_values = [t.value for t in self.trades]
        value_cv = np.std(trade_values) / np.mean(trade_values) if trade_values else 0
        consistency_score = max(0, 100 - (value_cv * 100))
        
        discipline_score = (limit_order_ratio * 50) + (consistency_score * 0.5)
        return min(100, max(0, discipline_score))
    
    def _calculate_risk_management_score(self) -> float:
        """Calculate risk management score"""
        if not self.performance_metrics:
            return 0
        
        # Factors: max drawdown, risk-reward ratio, position sizing
        drawdown_score = max(0, 100 - self.performance_metrics.max_drawdown * 2)
        rr_score = min(100, self.performance_metrics.risk_reward_ratio * 25)
        
        # Position sizing - check if any single trade is >10% of total volume
        total_volume = sum(t.value for t in self.trades)
        max_trade_value = max(t.value for t in self.trades)
        sizing_score = 100 if max_trade_value / total_volume < 0.1 else 50
        
        risk_score = (drawdown_score * 0.4) + (rr_score * 0.4) + (sizing_score * 0.2)
        return min(100, max(0, risk_score))
    
    def _calculate_emotional_control_score(self) -> float:
        """Calculate emotional control score"""
        if not self.trading_sessions:
            return 50
        
        # Factors: session emotions, trading frequency patterns
        calm_sessions = sum(1 for s in self.trading_sessions if s.emotions == 'CALM')
        emotional_sessions = len(self.trading_sessions) - calm_sessions
        
        emotional_control = (calm_sessions / len(self.trading_sessions)) * 100
        
        # Penalize revenge trading patterns
        revenge_sessions = sum(1 for s in self.trading_sessions if s.emotions in ['REVENGE', 'FOMO'])
        if revenge_sessions > 0:
            emotional_control -= (revenge_sessions / len(self.trading_sessions)) * 30
        
        return min(100, max(0, emotional_control))
    
    def _calculate_patience_score(self) -> float:
        """Calculate patience score based on holding periods"""
        if not self.trades:
            return 0
        
        # Analyze holding periods
        trades_with_duration = [t for t in self.trades if t.duration is not None]
        if not trades_with_duration:
            return 50  # Neutral score if no duration data
        
        # Average holding period
        avg_duration = np.mean([t.duration.total_seconds() for t in trades_with_duration])
        
        # Score based on average holding time
        # > 1 day = high patience, < 1 hour = low patience
        if avg_duration > 86400:  # > 1 day
            patience_score = 90
        elif avg_duration > 21600:  # > 6 hours
            patience_score = 75
        elif avg_duration > 3600:  # > 1 hour
            patience_score = 60
        elif avg_duration > 900:  # > 15 minutes
            patience_score = 40
        else:  # < 15 minutes
            patience_score = 20
        
        return patience_score
    
    def _calculate_consistency_score(self) -> float:
        """Calculate trading consistency score"""
        if not self.trading_sessions:
            return 0
        
        # Daily PnL consistency
        daily_pnls = [s.pnl for s in self.trading_sessions if s.pnl != 0]
        if not daily_pnls:
            return 50
        
        # Calculate coefficient of variation
        pnl_mean = np.mean(daily_pnls)
        pnl_std = np.std(daily_pnls)
        
        if pnl_mean == 0:
            return 50
        
        cv = abs(pnl_std / pnl_mean)
        consistency_score = max(0, 100 - (cv * 50))
        
        return min(100, consistency_score)
    
    def _calculate_overtrading_tendency(self) -> float:
        """Calculate overtrading tendency"""
        if not self.trading_sessions:
            return 0
        
        # Average trades per session
        avg_trades_per_session = np.mean([s.trades_count for s in self.trading_sessions])
        
        # High-frequency sessions
        high_freq_sessions = sum(1 for s in self.trading_sessions if s.trades_count > 20)
        high_freq_ratio = high_freq_sessions / len(self.trading_sessions)
        
        overtrading_score = (avg_trades_per_session * 5) + (high_freq_ratio * 50)
        return min(100, overtrading_score)
    
    def _calculate_fomo_susceptibility(self) -> float:
        """Calculate FOMO susceptibility"""
        if not self.trading_sessions:
            return 0
        
        # FOMO indicators: market orders during volatile periods, large position sizes
        fomo_sessions = sum(1 for s in self.trading_sessions if s.emotions == 'FOMO')
        market_order_ratio = sum(1 for t in self.trades if t.trade_type == 'MARKET') / len(self.trades)
        
        fomo_score = (fomo_sessions / len(self.trading_sessions)) * 70 + (market_order_ratio * 30)
        return min(100, fomo_score)
    
    def _calculate_revenge_trading_risk(self) -> float:
        """Calculate revenge trading risk"""
        if not self.trading_sessions:
            return 0
        
        # Look for sessions after losses with increased volume/frequency
        revenge_indicators = 0
        
        for i in range(1, len(self.trading_sessions)):
            prev_session = self.trading_sessions[i-1]
            current_session = self.trading_sessions[i]
            
            # If previous session had losses and current has high volume/frequency
            if (prev_session.pnl < 0 and 
                current_session.trades_count > prev_session.trades_count * 1.5):
                revenge_indicators += 1
        
        revenge_score = (revenge_indicators / max(1, len(self.trading_sessions) - 1)) * 100
        return min(100, revenge_score)
    
    def generate_ai_recommendations(self) -> List[StrategyRecommendation]:
        """Generate AI-powered personalized strategy recommendations"""
        if not self.performance_metrics or not self.trading_behavior:
            print("❌ Need performance metrics and behavior analysis first")
            return []
        
        try:
            # Prepare analysis data for AI
            analysis_data = {
                'performance': asdict(self.performance_metrics),
                'behavior': asdict(self.trading_behavior),
                'trading_patterns': self._extract_trading_patterns(),
                'session_analysis': self._analyze_session_patterns()
            }
            
            prompt = f"""
            As an expert trading coach, analyze this comprehensive trading history and provide personalized strategy recommendations:
            
            PERFORMANCE METRICS:
            - Win Rate: {self.performance_metrics.win_rate:.1f}%
            - Profit Factor: {self.performance_metrics.profit_factor:.2f}
            - Max Drawdown: {self.performance_metrics.max_drawdown:.1f}%
            - Risk-Reward Ratio: {self.performance_metrics.risk_reward_ratio:.2f}
            - Total Trades: {self.performance_metrics.total_trades}
            - Sharpe Ratio: {self.performance_metrics.sharpe_ratio:.2f}
            
            BEHAVIORAL ANALYSIS:
            - Discipline Score: {self.trading_behavior.discipline_score:.1f}/100
            - Risk Management: {self.trading_behavior.risk_management_score:.1f}/100
            - Emotional Control: {self.trading_behavior.emotional_control_score:.1f}/100
            - Patience Score: {self.trading_behavior.patience_score:.1f}/100
            - Consistency: {self.trading_behavior.consistency_score:.1f}/100
            - Overtrading Tendency: {self.trading_behavior.overtrading_tendency:.1f}/100
            - FOMO Susceptibility: {self.trading_behavior.fomo_susceptibility:.1f}/100
            - Revenge Trading Risk: {self.trading_behavior.revenge_trading_risk:.1f}/100
            
            STRENGTHS: {', '.join(self.trading_behavior.primary_strengths)}
            WEAKNESSES: {', '.join(self.trading_behavior.primary_weaknesses)}
            
            DETAILED ANALYSIS:
            {json.dumps(analysis_data, indent=2, default=str)}
            
            Provide 3-5 specific, actionable strategy recommendations that address:
            1. Trading style that matches their strengths
            2. Specific improvements for weaknesses
            3. Risk management adjustments
            4. Behavioral modifications
            5. Expected performance improvements
            
            Format each recommendation as:
            Strategy Type: [Scalping/Swing/Position/etc.]
            Confidence: [0-1]
            Reasoning: [Why this strategy fits]
            Improvements: [Specific actionable items]
            Risk Adjustments: [Risk management changes]
            Behavioral Modifications: [Psychology/habit changes]
            Expected Improvement: [Estimated % improvement]
            """
            
            completion = self.grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            ai_response = completion.choices[0].message.content
            
            # Parse AI response into structured recommendations
            recommendations = self._parse_strategy_recommendations(ai_response)
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating AI recommendations: {e}")
            return []
    
    def _extract_trading_patterns(self) -> Dict[str, Any]:
        """Extract key trading patterns"""
        if not self.trades:
            return {}
        
        # Time-based patterns
        trade_hours = [t.timestamp.hour for t in self.trades]
        most_active_hour = max(set(trade_hours), key=trade_hours.count)
        
        # Symbol preferences
        symbols = [t.symbol for t in self.trades]
        symbol_counts = {symbol: symbols.count(symbol) for symbol in set(symbols)}
        favorite_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Trade size patterns
        trade_values = [t.value for t in self.trades]
        avg_trade_size = np.mean(trade_values)
        
        return {
            'most_active_hour': most_active_hour,
            'favorite_symbols': [s[0] for s in favorite_symbols],
            'avg_trade_size': avg_trade_size,
            'trading_frequency': len(self.trades) / 180,  # trades per day over 6 months
        }
    
    def _analyze_session_patterns(self) -> Dict[str, Any]:
        """Analyze trading session patterns"""
        if not self.trading_sessions:
            return {}
        
        # Session performance
        profitable_sessions = sum(1 for s in self.trading_sessions if s.pnl > 0)
        session_win_rate = profitable_sessions / len(self.trading_sessions) * 100
        
        # Emotional patterns
        emotion_counts = {}
        for session in self.trading_sessions:
            emotion_counts[session.emotions] = emotion_counts.get(session.emotions, 0) + 1
        
        return {
            'session_win_rate': session_win_rate,
            'total_sessions': len(self.trading_sessions),
            'emotion_distribution': emotion_counts,
            'avg_trades_per_session': np.mean([s.trades_count for s in self.trading_sessions])
        }
    
    def _parse_strategy_recommendations(self, ai_response: str) -> List[StrategyRecommendation]:
        """Parse AI response into structured strategy recommendations"""
        recommendations = []
        
        # Split response into recommendation blocks
        sections = ai_response.split('\n\n')
        current_rec = {}
        
        for section in sections:
            lines = section.split('\n')
            
            for line in lines:
                line = line.strip()
                
                if 'Strategy Type:' in line:
                    # Save previous recommendation if complete
                    if current_rec.get('strategy_type'):
                        recommendations.append(self._create_recommendation(current_rec))
                    current_rec = {'strategy_type': line.split(':', 1)[1].strip()}
                
                elif 'Confidence:' in line:
                    try:
                        current_rec['confidence'] = float(line.split(':', 1)[1].strip())
                    except:
                        current_rec['confidence'] = 0.7
                
                elif 'Reasoning:' in line:
                    current_rec['reasoning'] = line.split(':', 1)[1].strip()
                
                elif 'Improvements:' in line:
                    improvements_text = line.split(':', 1)[1].strip()
                    current_rec['improvements'] = [imp.strip() for imp in improvements_text.split(',')]
                
                elif 'Risk Adjustments:' in line:
                    risk_text = line.split(':', 1)[1].strip()
                    current_rec['risk_adjustments'] = [adj.strip() for adj in risk_text.split(',')]
                
                elif 'Behavioral Modifications:' in line:
                    behavior_text = line.split(':', 1)[1].strip()
                    current_rec['behavioral_modifications'] = [mod.strip() for mod in behavior_text.split(',')]
                
                elif 'Expected Improvement:' in line:
                    try:
                        improvement_text = line.split(':', 1)[1].strip().replace('%', '')
                        current_rec['expected_improvement'] = float(improvement_text)
                    except:
                        current_rec['expected_improvement'] = 10.0
        
        # Add final recommendation
        if current_rec.get('strategy_type'):
            recommendations.append(self._create_recommendation(current_rec))
        
        return recommendations
    
    def _create_recommendation(self, rec_data: Dict) -> StrategyRecommendation:
        """Create a StrategyRecommendation from parsed data"""
        return StrategyRecommendation(
            strategy_type=rec_data.get('strategy_type', 'General'),
            confidence=rec_data.get('confidence', 0.7),
            reasoning=rec_data.get('reasoning', 'AI analysis based on trading patterns'),
            improvements=rec_data.get('improvements', ['Focus on risk management']),
            risk_adjustments=rec_data.get('risk_adjustments', ['Reduce position sizes']),
            behavioral_modifications=rec_data.get('behavioral_modifications', ['Improve patience']),
            expected_improvement=rec_data.get('expected_improvement', 10.0)
        )
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive trading analysis report"""
        if not self.performance_metrics or not self.trading_behavior:
            return "❌ Complete analysis required before generating report"
        
        report = f"""
# COMPREHENSIVE TRADING ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY
- Total Trades Analyzed: {self.performance_metrics.total_trades}
- Overall Win Rate: {self.performance_metrics.win_rate:.1f}%
- Total PnL: ${self.performance_metrics.total_pnl:,.2f}
- Max Drawdown: {self.performance_metrics.max_drawdown:.1f}%
- Risk-Reward Ratio: {self.performance_metrics.risk_reward_ratio:.2f}

## PERFORMANCE METRICS
- Profit Factor: {self.performance_metrics.profit_factor:.2f}
- Sharpe Ratio: {self.performance_metrics.sharpe_ratio:.2f}
- Average Win: ${self.performance_metrics.avg_win:.2f}
- Average Loss: ${self.performance_metrics.avg_loss:.2f}
- Largest Win: ${self.performance_metrics.largest_win:.2f}
- Largest Loss: ${self.performance_metrics.largest_loss:.2f}

## BEHAVIORAL ANALYSIS
- Discipline Score: {self.trading_behavior.discipline_score:.1f}/100
- Risk Management Score: {self.trading_behavior.risk_management_score:.1f}/100
- Emotional Control Score: {self.trading_behavior.emotional_control_score:.1f}/100
- Patience Score: {self.trading_behavior.patience_score:.1f}/100
- Consistency Score: {self.trading_behavior.consistency_score:.1f}/100

## RISK FACTORS
- Overtrading Tendency: {self.trading_behavior.overtrading_tendency:.1f}/100
- FOMO Susceptibility: {self.trading_behavior.fomo_susceptibility:.1f}/100
- Revenge Trading Risk: {self.trading_behavior.revenge_trading_risk:.1f}/100

## STRENGTHS
{chr(10).join(f"✅ {strength}" for strength in self.trading_behavior.primary_strengths)}

## WEAKNESSES
{chr(10).join(f"❌ {weakness}" for weakness in self.trading_behavior.primary_weaknesses)}

## TRADING SESSIONS ANALYSIS
- Total Sessions: {len(self.trading_sessions)}
- Average Trades per Session: {np.mean([s.trades_count for s in self.trading_sessions]):.1f}
- Session Win Rate: {sum(1 for s in self.trading_sessions if s.pnl > 0) / len(self.trading_sessions) * 100:.1f}%

## RECOMMENDATIONS
{chr(10).join(f"💡 {rec.strategy_type}: {rec.reasoning}" for rec in self.generate_ai_recommendations()[:3])}
"""
        
        return report


def main():
    """Demo function for testing the trading history analyzer"""
    print("📊 TRADING HISTORY ANALYZER DEMO")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TradingHistoryAnalyzer()
    
    # Import demo data
    print("📥 Importing demo trading data...")
    success = analyzer.import_trading_data("demo")
    
    if not success:
        print("❌ Failed to import trading data")
        return
    
    # Analyze performance
    print("\n📈 Analyzing performance metrics...")
    performance = analyzer.analyze_performance_metrics()
    
    if performance:
        print(f"✅ Performance analysis complete")
        print(f"   Win Rate: {performance.win_rate:.1f}%")
        print(f"   Profit Factor: {performance.profit_factor:.2f}")
        print(f"   Max Drawdown: {performance.max_drawdown:.1f}%")
    
    # Analyze behavior
    print("\n🧠 Analyzing trading behavior...")
    behavior = analyzer.analyze_trading_behavior()
    
    if behavior:
        print(f"✅ Behavioral analysis complete")
        print(f"   Discipline: {behavior.discipline_score:.1f}/100")
        print(f"   Risk Management: {behavior.risk_management_score:.1f}/100")
        print(f"   Emotional Control: {behavior.emotional_control_score:.1f}/100")
    
    # Generate recommendations
    print("\n🎯 Generating AI recommendations...")
    recommendations = analyzer.generate_ai_recommendations()
    
    print(f"✅ Generated {len(recommendations)} recommendations")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"   {i}. {rec.strategy_type} (Confidence: {rec.confidence:.1%})")
    
    # Generate report
    print("\n📋 Generating comprehensive report...")
    report = analyzer.generate_comprehensive_report()
    
    print("✅ Analysis complete!")
    print("\n" + "="*60)
    print(report)


if __name__ == "__main__":
    main() 