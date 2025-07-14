"""
Comprehensive Backtesting Framework for Trading Systems
Provides historical simulation, performance metrics, and robust testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@dataclass
class BacktestTrade:
    """Individual trade in backtest"""
    entry_time: datetime
    exit_time: datetime
    asset: str
    side: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    duration: timedelta
    reason: str  # 'SIGNAL', 'STOP_LOSS', 'TAKE_PROFIT', 'TIMEOUT'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Returns
    total_return: float
    annualized_return: float
    cagr: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: timedelta
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade metrics
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Advanced metrics
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    information_ratio: float
    
    # Timing metrics
    avg_trade_duration: timedelta
    hit_rate_by_duration: Dict[str, float]
    
    # Additional metrics
    recovery_factor: float
    ulcer_index: float
    sterling_ratio: float
    burke_ratio: float

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: str
    end_date: str
    initial_capital: float = 100000
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%
    max_position_size: float = 0.1  # 10% of capital
    risk_per_trade: float = 0.02  # 2% risk per trade
    compounding: bool = True
    benchmark: str = 'SPY'
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    
    # Stop loss and take profit
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    
    # Position sizing
    position_sizing_method: str = 'fixed_fractional'  # 'fixed', 'fixed_fractional', 'kelly', 'volatility'
    
    # Advanced settings
    allow_short_selling: bool = True
    margin_requirement: float = 0.5
    interest_rate: float = 0.02  # 2% annual
    
    # Data settings
    data_frequency: str = '1d'  # '1m', '5m', '1h', '1d'
    warmup_period: int = 252  # Days for indicator warmup

class BacktestEngine:
    """Advanced backtesting engine for trading strategies"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_cache = {}
        self.benchmark_data = None
        self.results = {}
        self.trade_history = []
        self.portfolio_history = []
        self.current_positions = {}
        self.capital_history = []
        
        # Performance tracking
        self.daily_returns = []
        self.drawdown_history = []
        self.equity_curve = []
        
        print("🧪 Backtesting Engine initialized")
        print(f"   Period: {config.start_date} to {config.end_date}")
        print(f"   Initial Capital: ${config.initial_capital:,.2f}")
        print(f"   Commission: {config.commission_rate:.3%}")
        print(f"   Slippage: {config.slippage_rate:.3%}")
    
    def fetch_data(self, symbol: str, period: str = None) -> pd.DataFrame:
        """Fetch historical data for backtesting"""
        if symbol in self.data_cache:
            return self.data_cache[symbol]
        
        try:
            # Use yfinance for data fetching
            ticker = yf.Ticker(symbol)
            
            # Determine period
            if period:
                data = ticker.history(period=period, interval=self.config.data_frequency)
            else:
                data = ticker.history(
                    start=self.config.start_date,
                    end=self.config.end_date,
                    interval=self.config.data_frequency
                )
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Add basic technical indicators
            data = self._add_technical_indicators(data)
            
            # Cache data
            self.data_cache[symbol] = data
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to data"""
        # Simple moving averages
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Exponential moving averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Volatility
        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std()
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Returns
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        return data
    
    def backtest_strategy(self, strategy_func: Callable, assets: List[str], 
                         strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run comprehensive backtest for a trading strategy"""
        print(f"🚀 Starting backtest for {len(assets)} assets...")
        
        # Initialize
        self.trade_history = []
        self.portfolio_history = []
        self.current_positions = {}
        self.capital_history = []
        
        # Fetch data for all assets
        print("📊 Fetching historical data...")
        asset_data = {}
        for asset in assets:
            data = self.fetch_data(asset)
            if not data.empty:
                asset_data[asset] = data
        
        if not asset_data:
            print("❌ No data available for backtesting")
            return {}
        
        # Get benchmark data
        print("📈 Fetching benchmark data...")
        self.benchmark_data = self.fetch_data(self.config.benchmark)
        
        # Align dates
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        # Initialize portfolio
        current_capital = self.config.initial_capital
        portfolio_value = current_capital
        
        # Create combined date index
        all_dates = set()
        for data in asset_data.values():
            all_dates.update(data.index)
        
        date_range = sorted([d for d in all_dates if start_date <= d <= end_date])
        
        print(f"🔄 Processing {len(date_range)} trading days...")
        
        # Main backtesting loop
        for i, current_date in enumerate(date_range):
            try:
                # Update portfolio value
                portfolio_value = self._calculate_portfolio_value(current_date, asset_data)
                
                # Check for stop losses and take profits
                self._check_exit_conditions(current_date, asset_data)
                
                # Generate trading signals
                signals = strategy_func(
                    current_date=current_date,
                    asset_data=asset_data,
                    portfolio_value=portfolio_value,
                    current_positions=self.current_positions,
                    params=strategy_params or {}
                )
                
                # Process signals
                for signal in signals:
                    self._process_signal(signal, current_date, asset_data, portfolio_value)
                
                # Record portfolio state
                self._record_portfolio_state(current_date, portfolio_value)
                
                # Progress update
                if i % 100 == 0:
                    progress = (i / len(date_range)) * 100
                    print(f"   Progress: {progress:.1f}% - Portfolio: ${portfolio_value:,.2f}")
                
            except Exception as e:
                print(f"❌ Error processing {current_date}: {e}")
                continue
        
        # Calculate final metrics
        print("📊 Calculating performance metrics...")
        metrics = self._calculate_metrics()
        
        # Generate results
        results = {
            'metrics': metrics,
            'trades': self.trade_history,
            'portfolio_history': self.portfolio_history,
            'capital_history': self.capital_history,
            'config': self.config,
            'assets': assets,
            'strategy_params': strategy_params
        }
        
        self.results = results
        
        print(f"✅ Backtest completed!")
        print(f"   Total Trades: {metrics.total_trades}")
        print(f"   Win Rate: {metrics.win_rate:.2%}")
        print(f"   Total Return: {metrics.total_return:.2%}")
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
        
        return results
    
    def _calculate_portfolio_value(self, current_date: datetime, asset_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate current portfolio value"""
        total_value = 0
        
        # Cash
        total_value += sum(pos.get('cash', 0) for pos in self.current_positions.values())
        
        # Positions
        for asset, position in self.current_positions.items():
            if asset in asset_data and current_date in asset_data[asset].index:
                current_price = asset_data[asset].loc[current_date, 'Close']
                position_value = position.get('quantity', 0) * current_price
                total_value += position_value
        
        # If no positions, use initial capital
        if total_value == 0:
            total_value = self.config.initial_capital
        
        return total_value
    
    def _check_exit_conditions(self, current_date: datetime, asset_data: Dict[str, pd.DataFrame]):
        """Check for stop loss and take profit conditions"""
        positions_to_close = []
        
        for asset, position in self.current_positions.items():
            if asset not in asset_data or current_date not in asset_data[asset].index:
                continue
            
            if position.get('quantity', 0) == 0:
                continue
            
            current_price = asset_data[asset].loc[current_date, 'Close']
            entry_price = position.get('entry_price', current_price)
            side = position.get('side', 'LONG')
            
            # Calculate P&L
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check stop loss
            if self.config.stop_loss_pct and pnl_pct <= -self.config.stop_loss_pct:
                positions_to_close.append((asset, current_price, 'STOP_LOSS'))
            
            # Check take profit
            elif self.config.take_profit_pct and pnl_pct >= self.config.take_profit_pct:
                positions_to_close.append((asset, current_price, 'TAKE_PROFIT'))
        
        # Close positions
        for asset, exit_price, reason in positions_to_close:
            self._close_position(asset, exit_price, current_date, reason)
    
    def _process_signal(self, signal: Dict[str, Any], current_date: datetime, 
                       asset_data: Dict[str, pd.DataFrame], portfolio_value: float):
        """Process a trading signal"""
        asset = signal.get('asset')
        action = signal.get('action')  # 'BUY', 'SELL', 'CLOSE'
        confidence = signal.get('confidence', 0.5)
        
        if not asset or not action:
            return
        
        if asset not in asset_data or current_date not in asset_data[asset].index:
            return
        
        current_price = asset_data[asset].loc[current_date, 'Close']
        
        # Calculate position size
        position_size = self._calculate_position_size(
            asset, current_price, portfolio_value, confidence
        )
        
        if position_size == 0:
            return
        
        # Apply slippage
        if action == 'BUY':
            execution_price = current_price * (1 + self.config.slippage_rate)
        else:
            execution_price = current_price * (1 - self.config.slippage_rate)
        
        # Calculate commission
        commission = position_size * execution_price * self.config.commission_rate
        
        # Execute trade
        if action == 'BUY':
            self._open_position(asset, 'LONG', position_size, execution_price, 
                              current_date, commission, signal)
        elif action == 'SELL':
            if self.config.allow_short_selling:
                self._open_position(asset, 'SHORT', position_size, execution_price, 
                                  current_date, commission, signal)
            else:
                # Close long position if exists
                if asset in self.current_positions:
                    self._close_position(asset, execution_price, current_date, 'SELL_SIGNAL')
        elif action == 'CLOSE':
            if asset in self.current_positions:
                self._close_position(asset, execution_price, current_date, 'CLOSE_SIGNAL')
    
    def _calculate_position_size(self, asset: str, price: float, portfolio_value: float, 
                               confidence: float) -> float:
        """Calculate position size based on configuration"""
        if self.config.position_sizing_method == 'fixed':
            return 1.0  # Fixed quantity
        
        elif self.config.position_sizing_method == 'fixed_fractional':
            max_position_value = portfolio_value * self.config.max_position_size
            return max_position_value / price
        
        elif self.config.position_sizing_method == 'volatility':
            # Volatility-based sizing
            if asset in self.data_cache:
                volatility = self.data_cache[asset]['Volatility'].iloc[-1]
                if pd.isna(volatility) or volatility == 0:
                    volatility = 0.02  # Default 2% volatility
                
                # Inverse volatility scaling
                vol_adjustment = min(0.02 / volatility, 2.0)  # Max 2x, min 0.5x
                base_size = portfolio_value * self.config.max_position_size
                return (base_size * vol_adjustment) / price
            else:
                return (portfolio_value * self.config.max_position_size) / price
        
        elif self.config.position_sizing_method == 'kelly':
            # Kelly criterion (simplified)
            win_rate = 0.55  # Assume 55% win rate
            avg_win = 0.05   # Assume 5% average win
            avg_loss = 0.03  # Assume 3% average loss
            
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, self.config.max_position_size))
            
            return (portfolio_value * kelly_fraction) / price
        
        else:
            return (portfolio_value * self.config.max_position_size) / price
    
    def _open_position(self, asset: str, side: str, quantity: float, price: float,
                      date: datetime, commission: float, signal: Dict[str, Any]):
        """Open a new position"""
        # Close existing position if different side
        if asset in self.current_positions:
            current_side = self.current_positions[asset].get('side')
            if current_side != side:
                self._close_position(asset, price, date, 'FLIP_POSITION')
        
        # Create new position
        self.current_positions[asset] = {
            'side': side,
            'quantity': quantity,
            'entry_price': price,
            'entry_date': date,
            'commission': commission,
            'signal': signal
        }
    
    def _close_position(self, asset: str, price: float, date: datetime, reason: str):
        """Close an existing position"""
        if asset not in self.current_positions:
            return
        
        position = self.current_positions[asset]
        
        # Calculate P&L
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']
        entry_commission = position['commission']
        
        # Exit commission
        exit_commission = quantity * price * self.config.commission_rate
        
        # Calculate P&L
        if side == 'LONG':
            pnl = (price - entry_price) * quantity - entry_commission - exit_commission
        else:  # SHORT
            pnl = (entry_price - price) * quantity - entry_commission - exit_commission
        
        pnl_pct = pnl / (entry_price * quantity)
        
        # Create trade record
        trade = BacktestTrade(
            entry_time=position['entry_date'],
            exit_time=date,
            asset=asset,
            side=side,
            entry_price=entry_price,
            exit_price=price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=entry_commission + exit_commission,
            slippage=0,  # Already applied
            duration=date - position['entry_date'],
            reason=reason,
            metadata=position.get('signal', {})
        )
        
        self.trade_history.append(trade)
        
        # Remove position
        del self.current_positions[asset]
    
    def _record_portfolio_state(self, date: datetime, portfolio_value: float):
        """Record portfolio state for analysis"""
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'positions': dict(self.current_positions),
            'cash': portfolio_value - sum(
                pos.get('quantity', 0) * pos.get('entry_price', 0)
                for pos in self.current_positions.values()
            )
        })
        
        # Calculate daily return
        if len(self.capital_history) > 0:
            prev_value = self.capital_history[-1]['value']
            daily_return = (portfolio_value - prev_value) / prev_value
        else:
            daily_return = 0
        
        self.capital_history.append({
            'date': date,
            'value': portfolio_value,
            'daily_return': daily_return
        })
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trade_history or not self.capital_history:
            return BacktestMetrics(
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
                total_return=0, annualized_return=0, cagr=0,
                max_drawdown=0, max_drawdown_duration=timedelta(0),
                volatility=0, sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                avg_win=0, avg_loss=0, profit_factor=0, expectancy=0,
                var_95=0, cvar_95=0, beta=0, alpha=0, information_ratio=0,
                avg_trade_duration=timedelta(0), hit_rate_by_duration={},
                recovery_factor=0, ulcer_index=0, sterling_ratio=0, burke_ratio=0
            )
        
        # Basic trade metrics
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.pnl > 0)
        losing_trades = sum(1 for t in self.trade_history if t.pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Returns
        initial_capital = self.config.initial_capital
        final_value = self.capital_history[-1]['value']
        total_return = (final_value - initial_capital) / initial_capital
        
        # Time-based metrics
        days = len(self.capital_history)
        years = days / 252  # Trading days per year
        
        if years > 0:
            cagr = (final_value / initial_capital) ** (1 / years) - 1
            annualized_return = cagr
        else:
            cagr = 0
            annualized_return = 0
        
        # Volatility and risk metrics
        daily_returns = [entry['daily_return'] for entry in self.capital_history[1:]]
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        
        # Sharpe ratio
        risk_free_rate = 0.02  # 2% annual
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = [r for r in daily_returns if r < 0]
        downside_volatility = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0
        sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown analysis
        equity_curve = [entry['value'] for entry in self.capital_history]
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Max drawdown duration
        drawdown_duration = timedelta(0)
        current_dd_start = None
        max_dd_duration = timedelta(0)
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and current_dd_start is None:
                current_dd_start = i
            elif dd >= 0 and current_dd_start is not None:
                duration = timedelta(days=i - current_dd_start)
                max_dd_duration = max(max_dd_duration, duration)
                current_dd_start = None
        
        # Calmar ratio
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown < 0 else 0
        
        # Trade-specific metrics
        winning_pnls = [t.pnl for t in self.trade_history if t.pnl > 0]
        losing_pnls = [t.pnl for t in self.trade_history if t.pnl < 0]
        
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0
        
        profit_factor = abs(sum(winning_pnls) / sum(losing_pnls)) if losing_pnls else float('inf')
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Value at Risk (VaR)
        var_95 = np.percentile(daily_returns, 5) if daily_returns else 0
        cvar_95 = np.mean([r for r in daily_returns if r <= var_95]) if daily_returns else 0
        
        # Beta and Alpha (vs benchmark)
        beta = 0
        alpha = 0
        information_ratio = 0
        
        if self.benchmark_data is not None and len(self.benchmark_data) > 0:
            # Calculate benchmark returns
            benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
            
            # Align returns
            common_dates = set(self.benchmark_data.index) & set(
                pd.to_datetime([entry['date'] for entry in self.capital_history])
            )
            
            if len(common_dates) > 10:
                strategy_returns = []
                bench_returns = []
                
                for entry in self.capital_history:
                    if entry['date'] in common_dates:
                        strategy_returns.append(entry['daily_return'])
                        if entry['date'] in benchmark_returns.index:
                            bench_returns.append(benchmark_returns.loc[entry['date']])
                
                if len(strategy_returns) == len(bench_returns) and len(strategy_returns) > 10:
                    # Calculate beta
                    covariance = np.cov(strategy_returns, bench_returns)[0, 1]
                    bench_variance = np.var(bench_returns)
                    beta = covariance / bench_variance if bench_variance > 0 else 0
                    
                    # Calculate alpha
                    bench_return = np.mean(bench_returns) * 252
                    alpha = annualized_return - (risk_free_rate + beta * (bench_return - risk_free_rate))
                    
                    # Information ratio
                    tracking_error = np.std(np.array(strategy_returns) - np.array(bench_returns)) * np.sqrt(252)
                    information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        
        # Trade duration analysis
        trade_durations = [t.duration for t in self.trade_history]
        avg_trade_duration = np.mean(trade_durations) if trade_durations else timedelta(0)
        
        # Hit rate by duration
        duration_buckets = {
            'short': [t for t in self.trade_history if t.duration <= timedelta(days=1)],
            'medium': [t for t in self.trade_history if timedelta(days=1) < t.duration <= timedelta(days=7)],
            'long': [t for t in self.trade_history if t.duration > timedelta(days=7)]
        }
        
        hit_rate_by_duration = {}
        for bucket_name, trades in duration_buckets.items():
            if trades:
                wins = sum(1 for t in trades if t.pnl > 0)
                hit_rate_by_duration[bucket_name] = wins / len(trades)
            else:
                hit_rate_by_duration[bucket_name] = 0
        
        # Advanced metrics
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown < 0 else 0
        
        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown ** 2)) if len(drawdown) > 0 else 0
        
        # Sterling Ratio
        sterling_ratio = abs(annualized_return / ulcer_index) if ulcer_index > 0 else 0
        
        # Burke Ratio
        burke_ratio = abs(annualized_return / np.sqrt(np.mean(drawdown ** 2))) if len(drawdown) > 0 else 0
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_return,
            annualized_return=annualized_return,
            cagr=cagr,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio,
            avg_trade_duration=avg_trade_duration,
            hit_rate_by_duration=hit_rate_by_duration,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio
        )
    
    def parameter_optimization(self, strategy_func: Callable, assets: List[str], 
                             param_ranges: Dict[str, List[Any]], 
                             optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Optimize strategy parameters using grid search"""
        print(f"🔍 Starting parameter optimization...")
        print(f"   Metric: {optimization_metric}")
        print(f"   Parameter ranges: {param_ranges}")
        
        # Generate parameter combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"   Total combinations: {len(param_combinations)}")
        
        # Run backtests for each combination
        results = []
        best_result = None
        best_metric = float('-inf')
        
        for i, combination in enumerate(param_combinations):
            try:
                # Create parameter dict
                params = dict(zip(param_names, combination))
                
                # Run backtest
                result = self.backtest_strategy(strategy_func, assets, params)
                
                if result and 'metrics' in result:
                    metric_value = getattr(result['metrics'], optimization_metric, 0)
                    
                    results.append({
                        'params': params,
                        'metric_value': metric_value,
                        'metrics': result['metrics']
                    })
                    
                    # Check if this is the best result
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_result = result
                        best_result['params'] = params
                    
                    print(f"   {i+1}/{len(param_combinations)}: {params} -> {metric_value:.4f}")
                
            except Exception as e:
                print(f"   Error with params {combination}: {e}")
                continue
        
        # Sort results by metric
        results.sort(key=lambda x: x['metric_value'], reverse=True)
        
        optimization_results = {
            'best_params': best_result['params'] if best_result else {},
            'best_metric': best_metric,
            'best_result': best_result,
            'all_results': results[:10],  # Top 10 results
            'optimization_metric': optimization_metric
        }
        
        print(f"✅ Optimization complete!")
        print(f"   Best {optimization_metric}: {best_metric:.4f}")
        print(f"   Best params: {best_result['params'] if best_result else 'None'}")
        
        return optimization_results
    
    def monte_carlo_simulation(self, strategy_func: Callable, assets: List[str], 
                             strategy_params: Dict[str, Any], num_simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for strategy robustness testing"""
        print(f"🎲 Starting Monte Carlo simulation ({num_simulations} runs)...")
        
        simulation_results = []
        
        for i in range(num_simulations):
            try:
                # Add random noise to parameters
                noisy_params = {}
                for key, value in strategy_params.items():
                    if isinstance(value, (int, float)):
                        noise = np.random.normal(0, abs(value) * 0.1)  # 10% noise
                        noisy_params[key] = value + noise
                    else:
                        noisy_params[key] = value
                
                # Run backtest with noisy parameters
                result = self.backtest_strategy(strategy_func, assets, noisy_params)
                
                if result and 'metrics' in result:
                    metrics = result['metrics']
                    simulation_results.append({
                        'run': i + 1,
                        'params': noisy_params,
                        'total_return': metrics.total_return,
                        'sharpe_ratio': metrics.sharpe_ratio,
                        'max_drawdown': metrics.max_drawdown,
                        'win_rate': metrics.win_rate,
                        'profit_factor': metrics.profit_factor
                    })
                
                if (i + 1) % 100 == 0:
                    print(f"   Completed {i + 1}/{num_simulations} simulations...")
                
            except Exception as e:
                print(f"   Error in simulation {i + 1}: {e}")
                continue
        
        if not simulation_results:
            print("❌ No successful simulations")
            return {}
        
        # Calculate statistics
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        statistics = {}
        
        for metric in metrics:
            values = [r[metric] for r in simulation_results]
            statistics[metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'percentile_5': np.percentile(values, 5),
                'percentile_25': np.percentile(values, 25),
                'percentile_75': np.percentile(values, 75),
                'percentile_95': np.percentile(values, 95)
            }
        
        monte_carlo_results = {
            'num_simulations': len(simulation_results),
            'statistics': statistics,
            'all_results': simulation_results,
            'robustness_score': self._calculate_robustness_score(statistics)
        }
        
        print(f"✅ Monte Carlo simulation complete!")
        print(f"   Successful runs: {len(simulation_results)}")
        print(f"   Average return: {statistics['total_return']['mean']:.2%}")
        print(f"   Average Sharpe: {statistics['sharpe_ratio']['mean']:.2f}")
        print(f"   Robustness score: {monte_carlo_results['robustness_score']:.2f}")
        
        return monte_carlo_results
    
    def _calculate_robustness_score(self, statistics: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall robustness score"""
        # Factors: consistency, positive returns, low drawdown
        
        # Return consistency (lower std/mean ratio is better)
        return_consistency = 1 - abs(statistics['total_return']['std'] / statistics['total_return']['mean']) if statistics['total_return']['mean'] != 0 else 0
        return_consistency = max(0, min(1, return_consistency))
        
        # Positive returns frequency
        positive_returns = sum(1 for r in statistics['total_return'] if r > 0) / len(statistics['total_return']) if statistics['total_return'] else 0
        
        # Drawdown consistency
        drawdown_consistency = 1 - (statistics['max_drawdown']['std'] / abs(statistics['max_drawdown']['mean'])) if statistics['max_drawdown']['mean'] != 0 else 0
        drawdown_consistency = max(0, min(1, drawdown_consistency))
        
        # Combined score
        robustness_score = (return_consistency * 0.4) + (positive_returns * 0.3) + (drawdown_consistency * 0.3)
        
        return robustness_score
    
    def generate_report(self, save_to_file: bool = True) -> str:
        """Generate comprehensive backtest report"""
        if not self.results:
            return "No backtest results available"
        
        metrics = self.results['metrics']
        
        report = f"""
🧪 COMPREHENSIVE BACKTEST REPORT
=================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 STRATEGY CONFIGURATION
Period: {self.config.start_date} to {self.config.end_date}
Assets: {', '.join(self.results['assets'])}
Initial Capital: ${self.config.initial_capital:,.2f}
Commission: {self.config.commission_rate:.3%}
Slippage: {self.config.slippage_rate:.3%}

📈 PERFORMANCE SUMMARY
Total Return: {metrics.total_return:.2%}
Annualized Return: {metrics.annualized_return:.2%}
CAGR: {metrics.cagr:.2%}
Volatility: {metrics.volatility:.2%}
Sharpe Ratio: {metrics.sharpe_ratio:.2f}
Sortino Ratio: {metrics.sortino_ratio:.2f}
Calmar Ratio: {metrics.calmar_ratio:.2f}

📉 RISK METRICS
Max Drawdown: {metrics.max_drawdown:.2%}
Max DD Duration: {metrics.max_drawdown_duration}
VaR (95%): {metrics.var_95:.2%}
CVaR (95%): {metrics.cvar_95:.2%}
Ulcer Index: {metrics.ulcer_index:.4f}

📊 TRADING STATISTICS
Total Trades: {metrics.total_trades}
Winning Trades: {metrics.winning_trades}
Losing Trades: {metrics.losing_trades}
Win Rate: {metrics.win_rate:.2%}
Average Win: {metrics.avg_win:.2f}
Average Loss: {metrics.avg_loss:.2f}
Profit Factor: {metrics.profit_factor:.2f}
Expectancy: {metrics.expectancy:.2f}

⏱️ TIMING ANALYSIS
Average Trade Duration: {metrics.avg_trade_duration}
Short-term Hit Rate: {metrics.hit_rate_by_duration.get('short', 0):.2%}
Medium-term Hit Rate: {metrics.hit_rate_by_duration.get('medium', 0):.2%}
Long-term Hit Rate: {metrics.hit_rate_by_duration.get('long', 0):.2%}

🔄 BENCHMARK COMPARISON
Beta: {metrics.beta:.2f}
Alpha: {metrics.alpha:.2%}
Information Ratio: {metrics.information_ratio:.2f}

🏆 ADVANCED METRICS
Recovery Factor: {metrics.recovery_factor:.2f}
Sterling Ratio: {metrics.sterling_ratio:.2f}
Burke Ratio: {metrics.burke_ratio:.2f}

💡 INTERPRETATION
"""
        
        # Performance interpretation
        if metrics.sharpe_ratio > 2:
            report += "🟢 EXCELLENT: Outstanding risk-adjusted returns\n"
        elif metrics.sharpe_ratio > 1:
            report += "🟡 GOOD: Solid risk-adjusted performance\n"
        elif metrics.sharpe_ratio > 0:
            report += "🟠 FAIR: Positive but modest risk-adjusted returns\n"
        else:
            report += "🔴 POOR: Negative risk-adjusted returns\n"
        
        # Drawdown assessment
        if abs(metrics.max_drawdown) < 0.1:
            report += "🟢 LOW RISK: Excellent drawdown control\n"
        elif abs(metrics.max_drawdown) < 0.2:
            report += "🟡 MODERATE RISK: Acceptable drawdown levels\n"
        else:
            report += "🔴 HIGH RISK: Significant drawdown exposure\n"
        
        # Win rate assessment
        if metrics.win_rate > 0.6:
            report += "🟢 HIGH ACCURACY: Excellent win rate\n"
        elif metrics.win_rate > 0.5:
            report += "🟡 MODERATE ACCURACY: Good win rate\n"
        else:
            report += "🟠 LOW ACCURACY: Consider improving entry signals\n"
        
        report += f"""
⚠️ RISK CONSIDERATIONS
• Past performance doesn't guarantee future results
• Strategy may be overfit to historical data
• Consider transaction costs and market impact
• Monitor for regime changes and model decay
• Use appropriate position sizing and risk management

🔄 RECOMMENDATIONS
"""
        
        if metrics.sharpe_ratio > 1 and abs(metrics.max_drawdown) < 0.15:
            report += "✅ DEPLOY: Strategy shows strong potential\n"
        elif metrics.sharpe_ratio > 0.5:
            report += "⚠️ OPTIMIZE: Consider parameter tuning\n"
        else:
            report += "❌ REVISE: Strategy needs significant improvement\n"
        
        # Save to file
        if save_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'backtest_report_{timestamp}.txt'
            with open(filename, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {filename}")
        
        return report
    
    def create_visualizations(self, save_plots: bool = True) -> Dict[str, Any]:
        """Create comprehensive backtest visualizations"""
        if not self.results:
            print("❌ No backtest results available for visualization")
            return {}
        
        print("📊 Creating backtest visualizations...")
        
        # Set up plotting style
        plt.style.use('dark_background')
        
        # Create subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Equity Curve
        ax1 = plt.subplot(3, 3, 1)
        equity_values = [entry['value'] for entry in self.capital_history]
        dates = [entry['date'] for entry in self.capital_history]
        
        plt.plot(dates, equity_values, color='#00ff88', linewidth=2, label='Strategy')
        
        # Add benchmark if available
        if self.benchmark_data is not None:
            benchmark_values = []
            initial_benchmark = self.benchmark_data['Close'].iloc[0]
            for date in dates:
                if date in self.benchmark_data.index:
                    benchmark_price = self.benchmark_data.loc[date, 'Close']
                    benchmark_value = self.config.initial_capital * (benchmark_price / initial_benchmark)
                    benchmark_values.append(benchmark_value)
                else:
                    benchmark_values.append(benchmark_values[-1] if benchmark_values else self.config.initial_capital)
            
            plt.plot(dates, benchmark_values, color='#ff6b6b', linewidth=2, label='Benchmark', alpha=0.7)
        
        plt.title('Equity Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = plt.subplot(3, 3, 2)
        running_max = np.maximum.accumulate(equity_values)
        drawdown = (np.array(equity_values) - running_max) / running_max * 100
        
        plt.fill_between(dates, drawdown, 0, color='#ff6b6b', alpha=0.6)
        plt.plot(dates, drawdown, color='#ff4444', linewidth=2)
        plt.title('Drawdown (%)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.grid(True, alpha=0.3)
        
        # 3. Monthly Returns Heatmap
        ax3 = plt.subplot(3, 3, 3)
        monthly_returns = self._calculate_monthly_returns()
        
        if monthly_returns:
            sns.heatmap(monthly_returns, annot=True, fmt='.1%', cmap='RdYlGn', 
                       center=0, ax=ax3, cbar_kws={'label': 'Monthly Return'})
        
        plt.title('Monthly Returns Heatmap', fontsize=14, fontweight='bold')
        
        # 4. Trade Distribution
        ax4 = plt.subplot(3, 3, 4)
        trade_pnls = [t.pnl_pct * 100 for t in self.trade_history]
        
        plt.hist(trade_pnls, bins=50, color='#00ff88', alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='white', linestyle='--', alpha=0.8)
        plt.title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('P&L (%)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 5. Rolling Sharpe Ratio
        ax5 = plt.subplot(3, 3, 5)
        rolling_sharpe = self._calculate_rolling_sharpe()
        
        if rolling_sharpe:
            plt.plot(rolling_sharpe.index, rolling_sharpe.values, color='#ffaa00', linewidth=2)
            plt.axhline(y=0, color='white', linestyle='--', alpha=0.5)
            plt.axhline(y=1, color='#00ff88', linestyle='--', alpha=0.5, label='Good (>1)')
            plt.axhline(y=2, color='#00aa00', linestyle='--', alpha=0.5, label='Excellent (>2)')
        
        plt.title('Rolling 252-Day Sharpe Ratio', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Win Rate by Asset
        ax6 = plt.subplot(3, 3, 6)
        asset_performance = self._calculate_asset_performance()
        
        if asset_performance:
            assets = list(asset_performance.keys())
            win_rates = [asset_performance[asset]['win_rate'] for asset in assets]
            
            colors = ['#00ff88' if wr > 0.5 else '#ff6b6b' for wr in win_rates]
            plt.bar(assets, win_rates, color=colors, alpha=0.7)
            plt.axhline(y=0.5, color='white', linestyle='--', alpha=0.5)
        
        plt.title('Win Rate by Asset', fontsize=14, fontweight='bold')
        plt.xlabel('Asset')
        plt.ylabel('Win Rate')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 7. Trade Duration Distribution
        ax7 = plt.subplot(3, 3, 7)
        trade_durations = [t.duration.days for t in self.trade_history]
        
        plt.hist(trade_durations, bins=30, color='#aa88ff', alpha=0.7, edgecolor='black')
        plt.title('Trade Duration Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Duration (Days)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 8. Cumulative P&L by Trade
        ax8 = plt.subplot(3, 3, 8)
        cumulative_pnl = np.cumsum([t.pnl for t in self.trade_history])
        trade_numbers = list(range(1, len(self.trade_history) + 1))
        
        plt.plot(trade_numbers, cumulative_pnl, color='#00ff88', linewidth=2)
        plt.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        plt.title('Cumulative P&L by Trade', fontsize=14, fontweight='bold')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative P&L ($)')
        plt.grid(True, alpha=0.3)
        
        # 9. Risk-Return Scatter
        ax9 = plt.subplot(3, 3, 9)
        if asset_performance:
            for asset in asset_performance:
                perf = asset_performance[asset]
                plt.scatter(perf['volatility'], perf['return'], 
                          s=100, alpha=0.7, label=asset)
        
        plt.title('Risk-Return by Asset', fontsize=14, fontweight='bold')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        if save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'backtest_analysis_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"📊 Visualizations saved to {filename}")
        
        plt.show()
        
        return {
            'equity_curve': equity_values,
            'drawdown': drawdown,
            'monthly_returns': monthly_returns,
            'trade_distribution': trade_pnls,
            'asset_performance': asset_performance
        }
    
    def _calculate_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns for heatmap"""
        if not self.capital_history:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.capital_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Calculate monthly returns
        monthly_returns = df['value'].resample('M').last().pct_change().dropna()
        
        # Reshape for heatmap
        monthly_returns.index = monthly_returns.index.to_period('M')
        
        # Create pivot table
        monthly_returns_df = monthly_returns.reset_index()
        monthly_returns_df['year'] = monthly_returns_df['date'].dt.year
        monthly_returns_df['month'] = monthly_returns_df['date'].dt.month
        
        pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='value')
        
        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_table.columns = [month_names[i-1] for i in pivot_table.columns]
        
        return pivot_table
    
    def _calculate_rolling_sharpe(self, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        if not self.capital_history:
            return pd.Series()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.capital_history)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Calculate rolling Sharpe
        rolling_returns = df['daily_return'].rolling(window=window)
        rolling_mean = rolling_returns.mean() * 252
        rolling_std = rolling_returns.std() * np.sqrt(252)
        
        rolling_sharpe = (rolling_mean - 0.02) / rolling_std  # Assuming 2% risk-free rate
        
        return rolling_sharpe.dropna()
    
    def _calculate_asset_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by asset"""
        asset_performance = {}
        
        for asset in set(t.asset for t in self.trade_history):
            asset_trades = [t for t in self.trade_history if t.asset == asset]
            
            if asset_trades:
                total_trades = len(asset_trades)
                winning_trades = sum(1 for t in asset_trades if t.pnl > 0)
                win_rate = winning_trades / total_trades
                
                total_return = sum(t.pnl_pct for t in asset_trades)
                returns = [t.pnl_pct for t in asset_trades]
                volatility = np.std(returns) if len(returns) > 1 else 0
                
                asset_performance[asset] = {
                    'win_rate': win_rate,
                    'return': total_return,
                    'volatility': volatility,
                    'total_trades': total_trades
                }
        
        return asset_performance

def example_strategy(current_date, asset_data, portfolio_value, current_positions, params):
    """Example moving average crossover strategy"""
    signals = []
    
    # Parameters
    short_window = params.get('short_window', 10)
    long_window = params.get('long_window', 30)
    
    for asset, data in asset_data.items():
        if current_date not in data.index:
            continue
        
        # Check if we have enough data
        current_idx = data.index.get_loc(current_date)
        if current_idx < long_window:
            continue
        
        # Calculate moving averages
        short_ma = data['Close'].iloc[current_idx-short_window:current_idx+1].mean()
        long_ma = data['Close'].iloc[current_idx-long_window:current_idx+1].mean()
        
        # Previous values
        prev_short_ma = data['Close'].iloc[current_idx-short_window-1:current_idx].mean()
        prev_long_ma = data['Close'].iloc[current_idx-long_window-1:current_idx].mean()
        
        # Generate signals
        if short_ma > long_ma and prev_short_ma <= prev_long_ma:
            # Golden cross - buy signal
            signals.append({
                'asset': asset,
                'action': 'BUY',
                'confidence': 0.7,
                'reason': 'Golden Cross'
            })
        elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
            # Death cross - sell signal
            signals.append({
                'asset': asset,
                'action': 'SELL',
                'confidence': 0.7,
                'reason': 'Death Cross'
            })
    
    return signals

def main():
    """Demo function for backtesting framework"""
    print("🧪 BACKTESTING FRAMEWORK DEMO")
    print("=" * 60)
    
    # Create backtest configuration
    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=100000,
        commission_rate=0.001,
        slippage_rate=0.0005,
        max_position_size=0.2,
        stop_loss_pct=0.05,
        take_profit_pct=0.10
    )
    
    # Initialize backtesting engine
    engine = BacktestEngine(config)
    
    # Test assets
    assets = ['AAPL', 'GOOGL', 'MSFT']
    
    # Run backtest
    print("\n🚀 Running backtest...")
    results = engine.backtest_strategy(
        strategy_func=example_strategy,
        assets=assets,
        strategy_params={'short_window': 10, 'long_window': 30}
    )
    
    if results:
        print("\n📊 Backtest Results:")
        metrics = results['metrics']
        print(f"   Total Return: {metrics.total_return:.2%}")
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"   Win Rate: {metrics.win_rate:.2%}")
        print(f"   Total Trades: {metrics.total_trades}")
        
        # Generate report
        print("\n📋 Generating comprehensive report...")
        report = engine.generate_report()
        print("✅ Report generated")
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        engine.create_visualizations()
        
        # Parameter optimization
        print("\n🔍 Running parameter optimization...")
        optimization_results = engine.parameter_optimization(
            strategy_func=example_strategy,
            assets=assets,
            param_ranges={
                'short_window': [5, 10, 15, 20],
                'long_window': [20, 30, 40, 50]
            }
        )
        
        if optimization_results:
            print(f"   Best parameters: {optimization_results['best_params']}")
            print(f"   Best Sharpe ratio: {optimization_results['best_metric']:.2f}")
        
        # Monte Carlo simulation
        print("\n🎲 Running Monte Carlo simulation...")
        monte_carlo_results = engine.monte_carlo_simulation(
            strategy_func=example_strategy,
            assets=assets,
            strategy_params={'short_window': 10, 'long_window': 30},
            num_simulations=100
        )
        
        if monte_carlo_results:
            print(f"   Robustness score: {monte_carlo_results['robustness_score']:.2f}")
            print(f"   Average return: {monte_carlo_results['statistics']['total_return']['mean']:.2%}")
    
    print("\n" + "=" * 60)
    print("🎯 Backtesting Framework Demo Complete!")

if __name__ == "__main__":
    main() 