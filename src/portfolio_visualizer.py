"""
Advanced Portfolio Visualization System
Creates comprehensive charts, dashboards, and analytics for portfolio performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_value: float
    daily_change: float
    daily_change_pct: float
    total_return: float
    total_return_pct: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    sortino_ratio: float
    calmar_ratio: float
    beta: float
    alpha: float
    var_95: float
    cvar_95: float

class PortfolioVisualizer:
    """Advanced portfolio visualization and analytics"""
    
    def __init__(self, dark_theme=True):
        self.dark_theme = dark_theme
        self.colors = {
            'primary': '#00D4FF',
            'secondary': '#FF6B6B',
            'success': '#4ECDC4',
            'warning': '#FFE66D',
            'danger': '#FF6B6B',
            'background': '#1E1E1E' if dark_theme else '#FFFFFF',
            'text': '#FFFFFF' if dark_theme else '#000000'
        }
        
        # Configure plotly theme
        self.plotly_theme = 'plotly_dark' if dark_theme else 'plotly_white'
        
        # Create output directory
        os.makedirs('visualizations', exist_ok=True)
        
        print("📊 Portfolio Visualizer initialized")
    
    def create_portfolio_performance_chart(self, 
                                         portfolio_data: Dict[str, Any], 
                                         time_series: List[Dict[str, Any]]) -> go.Figure:
        """Create comprehensive portfolio performance chart"""
        
        # Convert time series to DataFrame
        df = pd.DataFrame(time_series)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Portfolio Value Over Time',
                'Asset Allocation',
                'Daily Returns Distribution',
                'Cumulative Returns',
                'Volatility Analysis',
                'Risk Metrics',
                'Performance Comparison',
                'Drawdown Analysis'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "pie"}],
                [{"type": "histogram"}, {"secondary_y": True}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"colspan": 2}, None]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Portfolio Value Over Time
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['total_value'],
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if len(df) > 7:
            df['ma_7'] = df['total_value'].rolling(window=7).mean()
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['ma_7'],
                    mode='lines',
                    name='7-Day MA',
                    line=dict(color=self.colors['warning'], width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # 2. Asset Allocation Pie Chart
        allocation_data = portfolio_data.get('metrics', {}).get('token_allocations', {})
        if allocation_data:
            fig.add_trace(
                go.Pie(
                    labels=list(allocation_data.keys()),
                    values=list(allocation_data.values()),
                    name="Asset Allocation",
                    marker_colors=[self.colors['primary'], self.colors['secondary'], 
                                 self.colors['success'], self.colors['warning'], 
                                 self.colors['danger']]
                ),
                row=1, col=2
            )
        
        # 3. Daily Returns Distribution
        if len(df) > 1:
            df['daily_returns'] = df['total_value'].pct_change()
            fig.add_trace(
                go.Histogram(
                    x=df['daily_returns'].dropna(),
                    name='Daily Returns',
                    marker_color=self.colors['success'],
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # 4. Cumulative Returns
        if len(df) > 1:
            initial_value = df['total_value'].iloc[0]
            df['cumulative_returns'] = (df['total_value'] / initial_value - 1) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['cumulative_returns'],
                    mode='lines+markers',
                    name='Cumulative Returns (%)',
                    line=dict(color=self.colors['success'], width=3),
                    fill='tonexty'
                ),
                row=2, col=2
            )
        
        # 5. Volatility Analysis (Rolling Volatility)
        if len(df) > 14:
            df['rolling_vol'] = df['daily_returns'].rolling(window=14).std() * np.sqrt(252)
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['rolling_vol'],
                    mode='lines',
                    name='14-Day Rolling Volatility',
                    line=dict(color=self.colors['danger'], width=2)
                ),
                row=3, col=1
            )
        
        # 6. Risk Metrics Bar Chart
        risk_metrics = portfolio_data.get('risk_metrics', {})
        if risk_metrics:
            metrics_names = list(risk_metrics.keys())
            metrics_values = list(risk_metrics.values())
            
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=metrics_values,
                    name='Risk Metrics',
                    marker_color=self.colors['warning']
                ),
                row=3, col=2
            )
        
        # 7. Drawdown Analysis
        if len(df) > 1:
            df['peak'] = df['total_value'].expanding().max()
            df['drawdown'] = (df['total_value'] - df['peak']) / df['peak'] * 100
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['drawdown'],
                    mode='lines',
                    name='Drawdown (%)',
                    line=dict(color=self.colors['danger'], width=2),
                    fill='tozeroy'
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'🚀 Portfolio Performance Dashboard - ${portfolio_data.get("metrics", {}).get("total_value", 0):,.2f}',
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['text']}
            },
            template=self.plotly_theme,
            height=1200,
            showlegend=True,
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background']
        )
        
        return fig
    
    def create_trading_analysis_chart(self, 
                                    trading_data: List[Dict[str, Any]], 
                                    performance_metrics: Dict[str, Any]) -> go.Figure:
        """Create comprehensive trading analysis visualization"""
        
        # Convert to DataFrame
        df = pd.DataFrame(trading_data)
        if df.empty:
            return self._create_empty_chart("No trading data available")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Trading Performance Over Time',
                'Win/Loss Distribution',
                'Risk-Reward Analysis',
                'Trade Frequency Analysis',
                'Profit/Loss by Asset',
                'Confidence vs Performance'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Trading Performance Over Time
        df['cumulative_pnl'] = df['pnl'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add trade markers
        winning_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        fig.add_trace(
            go.Scatter(
                x=winning_trades['timestamp'],
                y=winning_trades['cumulative_pnl'],
                mode='markers',
                name='Winning Trades',
                marker=dict(color=self.colors['success'], size=10, symbol='triangle-up')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=losing_trades['timestamp'],
                y=losing_trades['cumulative_pnl'],
                mode='markers',
                name='Losing Trades',
                marker=dict(color=self.colors['danger'], size=10, symbol='triangle-down')
            ),
            row=1, col=1
        )
        
        # 2. Win/Loss Distribution
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        fig.add_trace(
            go.Pie(
                labels=['Winning Trades', 'Losing Trades'],
                values=[win_count, loss_count],
                name="Win/Loss Ratio",
                marker_colors=[self.colors['success'], self.colors['danger']]
            ),
            row=1, col=2
        )
        
        # 3. Risk-Reward Analysis
        fig.add_trace(
            go.Scatter(
                x=df['risk_amount'],
                y=df['pnl'],
                mode='markers',
                name='Risk vs Reward',
                marker=dict(
                    color=df['pnl'],
                    colorscale='RdYlGn',
                    size=10,
                    colorbar=dict(title="P&L", x=0.45)
                ),
                text=df['asset']
            ),
            row=2, col=1
        )
        
        # 4. Trade Frequency by Hour
        df['hour'] = df['timestamp'].dt.hour
        hourly_trades = df.groupby('hour').size()
        
        fig.add_trace(
            go.Bar(
                x=hourly_trades.index,
                y=hourly_trades.values,
                name='Trades per Hour',
                marker_color=self.colors['warning']
            ),
            row=2, col=2
        )
        
        # 5. Profit/Loss by Asset
        asset_pnl = df.groupby('asset')['pnl'].sum().sort_values(ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=asset_pnl.index,
                y=asset_pnl.values,
                name='P&L by Asset',
                marker_color=[self.colors['success'] if x > 0 else self.colors['danger'] for x in asset_pnl.values]
            ),
            row=3, col=1
        )
        
        # 6. Confidence vs Performance
        fig.add_trace(
            go.Scatter(
                x=df['confidence'],
                y=df['pnl'],
                mode='markers',
                name='Confidence vs P&L',
                marker=dict(
                    color=df['pnl'],
                    colorscale='RdYlGn',
                    size=12,
                    opacity=0.7
                ),
                text=df['asset']
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'📈 Trading Analysis Dashboard - Win Rate: {performance_metrics.get("win_rate", 0):.1f}%',
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['text']}
            },
            template=self.plotly_theme,
            height=1000,
            showlegend=True,
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background']
        )
        
        return fig
    
    def create_risk_analytics_dashboard(self, 
                                      portfolio_data: Dict[str, Any], 
                                      risk_metrics: Dict[str, Any]) -> go.Figure:
        """Create comprehensive risk analytics dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Risk Metrics Overview',
                'VaR Analysis',
                'Correlation Matrix',
                'Risk-Return Scatter',
                'Portfolio Beta',
                'Stress Test Results'
            ],
            specs=[
                [{"type": "bar"}, {"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "indicator"}, {"type": "bar"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Risk Metrics Overview
        metrics_names = ['Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Beta', 'Alpha']
        metrics_values = [
            risk_metrics.get('volatility', 0),
            risk_metrics.get('sharpe_ratio', 0),
            risk_metrics.get('max_drawdown', 0),
            risk_metrics.get('beta', 0),
            risk_metrics.get('alpha', 0)
        ]
        
        fig.add_trace(
            go.Bar(
                x=metrics_names,
                y=metrics_values,
                name='Risk Metrics',
                marker_color=[self.colors['primary'], self.colors['success'], 
                             self.colors['danger'], self.colors['warning'], 
                             self.colors['secondary']]
            ),
            row=1, col=1
        )
        
        # 2. VaR Analysis (Monte Carlo simulation)
        returns = np.random.normal(0.001, 0.02, 1000)  # Simulated returns
        portfolio_value = portfolio_data.get('metrics', {}).get('total_value', 100000)
        var_95 = np.percentile(returns, 5) * portfolio_value
        
        fig.add_trace(
            go.Histogram(
                x=returns * portfolio_value,
                name='Simulated Returns',
                marker_color=self.colors['primary'],
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Add VaR line
        fig.add_vline(
            x=var_95,
            line=dict(color=self.colors['danger'], width=3, dash='dash'),
            annotation_text=f'VaR 95%: ${var_95:,.0f}',
            row=1, col=2
        )
        
        # 3. Correlation Matrix (simulated)
        assets = ['BTC', 'ETH', 'SUI', 'SOL', 'SEI']
        corr_matrix = np.random.rand(5, 5)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1)  # Set diagonal to 1
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix,
                x=assets,
                y=assets,
                colorscale='RdYlBu',
                text=np.round(corr_matrix, 2),
                texttemplate="%{text}",
                textfont={"size": 12},
                name='Correlation Matrix'
            ),
            row=1, col=3
        )
        
        # 4. Risk-Return Scatter
        expected_returns = np.random.normal(0.05, 0.03, len(assets))
        expected_risks = np.random.normal(0.15, 0.05, len(assets))
        
        fig.add_trace(
            go.Scatter(
                x=expected_risks,
                y=expected_returns,
                mode='markers+text',
                name='Risk-Return Profile',
                marker=dict(
                    size=15,
                    color=self.colors['primary'],
                    opacity=0.8
                ),
                text=assets,
                textposition="middle right"
            ),
            row=2, col=1
        )
        
        # 5. Portfolio Beta Indicator
        beta_value = risk_metrics.get('beta', 1.0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=beta_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Portfolio Beta"},
                delta={'reference': 1.0},
                gauge={
                    'axis': {'range': [0, 2]},
                    'bar': {'color': self.colors['primary']},
                    'steps': [
                        {'range': [0, 0.5], 'color': self.colors['success']},
                        {'range': [0.5, 1.5], 'color': self.colors['warning']},
                        {'range': [1.5, 2], 'color': self.colors['danger']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ),
            row=2, col=2
        )
        
        # 6. Stress Test Results
        stress_scenarios = ['Market Crash', 'Interest Rate Hike', 'Inflation Spike', 'Liquidity Crisis']
        stress_impacts = [-0.25, -0.15, -0.10, -0.30]  # Simulated impacts
        
        fig.add_trace(
            go.Bar(
                x=stress_scenarios,
                y=stress_impacts,
                name='Stress Test Impact',
                marker_color=self.colors['danger']
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '⚖️ Risk Analytics Dashboard',
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['text']}
            },
            template=self.plotly_theme,
            height=800,
            showlegend=True,
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background']
        )
        
        return fig
    
    def create_market_analysis_chart(self, 
                                   market_data: Dict[str, Any], 
                                   asset_data: Dict[str, List[Dict[str, Any]]]) -> go.Figure:
        """Create comprehensive market analysis visualization"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Asset Price Movements',
                'Volume Analysis',
                'Market Sentiment',
                'Technical Indicators'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "bar"}],
                [{"type": "scatter"}, {"secondary_y": True}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # 1. Asset Price Movements
        colors_list = [self.colors['primary'], self.colors['secondary'], 
                      self.colors['success'], self.colors['warning'], 
                      self.colors['danger']]
        
        for i, (asset, data) in enumerate(asset_data.items()):
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['price'],
                        mode='lines',
                        name=f'{asset} Price',
                        line=dict(color=colors_list[i % len(colors_list)], width=2)
                    ),
                    row=1, col=1
                )
        
        # 2. Volume Analysis
        volume_data = market_data.get('volume_analysis', {})
        if volume_data:
            assets = list(volume_data.keys())
            volumes = list(volume_data.values())
            
            fig.add_trace(
                go.Bar(
                    x=assets,
                    y=volumes,
                    name='24h Volume',
                    marker_color=self.colors['primary']
                ),
                row=1, col=2
            )
        
        # 3. Market Sentiment
        sentiment_data = market_data.get('sentiment', {})
        if sentiment_data:
            sentiment_scores = list(sentiment_data.values())
            sentiment_labels = list(sentiment_data.keys())
            
            fig.add_trace(
                go.Scatter(
                    x=sentiment_labels,
                    y=sentiment_scores,
                    mode='markers+lines',
                    name='Market Sentiment',
                    marker=dict(
                        size=15,
                        color=sentiment_scores,
                        colorscale='RdYlGn',
                        showscale=True
                    ),
                    line=dict(color=self.colors['primary'], width=3)
                ),
                row=2, col=1
            )
        
        # 4. Technical Indicators
        indicators = market_data.get('technical_indicators', {})
        if indicators:
            for i, (indicator, value) in enumerate(indicators.items()):
                fig.add_trace(
                    go.Scatter(
                        x=[indicator],
                        y=[value],
                        mode='markers+text',
                        name=f'{indicator}',
                        marker=dict(
                            size=20,
                            color=colors_list[i % len(colors_list)]
                        ),
                        text=[f'{value:.2f}'],
                        textposition="middle center"
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '📊 Market Analysis Dashboard',
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['text']}
            },
            template=self.plotly_theme,
            height=700,
            showlegend=True,
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background']
        )
        
        return fig
    
    def create_security_monitoring_dashboard(self, 
                                           security_events: List[Dict[str, Any]], 
                                           security_metrics: Dict[str, Any]) -> go.Figure:
        """Create security monitoring dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Security Events Timeline',
                'Event Types Distribution',
                'Security Levels',
                'Threat Detection Status'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        if security_events:
            # Convert to DataFrame
            df = pd.DataFrame(security_events)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 1. Security Events Timeline
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df.index,
                    mode='markers',
                    name='Security Events',
                    marker=dict(
                        size=10,
                        color=[self.colors['success'] if success else self.colors['danger'] 
                              for success in df['success']],
                        symbol='circle'
                    ),
                    text=df['action_description'],
                    hovertemplate='<b>%{text}</b><br>Time: %{x}<br>Success: %{marker.color}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 2. Event Types Distribution
            event_types = df['action_type'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=event_types.index,
                    values=event_types.values,
                    name="Event Types",
                    marker_colors=[self.colors['primary'], self.colors['secondary'], 
                                 self.colors['success'], self.colors['warning']]
                ),
                row=1, col=2
            )
            
            # 3. Security Levels
            security_levels = df['security_level'].value_counts()
            fig.add_trace(
                go.Bar(
                    x=security_levels.index,
                    y=security_levels.values,
                    name='Security Levels',
                    marker_color=[
                        self.colors['success'] if level == 'LOW' else
                        self.colors['warning'] if level == 'MEDIUM' else
                        self.colors['danger'] if level == 'HIGH' else
                        self.colors['danger']
                        for level in security_levels.index
                    ]
                ),
                row=2, col=1
            )
        
        # 4. Threat Detection Status
        threat_level = security_metrics.get('threat_level', 'LOW')
        threat_score = {'LOW': 0.2, 'MEDIUM': 0.5, 'HIGH': 0.8, 'CRITICAL': 1.0}.get(threat_level, 0.2)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=threat_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Threat Level: {threat_level}"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': self.colors['primary']},
                    'steps': [
                        {'range': [0, 0.25], 'color': self.colors['success']},
                        {'range': [0.25, 0.5], 'color': self.colors['warning']},
                        {'range': [0.5, 0.75], 'color': self.colors['danger']},
                        {'range': [0.75, 1], 'color': '#8B0000'}
                    ]
                }
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': '🛡️ Security Monitoring Dashboard',
                'x': 0.5,
                'font': {'size': 20, 'color': self.colors['text']}
            },
            template=self.plotly_theme,
            height=700,
            showlegend=True,
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background']
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=20, color=self.colors['text'])
        )
        
        fig.update_layout(
            template=self.plotly_theme,
            height=400,
            paper_bgcolor=self.colors['background'],
            plot_bgcolor=self.colors['background']
        )
        
        return fig
    
    def save_visualization(self, fig: go.Figure, filename: str, format: str = 'html'):
        """Save visualization to file"""
        filepath = f'visualizations/{filename}.{format}'
        
        if format == 'html':
            fig.write_html(filepath)
        elif format == 'png':
            fig.write_image(filepath)
        elif format == 'pdf':
            fig.write_image(filepath)
        
        print(f"📊 Visualization saved: {filepath}")
        return filepath
    
    def create_comprehensive_report(self, 
                                  portfolio_data: Dict[str, Any],
                                  trading_data: List[Dict[str, Any]],
                                  market_data: Dict[str, Any],
                                  security_data: Dict[str, Any]) -> Dict[str, str]:
        """Create comprehensive visualization report"""
        
        print("🎨 Creating comprehensive visualization report...")
        
        visualizations = {}
        
        # 1. Portfolio Performance
        portfolio_fig = self.create_portfolio_performance_chart(
            portfolio_data, 
            trading_data  # Using trading data as time series
        )
        visualizations['portfolio_performance'] = self.save_visualization(
            portfolio_fig, 'portfolio_performance'
        )
        
        # 2. Trading Analysis
        if trading_data:
            trading_fig = self.create_trading_analysis_chart(
                trading_data, 
                portfolio_data.get('performance_metrics', {})
            )
            visualizations['trading_analysis'] = self.save_visualization(
                trading_fig, 'trading_analysis'
            )
        
        # 3. Risk Analytics
        risk_fig = self.create_risk_analytics_dashboard(
            portfolio_data,
            portfolio_data.get('risk_metrics', {})
        )
        visualizations['risk_analytics'] = self.save_visualization(
            risk_fig, 'risk_analytics'
        )
        
        # 4. Market Analysis
        market_fig = self.create_market_analysis_chart(
            market_data,
            market_data.get('assets', {})
        )
        visualizations['market_analysis'] = self.save_visualization(
            market_fig, 'market_analysis'
        )
        
        # 5. Security Monitoring
        security_fig = self.create_security_monitoring_dashboard(
            security_data.get('events', []),
            security_data.get('metrics', {})
        )
        visualizations['security_monitoring'] = self.save_visualization(
            security_fig, 'security_monitoring'
        )
        
        print(f"✅ Created {len(visualizations)} comprehensive visualizations")
        return visualizations


def main():
    """Demo the portfolio visualizer"""
    print("🎨 PORTFOLIO VISUALIZER DEMO")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = PortfolioVisualizer()
    
    # Create demo data
    demo_portfolio = {
        'metrics': {
            'total_value': 125000.0,
            'daily_change': 2500.0,
            'daily_change_pct': 2.0,
            'token_allocations': {
                'BTC': 45.0,
                'ETH': 25.0,
                'SUI': 15.0,
                'SOL': 10.0,
                'SEI': 5.0
            }
        },
        'risk_metrics': {
            'volatility': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'beta': 1.1,
            'alpha': 0.03
        }
    }
    
    demo_trading = [
        {
            'timestamp': '2024-01-15T10:00:00Z',
            'asset': 'BTC',
            'action': 'BUY',
            'pnl': 150.0,
            'risk_amount': 100.0,
            'confidence': 0.85,
            'total_value': 120000.0
        },
        {
            'timestamp': '2024-01-15T14:30:00Z',
            'asset': 'SUI',
            'action': 'SELL',
            'pnl': -50.0,
            'risk_amount': 75.0,
            'confidence': 0.65,
            'total_value': 119950.0
        }
    ]
    
    demo_market = {
        'volume_analysis': {
            'BTC': 1500000000,
            'ETH': 800000000,
            'SUI': 50000000,
            'SOL': 200000000
        },
        'sentiment': {
            'BTC': 0.7,
            'ETH': 0.6,
            'SUI': 0.8,
            'SOL': 0.5
        },
        'technical_indicators': {
            'RSI': 65,
            'MACD': 0.02,
            'BB_Upper': 118500,
            'BB_Lower': 115000
        }
    }
    
    demo_security = {
        'events': [
            {
                'timestamp': '2024-01-15T10:00:00Z',
                'action_type': 'AUTHENTICATION',
                'action_description': 'User login successful',
                'security_level': 'LOW',
                'success': True
            },
            {
                'timestamp': '2024-01-15T14:30:00Z',
                'action_type': 'TRADE_EXECUTION',
                'action_description': 'Trade executed successfully',
                'security_level': 'MEDIUM',
                'success': True
            }
        ],
        'metrics': {
            'threat_level': 'LOW'
        }
    }
    
    # Create comprehensive report
    visualizations = visualizer.create_comprehensive_report(
        demo_portfolio,
        demo_trading,
        demo_market,
        demo_security
    )
    
    print("\n🎉 VISUALIZATION REPORT CREATED!")
    print("=" * 60)
    for name, filepath in visualizations.items():
        print(f"📊 {name}: {filepath}")
    
    print("\n✅ Open the HTML files in your browser to view the interactive dashboards!")


if __name__ == "__main__":
    main() 