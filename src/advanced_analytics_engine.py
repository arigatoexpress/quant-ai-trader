"""
Advanced Analytics Engine
Provides comprehensive insights, recommendations, and performance analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceInsight:
    """Performance insight with recommendations"""
    insight_type: str
    title: str
    description: str
    confidence: float
    impact_score: float
    recommendation: str
    action_items: List[str]
    metrics: Dict[str, float]
    priority: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
    category: str = "PERFORMANCE"  # PERFORMANCE, RISK, OPPORTUNITY, SECURITY

@dataclass
class MarketOpportunity:
    """Market opportunity analysis"""
    asset: str
    opportunity_type: str
    description: str
    confidence: float
    potential_return: float
    risk_level: str
    time_horizon: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    supporting_data: Dict[str, Any]

@dataclass
class TradingRecommendation:
    """AI-powered trading recommendation"""
    asset: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str
    target_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float
    risk_reward_ratio: float
    timeframe: str
    conditions: List[str]
    supporting_metrics: Dict[str, float]

class AdvancedAnalyticsEngine:
    """Advanced analytics engine for comprehensive insights"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.market_clusters = KMeans(n_clusters=5, random_state=42)
        
        # Analytics cache
        self.insights_cache = {}
        self.recommendations_cache = {}
        self.last_analysis_time = None
        
        print("🧠 Advanced Analytics Engine initialized")
    
    def analyze_portfolio_performance(self, 
                                    portfolio_data: Dict[str, Any], 
                                    historical_data: List[Dict[str, Any]]) -> List[PerformanceInsight]:
        """Comprehensive portfolio performance analysis"""
        
        insights = []
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(historical_data)
        if df.empty:
            return insights
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # 1. Performance Trend Analysis
        if len(df) > 1:
            insights.extend(self._analyze_performance_trends(df))
        
        # 2. Risk-Adjusted Returns
        insights.extend(self._analyze_risk_adjusted_returns(df, portfolio_data))
        
        # 3. Volatility Analysis
        insights.extend(self._analyze_volatility_patterns(df))
        
        # 4. Drawdown Analysis
        insights.extend(self._analyze_drawdown_characteristics(df))
        
        # 5. Asset Allocation Analysis
        insights.extend(self._analyze_asset_allocation(portfolio_data))
        
        # 6. Performance Attribution
        insights.extend(self._analyze_performance_attribution(df, portfolio_data))
        
        # Sort by impact score
        insights.sort(key=lambda x: x.impact_score, reverse=True)
        
        return insights
    
    def _analyze_performance_trends(self, df: pd.DataFrame) -> List[PerformanceInsight]:
        """Analyze performance trends"""
        insights = []
        
        # Calculate returns
        df['daily_return'] = df['total_value'].pct_change()
        df['cumulative_return'] = (df['total_value'] / df['total_value'].iloc[0] - 1) * 100
        
        # Trend analysis
        recent_30d = df.tail(30)
        recent_7d = df.tail(7)
        
        if len(recent_30d) > 1:
            monthly_return = recent_30d['cumulative_return'].iloc[-1] - recent_30d['cumulative_return'].iloc[0]
            weekly_return = recent_7d['cumulative_return'].iloc[-1] - recent_7d['cumulative_return'].iloc[0] if len(recent_7d) > 1 else 0
            
            # Momentum insight
            if monthly_return > 5:
                insights.append(PerformanceInsight(
                    insight_type="MOMENTUM",
                    title="Strong Positive Momentum",
                    description=f"Portfolio shows strong positive momentum with {monthly_return:.1f}% return over 30 days",
                    confidence=0.8,
                    impact_score=8.5,
                    recommendation="Consider taking partial profits or rebalancing to maintain momentum",
                    action_items=[
                        "Review position sizes",
                        "Consider profit-taking opportunities",
                        "Monitor for momentum exhaustion signals"
                    ],
                    metrics={
                        "monthly_return": monthly_return,
                        "weekly_return": weekly_return,
                        "momentum_score": min(monthly_return / 10, 1.0)
                    },
                    priority="HIGH",
                    category="OPPORTUNITY"
                ))
            elif monthly_return < -5:
                insights.append(PerformanceInsight(
                    insight_type="DOWNTREND",
                    title="Negative Performance Trend",
                    description=f"Portfolio experiencing downtrend with {monthly_return:.1f}% decline over 30 days",
                    confidence=0.8,
                    impact_score=9.0,
                    recommendation="Implement defensive strategies and review risk management",
                    action_items=[
                        "Reduce position sizes",
                        "Increase cash allocation",
                        "Review stop-loss levels",
                        "Consider hedging strategies"
                    ],
                    metrics={
                        "monthly_return": monthly_return,
                        "weekly_return": weekly_return,
                        "downtrend_severity": min(abs(monthly_return) / 10, 1.0)
                    },
                    priority="CRITICAL",
                    category="RISK"
                ))
        
        return insights
    
    def _analyze_risk_adjusted_returns(self, df: pd.DataFrame, portfolio_data: Dict[str, Any]) -> List[PerformanceInsight]:
        """Analyze risk-adjusted returns"""
        insights = []
        
        if len(df) < 30:
            return insights
        
        # Calculate risk metrics
        daily_returns = df['total_value'].pct_change().dropna()
        
        # Sharpe ratio
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        # Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        sortino_ratio = (daily_returns.mean() / negative_returns.std()) * np.sqrt(252) if len(negative_returns) > 0 and negative_returns.std() > 0 else 0
        
        # Calmar ratio
        max_drawdown = self._calculate_max_drawdown(df['total_value'])
        calmar_ratio = (daily_returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sharpe ratio insight
        if sharpe_ratio > 1.5:
            insights.append(PerformanceInsight(
                insight_type="RISK_ADJUSTED_PERFORMANCE",
                title="Excellent Risk-Adjusted Returns",
                description=f"Portfolio demonstrates excellent risk-adjusted performance with Sharpe ratio of {sharpe_ratio:.2f}",
                confidence=0.9,
                impact_score=8.0,
                recommendation="Maintain current strategy while gradually increasing position sizes",
                action_items=[
                    "Consider increasing allocation to high-performing assets",
                    "Document successful strategies for replication",
                    "Monitor for strategy degradation"
                ],
                metrics={
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio,
                    "calmar_ratio": calmar_ratio
                },
                priority="HIGH",
                category="PERFORMANCE"
            ))
        elif sharpe_ratio < 0.5:
            insights.append(PerformanceInsight(
                insight_type="RISK_ADJUSTED_PERFORMANCE",
                title="Poor Risk-Adjusted Returns",
                description=f"Portfolio shows poor risk-adjusted performance with Sharpe ratio of {sharpe_ratio:.2f}",
                confidence=0.9,
                impact_score=9.5,
                recommendation="Urgent review of risk management and strategy optimization needed",
                action_items=[
                    "Reduce position sizes immediately",
                    "Review and tighten risk management rules",
                    "Consider strategy overhaul",
                    "Implement stricter stop-loss levels"
                ],
                metrics={
                    "sharpe_ratio": sharpe_ratio,
                    "sortino_ratio": sortino_ratio,
                    "calmar_ratio": calmar_ratio
                },
                priority="CRITICAL",
                category="RISK"
            ))
        
        return insights
    
    def _analyze_volatility_patterns(self, df: pd.DataFrame) -> List[PerformanceInsight]:
        """Analyze volatility patterns"""
        insights = []
        
        if len(df) < 14:
            return insights
        
        # Calculate rolling volatility
        daily_returns = df['total_value'].pct_change().dropna()
        rolling_vol = daily_returns.rolling(window=14).std() * np.sqrt(252)
        
        current_vol = rolling_vol.iloc[-1]
        avg_vol = rolling_vol.mean()
        vol_percentile = (rolling_vol <= current_vol).sum() / len(rolling_vol) * 100
        
        # Volatility regime change
        if current_vol > avg_vol * 1.5:
            insights.append(PerformanceInsight(
                insight_type="VOLATILITY_REGIME",
                title="High Volatility Regime",
                description=f"Portfolio experiencing high volatility regime ({current_vol:.1%} vs {avg_vol:.1%} average)",
                confidence=0.85,
                impact_score=7.5,
                recommendation="Adjust position sizes and implement volatility-based risk management",
                action_items=[
                    "Reduce position sizes by 25-50%",
                    "Implement volatility-based stop losses",
                    "Consider volatility hedging strategies",
                    "Monitor for volatility mean reversion"
                ],
                metrics={
                    "current_volatility": current_vol,
                    "average_volatility": avg_vol,
                    "volatility_percentile": vol_percentile
                },
                priority="HIGH",
                category="RISK"
            ))
        elif current_vol < avg_vol * 0.5:
            insights.append(PerformanceInsight(
                insight_type="VOLATILITY_REGIME",
                title="Low Volatility Environment",
                description=f"Portfolio in low volatility regime ({current_vol:.1%} vs {avg_vol:.1%} average)",
                confidence=0.85,
                impact_score=6.0,
                recommendation="Consider increasing position sizes in low volatility environment",
                action_items=[
                    "Gradually increase position sizes",
                    "Look for mean reversion opportunities",
                    "Prepare for potential volatility expansion",
                    "Monitor volatility indicators"
                ],
                metrics={
                    "current_volatility": current_vol,
                    "average_volatility": avg_vol,
                    "volatility_percentile": vol_percentile
                },
                priority="MEDIUM",
                category="OPPORTUNITY"
            ))
        
        return insights
    
    def _analyze_drawdown_characteristics(self, df: pd.DataFrame) -> List[PerformanceInsight]:
        """Analyze drawdown characteristics"""
        insights = []
        
        if len(df) < 10:
            return insights
        
        # Calculate drawdown
        peak = df['total_value'].expanding().max()
        drawdown = (df['total_value'] - peak) / peak * 100
        
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1]
        
        # Drawdown recovery analysis
        if current_drawdown < -10:
            insights.append(PerformanceInsight(
                insight_type="DRAWDOWN_ANALYSIS",
                title="Significant Drawdown Alert",
                description=f"Portfolio in significant drawdown of {current_drawdown:.1f}% (max: {max_drawdown:.1f}%)",
                confidence=0.95,
                impact_score=9.0,
                recommendation="Implement immediate risk reduction and recovery strategies",
                action_items=[
                    "Halt new position openings",
                    "Review and close unprofitable positions",
                    "Implement strict risk management",
                    "Consider portfolio restructuring"
                ],
                metrics={
                    "current_drawdown": current_drawdown,
                    "max_drawdown": max_drawdown,
                    "drawdown_duration": len(drawdown[drawdown < -5])
                },
                priority="CRITICAL",
                category="RISK"
            ))
        elif current_drawdown > -2 and max_drawdown < -5:
            insights.append(PerformanceInsight(
                insight_type="DRAWDOWN_RECOVERY",
                title="Drawdown Recovery",
                description=f"Portfolio recovering from drawdown (current: {current_drawdown:.1f}%, max: {max_drawdown:.1f}%)",
                confidence=0.8,
                impact_score=7.0,
                recommendation="Gradually increase position sizes as recovery continues",
                action_items=[
                    "Monitor recovery sustainability",
                    "Gradually increase position sizes",
                    "Document lessons learned",
                    "Strengthen risk management"
                ],
                metrics={
                    "current_drawdown": current_drawdown,
                    "max_drawdown": max_drawdown,
                    "recovery_percentage": (current_drawdown / max_drawdown) * 100
                },
                priority="MEDIUM",
                category="OPPORTUNITY"
            ))
        
        return insights
    
    def _analyze_asset_allocation(self, portfolio_data: Dict[str, Any]) -> List[PerformanceInsight]:
        """Analyze asset allocation"""
        insights = []
        
        metrics = portfolio_data.get('metrics', {})
        allocations = metrics.get('token_allocations', {})
        
        if not allocations:
            return insights
        
        # Concentration analysis
        max_allocation = max(allocations.values())
        allocation_std = np.std(list(allocations.values()))
        
        if max_allocation > 50:
            insights.append(PerformanceInsight(
                insight_type="CONCENTRATION_RISK",
                title="High Concentration Risk",
                description=f"Portfolio concentrated in single asset ({max_allocation:.1f}% allocation)",
                confidence=0.9,
                impact_score=8.0,
                recommendation="Diversify portfolio to reduce concentration risk",
                action_items=[
                    "Reduce allocation to overweight positions",
                    "Increase allocation to underweight assets",
                    "Consider adding new assets for diversification",
                    "Implement rebalancing strategy"
                ],
                metrics={
                    "max_allocation": max_allocation,
                    "allocation_std": allocation_std,
                    "concentration_score": max_allocation / 20  # Normalize to 0-5 scale
                },
                priority="HIGH",
                category="RISK"
            ))
        
        # Diversification analysis
        num_assets = len(allocations)
        if num_assets < 3:
            insights.append(PerformanceInsight(
                insight_type="DIVERSIFICATION",
                title="Low Diversification",
                description=f"Portfolio contains only {num_assets} assets, increasing risk",
                confidence=0.85,
                impact_score=7.5,
                recommendation="Increase diversification by adding more assets",
                action_items=[
                    "Research additional assets for portfolio",
                    "Consider different asset classes",
                    "Implement correlation analysis",
                    "Plan gradual diversification strategy"
                ],
                metrics={
                    "num_assets": num_assets,
                    "diversification_score": num_assets / 10,  # Normalize
                    "allocation_variance": allocation_std
                },
                priority="MEDIUM",
                category="RISK"
            ))
        
        return insights
    
    def _analyze_performance_attribution(self, df: pd.DataFrame, portfolio_data: Dict[str, Any]) -> List[PerformanceInsight]:
        """Analyze performance attribution"""
        insights = []
        
        # This would require detailed position-level data
        # For now, provide general attribution insights
        
        allocations = portfolio_data.get('metrics', {}).get('token_allocations', {})
        if not allocations:
            return insights
        
        # Top performer analysis
        top_asset = max(allocations, key=allocations.get)
        top_allocation = allocations[top_asset]
        
        insights.append(PerformanceInsight(
            insight_type="PERFORMANCE_ATTRIBUTION",
            title="Top Contributing Asset",
            description=f"{top_asset} is the largest position ({top_allocation:.1f}% allocation)",
            confidence=0.7,
            impact_score=6.0,
            recommendation="Monitor top performer for continued strength or signs of weakness",
            action_items=[
                f"Monitor {top_asset} technical indicators",
                "Set appropriate profit-taking levels",
                "Consider position sizing adjustments",
                "Prepare rebalancing strategy"
            ],
            metrics={
                "top_asset": top_asset,
                "top_allocation": top_allocation,
                "contribution_estimate": top_allocation / 100
            },
            priority="MEDIUM",
            category="PERFORMANCE"
        ))
        
        return insights
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return drawdown.min()
    
    def generate_trading_recommendations(self, 
                                       market_data: Dict[str, Any],
                                       portfolio_data: Dict[str, Any],
                                       trading_history: List[Dict[str, Any]]) -> List[TradingRecommendation]:
        """Generate AI-powered trading recommendations"""
        
        recommendations = []
        
        # Analyze each asset
        assets = ['BTC', 'ETH', 'SUI', 'SOL', 'SEI']
        
        for asset in assets:
            recommendation = self._generate_asset_recommendation(asset, market_data, portfolio_data, trading_history)
            if recommendation:
                recommendations.append(recommendation)
        
        # Sort by confidence and potential return
        recommendations.sort(key=lambda x: (x.confidence * x.risk_reward_ratio), reverse=True)
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _generate_asset_recommendation(self, 
                                     asset: str,
                                     market_data: Dict[str, Any],
                                     portfolio_data: Dict[str, Any],
                                     trading_history: List[Dict[str, Any]]) -> Optional[TradingRecommendation]:
        """Generate recommendation for specific asset"""
        
        # Simulate market analysis (in real implementation, use actual market data)
        np.random.seed(hash(asset) % 1000)
        
        # Mock technical indicators
        rsi = np.random.uniform(30, 70)
        macd = np.random.uniform(-0.05, 0.05)
        price_momentum = np.random.uniform(-0.1, 0.1)
        volume_trend = np.random.uniform(0.5, 2.0)
        
        # Current allocation
        current_allocation = portfolio_data.get('metrics', {}).get('token_allocations', {}).get(asset, 0)
        
        # Generate recommendation logic
        if rsi < 40 and macd > 0 and price_momentum > 0.02:
            # Bullish signal
            action = "BUY"
            confidence = 0.75 + (40 - rsi) / 100
            target_price = 1.05 + price_momentum
            stop_loss = 0.95
            reasoning = f"Oversold RSI ({rsi:.1f}), positive MACD, strong momentum"
            
        elif rsi > 60 and macd < 0 and current_allocation > 20:
            # Bearish signal for overweight position
            action = "SELL"
            confidence = 0.65 + (rsi - 60) / 100
            target_price = 0.95 + price_momentum
            stop_loss = 1.05
            reasoning = f"Overbought RSI ({rsi:.1f}), negative MACD, overweight position"
            
        else:
            # Hold
            action = "HOLD"
            confidence = 0.5
            target_price = None
            stop_loss = None
            reasoning = "No clear directional signal, maintain current position"
        
        # Calculate metrics
        if action != "HOLD":
            risk_reward = abs(target_price - 1.0) / abs(stop_loss - 1.0) if stop_loss else 1.0
            position_size = min(0.1, max(0.02, confidence * 0.1))  # 2-10% based on confidence
        else:
            risk_reward = 1.0
            position_size = 0.0
        
        return TradingRecommendation(
            asset=asset,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            target_price=target_price,
            stop_loss=stop_loss,
            take_profit=target_price * 1.02 if target_price and target_price > 1.0 else None,
            position_size=position_size,
            risk_reward_ratio=risk_reward,
            timeframe="1-7 days",
            conditions=[
                f"RSI: {rsi:.1f}",
                f"MACD: {macd:.3f}",
                f"Momentum: {price_momentum:.2%}",
                f"Volume Trend: {volume_trend:.1f}x"
            ],
            supporting_metrics={
                "rsi": rsi,
                "macd": macd,
                "price_momentum": price_momentum,
                "volume_trend": volume_trend,
                "current_allocation": current_allocation
            }
        )
    
    def detect_market_opportunities(self, 
                                  market_data: Dict[str, Any],
                                  portfolio_data: Dict[str, Any]) -> List[MarketOpportunity]:
        """Detect market opportunities"""
        
        opportunities = []
        
        # Simulate opportunity detection
        assets = ['BTC', 'ETH', 'SUI', 'SOL', 'SEI']
        
        for asset in assets:
            # Generate mock opportunity
            np.random.seed(hash(asset + "opportunity") % 1000)
            
            opportunity_types = ['BREAKOUT', 'REVERSAL', 'MOMENTUM', 'MEAN_REVERSION', 'ARBITRAGE']
            opp_type = np.random.choice(opportunity_types)
            
            confidence = np.random.uniform(0.6, 0.9)
            potential_return = np.random.uniform(0.03, 0.15)
            
            # Generate opportunity based on type
            if opp_type == 'BREAKOUT':
                description = f"{asset} showing breakout pattern above key resistance"
                entry_conditions = ["Price breaks above resistance", "Volume confirmation", "RSI > 50"]
                exit_conditions = ["Profit target reached", "Volume divergence", "Resistance becomes support"]
                
            elif opp_type == 'REVERSAL':
                description = f"{asset} showing potential reversal from oversold levels"
                entry_conditions = ["RSI < 30", "Bullish divergence", "Support level hold"]
                exit_conditions = ["RSI > 70", "Resistance level", "Momentum weakening"]
                
            elif opp_type == 'MOMENTUM':
                description = f"{asset} momentum acceleration opportunity"
                entry_conditions = ["Strong momentum", "Volume expansion", "Trend continuation"]
                exit_conditions = ["Momentum exhaustion", "Volume decline", "Technical divergence"]
                
            elif opp_type == 'MEAN_REVERSION':
                description = f"{asset} mean reversion opportunity from extreme levels"
                entry_conditions = ["Price at extreme", "RSI oversold/overbought", "Volume spike"]
                exit_conditions = ["Return to mean", "Momentum reversal", "Volume normalization"]
                
            else:  # ARBITRAGE
                description = f"{asset} arbitrage opportunity across exchanges"
                entry_conditions = ["Price discrepancy", "Liquidity available", "Execution capability"]
                exit_conditions = ["Price convergence", "Liquidity exhaustion", "Time limit"]
            
            opportunity = MarketOpportunity(
                asset=asset,
                opportunity_type=opp_type,
                description=description,
                confidence=confidence,
                potential_return=potential_return,
                risk_level="MEDIUM" if confidence > 0.7 else "HIGH",
                time_horizon="1-3 days" if opp_type in ['BREAKOUT', 'MOMENTUM'] else "1-7 days",
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                supporting_data={
                    "confidence_score": confidence,
                    "return_potential": potential_return,
                    "risk_score": 1 - confidence,
                    "opportunity_score": confidence * potential_return
                }
            )
            
            opportunities.append(opportunity)
        
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x.confidence * x.potential_return, reverse=True)
        
        return opportunities[:3]  # Return top 3 opportunities
    
    def perform_anomaly_detection(self, 
                                 portfolio_data: Dict[str, Any],
                                 trading_data: List[Dict[str, Any]]) -> List[PerformanceInsight]:
        """Detect anomalies in portfolio performance"""
        
        insights = []
        
        if not trading_data or len(trading_data) < 10:
            return insights
        
        # Convert to DataFrame
        df = pd.DataFrame(trading_data)
        
        # Prepare features for anomaly detection
        features = ['total_value', 'daily_change', 'daily_change_pct']
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            return insights
        
        # Fit anomaly detector
        X = df[available_features].fillna(0)
        if len(X) < 10:
            return insights
        
        X_scaled = self.scaler.fit_transform(X)
        anomaly_scores = self.anomaly_detector.fit_predict(X_scaled)
        
        # Find anomalies
        anomalies = df[anomaly_scores == -1]
        
        if len(anomalies) > 0:
            insights.append(PerformanceInsight(
                insight_type="ANOMALY_DETECTION",
                title="Performance Anomalies Detected",
                description=f"Detected {len(anomalies)} anomalous performance periods",
                confidence=0.7,
                impact_score=6.5,
                recommendation="Review anomalous periods for potential improvements or risks",
                action_items=[
                    "Investigate root causes of anomalies",
                    "Review trading decisions during anomalous periods",
                    "Adjust risk management if needed",
                    "Document lessons learned"
                ],
                metrics={
                    "anomaly_count": len(anomalies),
                    "anomaly_percentage": len(anomalies) / len(df) * 100,
                    "last_anomaly_days_ago": (datetime.now() - pd.to_datetime(anomalies.iloc[-1]['timestamp'])).days if 'timestamp' in anomalies.columns else 0
                },
                priority="MEDIUM",
                category="RISK"
            ))
        
        return insights
    
    def generate_comprehensive_report(self, 
                                    portfolio_data: Dict[str, Any],
                                    trading_data: List[Dict[str, Any]],
                                    market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        
        print("📊 Generating comprehensive analytics report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_insights": self.analyze_portfolio_performance(portfolio_data, trading_data),
            "trading_recommendations": self.generate_trading_recommendations(market_data, portfolio_data, trading_data),
            "market_opportunities": self.detect_market_opportunities(market_data, portfolio_data),
            "anomaly_analysis": self.perform_anomaly_detection(portfolio_data, trading_data),
            "summary_metrics": {
                "total_insights": 0,
                "high_priority_insights": 0,
                "trading_opportunities": 0,
                "risk_alerts": 0,
                "performance_score": 0.0
            }
        }
        
        # Calculate summary metrics
        all_insights = report["portfolio_insights"] + report["anomaly_analysis"]
        report["summary_metrics"]["total_insights"] = len(all_insights)
        report["summary_metrics"]["high_priority_insights"] = len([i for i in all_insights if i.priority in ['HIGH', 'CRITICAL']])
        report["summary_metrics"]["trading_opportunities"] = len([r for r in report["trading_recommendations"] if r.action in ['BUY', 'SELL']])
        report["summary_metrics"]["risk_alerts"] = len([i for i in all_insights if i.category == 'RISK'])
        report["summary_metrics"]["performance_score"] = np.mean([i.impact_score for i in all_insights]) if all_insights else 0.0
        
        print(f"✅ Generated report with {len(all_insights)} insights and {len(report['trading_recommendations'])} recommendations")
        
        return report


def main():
    """Demo the advanced analytics engine"""
    print("🧠 ADVANCED ANALYTICS ENGINE DEMO")
    print("=" * 60)
    
    # Initialize engine
    engine = AdvancedAnalyticsEngine()
    
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
        }
    }
    
    # Generate demo trading data
    demo_trading = []
    base_value = 100000.0
    for i in range(30):
        value = base_value * (1 + np.random.normal(0.001, 0.02))
        demo_trading.append({
            'timestamp': (datetime.now() - timedelta(days=30-i)).isoformat(),
            'total_value': value,
            'daily_change': value - base_value,
            'daily_change_pct': (value - base_value) / base_value * 100
        })
        base_value = value
    
    demo_market = {
        'sentiment': {'BTC': 0.7, 'ETH': 0.6, 'SUI': 0.8},
        'volatility': {'BTC': 0.6, 'ETH': 0.7, 'SUI': 0.9}
    }
    
    # Generate comprehensive report
    report = engine.generate_comprehensive_report(demo_portfolio, demo_trading, demo_market)
    
    # Display results
    print("\n📈 PORTFOLIO INSIGHTS:")
    print("=" * 40)
    for insight in report['portfolio_insights'][:3]:
        print(f"🔍 {insight.title}")
        print(f"   Priority: {insight.priority}")
        print(f"   Impact: {insight.impact_score:.1f}/10")
        print(f"   Recommendation: {insight.recommendation}")
        print()
    
    print("🎯 TRADING RECOMMENDATIONS:")
    print("=" * 40)
    for rec in report['trading_recommendations'][:3]:
        print(f"💡 {rec.asset}: {rec.action}")
        print(f"   Confidence: {rec.confidence:.2%}")
        print(f"   Risk/Reward: {rec.risk_reward_ratio:.2f}")
        print(f"   Reasoning: {rec.reasoning}")
        print()
    
    print("🌟 MARKET OPPORTUNITIES:")
    print("=" * 40)
    for opp in report['market_opportunities'][:2]:
        print(f"🚀 {opp.asset}: {opp.opportunity_type}")
        print(f"   Confidence: {opp.confidence:.2%}")
        print(f"   Potential Return: {opp.potential_return:.2%}")
        print(f"   Description: {opp.description}")
        print()
    
    print("📊 SUMMARY METRICS:")
    print("=" * 40)
    for key, value in report['summary_metrics'].items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print("\n🎉 Advanced analytics demo completed!")


if __name__ == "__main__":
    main() 