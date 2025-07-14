"""
Complete Trading Intelligence System
Integrates portfolio management, trading history analysis, and AI recommendations
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yaml
from dataclasses import dataclass, asdict

from portfolio_agent import PortfolioAgent
from trading_history_analyzer import TradingHistoryAnalyzer
from agentic_eliza import AgenticElizaOS


@dataclass
class TradingIntelligence:
    """Comprehensive trading intelligence report"""
    timestamp: datetime
    portfolio_analysis: Dict[str, Any]
    trading_behavior: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    ai_recommendations: List[Dict[str, Any]]
    risk_assessment: Dict[str, str]
    improvement_plan: List[str]
    integrated_score: float  # Overall trading intelligence score


class CompleteTradingSystem:
    """
    Comprehensive trading intelligence system that combines:
    - Multi-chain portfolio management
    - Trading history analysis
    - AI-powered recommendations
    - Behavioral psychology assessment
    """
    
    def __init__(self, config_path=None):
        # Initialize all subsystems
        self.portfolio_agent = PortfolioAgent(config_path)
        self.trading_analyzer = TradingHistoryAnalyzer(config_path)
        self.agentic_system = AgenticElizaOS(config_path)
        
        # System state
        self.last_analysis: Optional[TradingIntelligence] = None
        self.analysis_history: List[TradingIntelligence] = []
        
        print("🎯 Complete Trading Intelligence System initialized")
        print("   ✅ Portfolio Management: Ready")
        print("   ✅ Trading History Analysis: Ready")
        print("   ✅ AI Recommendation Engine: Ready")
        print("   ✅ Behavioral Psychology: Ready")
    
    def import_trading_history(self, data_source: str, **kwargs) -> bool:
        """Import trading history for analysis"""
        print("📥 Importing trading history...")
        
        success = self.trading_analyzer.import_trading_data(data_source, **kwargs)
        
        if success:
            print(f"✅ Successfully imported {len(self.trading_analyzer.trades)} trades")
        else:
            print("❌ Failed to import trading data")
        
        return success
    
    def analyze_complete_trading_profile(self) -> TradingIntelligence:
        """Perform comprehensive trading profile analysis"""
        print("\n🔍 PERFORMING COMPREHENSIVE TRADING ANALYSIS")
        print("=" * 70)
        
        analysis_start = datetime.now()
        
        # 1. Portfolio Analysis
        print("💰 Analyzing multi-chain portfolio...")
        portfolio_analysis = self.portfolio_agent.force_portfolio_analysis()
        
        # 2. Trading Behavior Analysis
        print("🧠 Analyzing trading behavior and psychology...")
        trading_behavior = self.trading_analyzer.analyze_trading_behavior()
        performance_metrics = self.trading_analyzer.analyze_performance_metrics()
        
        # 3. AI Recommendations
        print("🤖 Generating AI-powered recommendations...")
        portfolio_recommendations = self.portfolio_agent.get_portfolio_recommendations()
        trading_recommendations = self.trading_analyzer.generate_ai_recommendations()
        
        # 4. Risk Assessment
        print("⚖️  Performing integrated risk assessment...")
        risk_assessment = self._assess_integrated_risk(
            portfolio_analysis, trading_behavior, performance_metrics
        )
        
        # 5. Improvement Plan
        print("📈 Creating personalized improvement plan...")
        improvement_plan = self._create_improvement_plan(
            trading_behavior, performance_metrics, portfolio_analysis
        )
        
        # 6. Calculate Overall Score
        integrated_score = self._calculate_integrated_score(
            portfolio_analysis, trading_behavior, performance_metrics
        )
        
        # Create comprehensive report
        intelligence = TradingIntelligence(
            timestamp=analysis_start,
            portfolio_analysis=portfolio_analysis or {},
            trading_behavior=asdict(trading_behavior) if trading_behavior else {},
            performance_metrics=asdict(performance_metrics) if performance_metrics else {},
            ai_recommendations=[
                asdict(rec) for rec in (trading_recommendations or [])
            ] + [
                asdict(rec) for rec in (portfolio_recommendations or [])
            ],
            risk_assessment=risk_assessment,
            improvement_plan=improvement_plan,
            integrated_score=integrated_score
        )
        
        self.last_analysis = intelligence
        self.analysis_history.append(intelligence)
        
        print(f"✅ Complete analysis finished in {datetime.now() - analysis_start}")
        return intelligence
    
    def _assess_integrated_risk(self, portfolio_analysis: Dict, trading_behavior, performance_metrics) -> Dict[str, str]:
        """Assess integrated risk across portfolio and trading behavior"""
        risk_assessment = {}
        
        # Portfolio risk
        if portfolio_analysis:
            portfolio_risk = portfolio_analysis.get("metrics", {}).get('risk_level', 'MEDIUM')
            risk_assessment['portfolio_risk'] = portfolio_risk
        else:
            risk_assessment['portfolio_risk'] = 'UNKNOWN'
        
        # Trading behavior risk
        if trading_behavior:
            if trading_behavior.risk_management_score < 50:
                risk_assessment['trading_risk'] = 'HIGH'
            elif trading_behavior.risk_management_score < 70:
                risk_assessment['trading_risk'] = 'MEDIUM'
            else:
                risk_assessment['trading_risk'] = 'LOW'
        else:
            risk_assessment['trading_risk'] = 'UNKNOWN'
        
        # Performance risk
        if performance_metrics:
            if performance_metrics.max_drawdown > 20:
                risk_assessment['performance_risk'] = 'HIGH'
            elif performance_metrics.max_drawdown > 10:
                risk_assessment['performance_risk'] = 'MEDIUM'
            else:
                risk_assessment['performance_risk'] = 'LOW'
        else:
            risk_assessment['performance_risk'] = 'UNKNOWN'
        
        # Behavioral risk factors
        if trading_behavior:
            behavioral_risks = []
            if trading_behavior.overtrading_tendency > 70:
                behavioral_risks.append('OVERTRADING')
            if trading_behavior.fomo_susceptibility > 70:
                behavioral_risks.append('FOMO')
            if trading_behavior.revenge_trading_risk > 70:
                behavioral_risks.append('REVENGE_TRADING')
            if trading_behavior.emotional_control_score < 50:
                behavioral_risks.append('EMOTIONAL_INSTABILITY')
            
            risk_assessment['behavioral_risks'] = behavioral_risks
        
        # Overall risk level
        risk_levels = [v for v in risk_assessment.values() if v in ['LOW', 'MEDIUM', 'HIGH']]
        if 'HIGH' in risk_levels:
            risk_assessment['overall_risk'] = 'HIGH'
        elif 'MEDIUM' in risk_levels:
            risk_assessment['overall_risk'] = 'MEDIUM'
        else:
            risk_assessment['overall_risk'] = 'LOW'
        
        return risk_assessment
    
    def _create_improvement_plan(self, trading_behavior, performance_metrics, portfolio_analysis) -> List[str]:
        """Create personalized improvement plan"""
        improvements = []
        
        # Portfolio improvements
        if portfolio_analysis:
            portfolio_risk = portfolio_analysis.get("metrics", {}).get('risk_level', 'MEDIUM')
            if portfolio_risk == 'HIGH':
                improvements.append("🔴 CRITICAL: Reduce portfolio risk through diversification")
            
            diversification = portfolio_analysis.get("metrics", {}).get('diversification_score', 0)
            if diversification < 50:
                improvements.append("🔗 Improve portfolio diversification across chains and tokens")
        
        # Trading behavior improvements
        if trading_behavior:
            if trading_behavior.discipline_score < 60:
                improvements.append("🎯 Improve trading discipline through rule-based systems")
            
            if trading_behavior.risk_management_score < 60:
                improvements.append("⚖️  Enhance risk management with stricter position sizing")
            
            if trading_behavior.emotional_control_score < 60:
                improvements.append("🧠 Work on emotional control through meditation and journaling")
            
            if trading_behavior.patience_score < 60:
                improvements.append("⏰ Develop patience through longer-term strategies")
            
            if trading_behavior.overtrading_tendency > 70:
                improvements.append("📉 Reduce trading frequency and focus on quality setups")
            
            if trading_behavior.fomo_susceptibility > 70:
                improvements.append("😰 Implement systematic entry rules to combat FOMO")
            
            if trading_behavior.revenge_trading_risk > 70:
                improvements.append("🛑 Create mandatory cooling-off periods after losses")
        
        # Performance improvements
        if performance_metrics:
            if performance_metrics.win_rate < 50:
                improvements.append("📈 Improve trade selection and market analysis skills")
            
            if performance_metrics.profit_factor < 1.2:
                improvements.append("💰 Focus on increasing profit factor through better RR ratios")
            
            if performance_metrics.max_drawdown > 15:
                improvements.append("🛡️  Implement stricter drawdown controls and position sizing")
        
        # If no specific improvements found
        if not improvements:
            improvements.append("🎉 Continue current excellent trading practices!")
            improvements.append("📊 Focus on consistency and gradual optimization")
        
        return improvements
    
    def _calculate_integrated_score(self, portfolio_analysis: Dict, trading_behavior, performance_metrics) -> float:
        """Calculate overall trading intelligence score (0-100)"""
        scores = []
        
        # Portfolio score (25% weight)
        if portfolio_analysis:
            diversification = portfolio_analysis.get("metrics", {}).get('diversification_score', 50)
            risk_level = portfolio_analysis.get("metrics", {}).get('risk_level', 'MEDIUM')
            
            risk_score = {'LOW': 90, 'MEDIUM': 70, 'HIGH': 30}.get(risk_level, 50)
            portfolio_score = (diversification + risk_score) / 2
            scores.append(('portfolio', portfolio_score, 0.25))
        
        # Trading behavior score (35% weight)
        if trading_behavior:
            behavior_score = (
                trading_behavior.discipline_score * 0.2 +
                trading_behavior.risk_management_score * 0.25 +
                trading_behavior.emotional_control_score * 0.2 +
                trading_behavior.patience_score * 0.15 +
                trading_behavior.consistency_score * 0.2
            )
            scores.append(('behavior', behavior_score, 0.35))
        
        # Performance score (40% weight)
        if performance_metrics:
            # Normalize performance metrics to 0-100 scale
            win_rate_score = min(100, performance_metrics.win_rate * 1.5)  # 67% = 100 points
            profit_factor_score = min(100, performance_metrics.profit_factor * 40)  # 2.5 = 100 points
            drawdown_score = max(0, 100 - performance_metrics.max_drawdown * 3)  # 0% = 100, 33% = 0
            
            performance_score = (win_rate_score + profit_factor_score + drawdown_score) / 3
            scores.append(('performance', performance_score, 0.40))
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weight for _, _, weight in scores)
            weighted_sum = sum(score * weight for _, score, weight in scores)
            integrated_score = weighted_sum / total_weight
        else:
            integrated_score = 50.0  # Neutral score if no data
        
        return min(100, max(0, integrated_score))
    
    def print_comprehensive_report(self, intelligence: Optional[TradingIntelligence] = None):
        """Print comprehensive trading intelligence report"""
        if intelligence is None:
            intelligence = self.last_analysis
        
        if not intelligence:
            print("❌ No analysis available. Run analyze_complete_trading_profile() first.")
            return
        
        print(f"\n📊 COMPREHENSIVE TRADING INTELLIGENCE REPORT")
        print("=" * 80)
        print(f"Generated: {intelligence.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Trading Intelligence Score: {intelligence.integrated_score:.1f}/100")
        
        # Score interpretation
        if intelligence.integrated_score >= 80:
            score_status = "🟢 EXCELLENT"
        elif intelligence.integrated_score >= 65:
            score_status = "🟡 GOOD"
        elif intelligence.integrated_score >= 50:
            score_status = "🟠 FAIR"
        else:
            score_status = "🔴 NEEDS IMPROVEMENT"
        
        print(f"Performance Level: {score_status}")
        
        # Portfolio Summary
        print(f"\n💰 PORTFOLIO SUMMARY")
        print("-" * 50)
        if intelligence.portfolio_analysis:
            metrics = intelligence.portfolio_analysis.get("metrics", {})
            print(f"Total Value: ${metrics.get('total_value', 0):,.2f}")
            print(f"Risk Level: {metrics.get('risk_level', 'N/A')}")
            print(f"Diversification Score: {metrics.get('diversification_score', 0):.1f}/100")
            print(f"Active Chains: {metrics.get('total_chains', 0)}")
            print(f"Total Tokens: {metrics.get('total_tokens', 0)}")
        else:
            print("No portfolio data available")
        
        # Trading Performance
        print(f"\n📈 TRADING PERFORMANCE")
        print("-" * 50)
        if intelligence.performance_metrics:
            perf = intelligence.performance_metrics
            print(f"Win Rate: {perf.get('win_rate', 0):.1f}%")
            print(f"Profit Factor: {perf.get('profit_factor', 0):.2f}")
            print(f"Max Drawdown: {perf.get('max_drawdown', 0):.1f}%")
            print(f"Total Trades: {perf.get('total_trades', 0)}")
            print(f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
        else:
            print("No trading performance data available")
        
        # Behavioral Analysis
        print(f"\n🧠 BEHAVIORAL ANALYSIS")
        print("-" * 50)
        if intelligence.trading_behavior:
            behavior = intelligence.trading_behavior
            print(f"Discipline Score: {behavior.get('discipline_score', 0):.1f}/100")
            print(f"Risk Management: {behavior.get('risk_management_score', 0):.1f}/100")
            print(f"Emotional Control: {behavior.get('emotional_control_score', 0):.1f}/100")
            print(f"Patience Score: {behavior.get('patience_score', 0):.1f}/100")
            print(f"Consistency: {behavior.get('consistency_score', 0):.1f}/100")
            
            print(f"\nRisk Factors:")
            print(f"Overtrading Tendency: {behavior.get('overtrading_tendency', 0):.1f}/100")
            print(f"FOMO Susceptibility: {behavior.get('fomo_susceptibility', 0):.1f}/100")
            print(f"Revenge Trading Risk: {behavior.get('revenge_trading_risk', 0):.1f}/100")
        else:
            print("No behavioral analysis data available")
        
        # Risk Assessment
        print(f"\n⚖️  RISK ASSESSMENT")
        print("-" * 50)
        for risk_type, level in intelligence.risk_assessment.items():
            if isinstance(level, list):
                if level:
                    print(f"{risk_type.replace('_', ' ').title()}: {', '.join(level)}")
            else:
                print(f"{risk_type.replace('_', ' ').title()}: {level}")
        
        # AI Recommendations
        print(f"\n🤖 AI RECOMMENDATIONS")
        print("-" * 50)
        if intelligence.ai_recommendations:
            for i, rec in enumerate(intelligence.ai_recommendations[:5], 1):
                print(f"{i}. {rec.get('strategy_type', 'General')}")
                print(f"   Confidence: {rec.get('confidence', 0):.1%}")
                print(f"   Reasoning: {rec.get('reasoning', 'N/A')}")
        else:
            print("No AI recommendations available")
        
        # Improvement Plan
        print(f"\n📈 PERSONALIZED IMPROVEMENT PLAN")
        print("-" * 50)
        for i, improvement in enumerate(intelligence.improvement_plan, 1):
            print(f"{i}. {improvement}")
        
        print("=" * 80)
    
    def start_continuous_monitoring(self):
        """Start continuous monitoring of both portfolio and trading behavior"""
        print("\n🔄 STARTING CONTINUOUS MONITORING")
        print("=" * 60)
        
        # Start portfolio monitoring
        self.portfolio_agent.start_autonomous_portfolio_management()
        
        print("✅ Continuous monitoring activated:")
        print("   • Portfolio monitoring: ACTIVE")
        print("   • Risk assessment: CONTINUOUS")
        print("   • AI recommendations: UPDATING")
        print("   • Behavioral tracking: ENABLED")
        print("\n💡 System will now provide real-time insights and alerts")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self.portfolio_agent.stop_autonomous_portfolio_management()
        print("⏹️  Continuous monitoring deactivated")
    
    def get_trading_intelligence_summary(self) -> Dict[str, Any]:
        """Get a summary of current trading intelligence"""
        if not self.last_analysis:
            return {"error": "No analysis available"}
        
        intelligence = self.last_analysis
        
        return {
            "overall_score": intelligence.integrated_score,
            "portfolio_value": intelligence.portfolio_analysis.get("metrics", {}).get('total_value', 0),
            "win_rate": intelligence.performance_metrics.get('win_rate', 0),
            "risk_level": intelligence.risk_assessment.get('overall_risk', 'UNKNOWN'),
            "top_recommendations": [rec.get('strategy_type', 'N/A') for rec in intelligence.ai_recommendations[:3]],
            "priority_improvements": intelligence.improvement_plan[:3],
            "last_updated": intelligence.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def compare_with_previous_analysis(self) -> Dict[str, Any]:
        """Compare current analysis with previous analysis to show progress"""
        if len(self.analysis_history) < 2:
            return {"message": "Need at least 2 analyses for comparison"}
        
        current = self.analysis_history[-1]
        previous = self.analysis_history[-2]
        
        comparison = {
            "time_period": f"{previous.timestamp.date()} to {current.timestamp.date()}",
            "score_change": current.integrated_score - previous.integrated_score,
            "improvements": [],
            "deteriorations": []
        }
        
        # Compare behavioral scores
        if current.trading_behavior and previous.trading_behavior:
            score_comparisons = [
                ("discipline", "Discipline"),
                ("risk_management_score", "Risk Management"),
                ("emotional_control_score", "Emotional Control"),
                ("patience_score", "Patience"),
                ("consistency_score", "Consistency")
            ]
            
            for key, name in score_comparisons:
                current_score = current.trading_behavior.get(key, 0)
                previous_score = previous.trading_behavior.get(key, 0)
                change = current_score - previous_score
                
                if change > 5:
                    comparison["improvements"].append(f"{name}: +{change:.1f} points")
                elif change < -5:
                    comparison["deteriorations"].append(f"{name}: {change:.1f} points")
        
        return comparison


def main():
    """Demo the complete trading system"""
    print("🎯 COMPLETE TRADING INTELLIGENCE SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize system
    system = CompleteTradingSystem()
    
    # Import demo trading data
    print("\n📥 Importing demo trading history...")
    system.import_trading_history("demo")
    
    # Perform comprehensive analysis
    print("\n🔍 Performing comprehensive analysis...")
    intelligence = system.analyze_complete_trading_profile()
    
    # Display results
    system.print_comprehensive_report(intelligence)
    
    # Show summary
    print("\n📋 TRADING INTELLIGENCE SUMMARY")
    print("=" * 60)
    summary = system.get_trading_intelligence_summary()
    
    for key, value in summary.items():
        if key != "error":
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Start monitoring
    print("\n🔄 Starting continuous monitoring for 30 seconds...")
    system.start_continuous_monitoring()
    
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    
    system.stop_continuous_monitoring()
    
    print("\n🎉 COMPLETE TRADING SYSTEM DEMO FINISHED!")
    print("✅ Your trading is now under comprehensive AI intelligence!")


if __name__ == "__main__":
    main() 