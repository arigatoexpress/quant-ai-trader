"""
Trading History Analysis Demo
Comprehensive demonstration of trading pattern analysis and personalized recommendations
"""

import time
from datetime import datetime, timedelta
from trading_history_analyzer import TradingHistoryAnalyzer, Trade
from portfolio_agent import PortfolioAgent
from portfolio_analyzer import MultiChainPortfolioAnalyzer


def print_analysis_banner():
    """Print trading analysis banner"""
    print("📊 COMPREHENSIVE TRADING HISTORY ANALYSIS")
    print("=" * 80)
    print("🧠 Trading Psychology Analysis")
    print("📈 Performance Metrics Evaluation")
    print("🎯 Personalized Strategy Recommendations")
    print("🔗 Integration with Portfolio Management")
    print("=" * 80)


def demonstrate_trading_data_import():
    """Demonstrate trading data import capabilities"""
    print("\n📥 TRADING DATA IMPORT CAPABILITIES")
    print("=" * 60)
    
    analyzer = TradingHistoryAnalyzer()
    
    print("🔄 Supported Data Sources:")
    print("   • CSV files from exchanges")
    print("   • Exchange API integration")
    print("   • Manual trade entry")
    print("   • Demo data generation")
    
    print("\n📊 Generating comprehensive demo trading history...")
    print("   • 200 trades over 6 months")
    print("   • 8 different cryptocurrencies")
    print("   • Multiple exchanges")
    print("   • Various trading patterns")
    
    success = analyzer.import_trading_data("demo")
    
    if success:
        print("✅ Demo trading data generated successfully")
        print(f"   📋 Total trades: {len(analyzer.trades)}")
        
        # Show sample trades
        print("\n🔍 Sample Trades:")
        for i, trade in enumerate(analyzer.trades[:3], 1):
            print(f"   {i}. {trade.timestamp.strftime('%Y-%m-%d %H:%M')} - "
                  f"{trade.side} {trade.quantity:.3f} {trade.symbol} @ ${trade.price:.2f}")
    
    return analyzer


def demonstrate_performance_analysis(analyzer):
    """Demonstrate comprehensive performance analysis"""
    print("\n📈 PERFORMANCE METRICS ANALYSIS")
    print("=" * 60)
    
    print("🔄 Analyzing trading performance...")
    performance = analyzer.analyze_performance_metrics()
    
    if performance:
        print("✅ Performance analysis complete!")
        
        print(f"\n💰 PROFITABILITY METRICS:")
        print(f"   Total Trades: {performance.total_trades}")
        print(f"   Win Rate: {performance.win_rate:.1f}%")
        print(f"   Total PnL: ${performance.total_pnl:,.2f}")
        print(f"   Profit Factor: {performance.profit_factor:.2f}")
        print(f"   Average Win: ${performance.avg_win:.2f}")
        print(f"   Average Loss: ${performance.avg_loss:.2f}")
        print(f"   Largest Win: ${performance.largest_win:.2f}")
        print(f"   Largest Loss: ${performance.largest_loss:.2f}")
        
        print(f"\n⚖️  RISK METRICS:")
        print(f"   Max Drawdown: {performance.max_drawdown:.1f}%")
        print(f"   Sharpe Ratio: {performance.sharpe_ratio:.2f}")
        print(f"   Risk-Reward Ratio: {performance.risk_reward_ratio:.2f}")
        print(f"   Calmar Ratio: {performance.calmar_ratio:.2f}")
        print(f"   Sortino Ratio: {performance.sortino_ratio:.2f}")
        
        print(f"\n⏰ TIMING METRICS:")
        print(f"   Average Holding Period: {performance.avg_holding_period}")
        
        # Performance assessment
        print(f"\n📊 PERFORMANCE ASSESSMENT:")
        if performance.win_rate > 60:
            print("   🟢 Win Rate: EXCELLENT (>60%)")
        elif performance.win_rate > 50:
            print("   🟡 Win Rate: GOOD (50-60%)")
        else:
            print("   🔴 Win Rate: NEEDS IMPROVEMENT (<50%)")
        
        if performance.profit_factor > 1.5:
            print("   🟢 Profit Factor: EXCELLENT (>1.5)")
        elif performance.profit_factor > 1.0:
            print("   🟡 Profit Factor: PROFITABLE (>1.0)")
        else:
            print("   🔴 Profit Factor: LOSING (<1.0)")
        
        if performance.max_drawdown < 10:
            print("   🟢 Drawdown: EXCELLENT (<10%)")
        elif performance.max_drawdown < 20:
            print("   🟡 Drawdown: ACCEPTABLE (10-20%)")
        else:
            print("   🔴 Drawdown: HIGH RISK (>20%)")
    
    return performance


def demonstrate_behavioral_analysis(analyzer):
    """Demonstrate trading psychology and behavioral analysis"""
    print("\n🧠 TRADING PSYCHOLOGY ANALYSIS")
    print("=" * 60)
    
    print("🔄 Analyzing trading behavior patterns...")
    behavior = analyzer.analyze_trading_behavior()
    
    if behavior:
        print("✅ Behavioral analysis complete!")
        
        print(f"\n🎯 CORE TRADING SKILLS:")
        skills = [
            ("Discipline", behavior.discipline_score),
            ("Risk Management", behavior.risk_management_score),
            ("Emotional Control", behavior.emotional_control_score),
            ("Patience", behavior.patience_score),
            ("Consistency", behavior.consistency_score)
        ]
        
        for skill, score in skills:
            if score > 80:
                status = "🟢 EXCELLENT"
            elif score > 60:
                status = "🟡 GOOD"
            elif score > 40:
                status = "🟠 FAIR"
            else:
                status = "🔴 POOR"
            
            print(f"   {skill}: {score:.1f}/100 {status}")
        
        print(f"\n⚠️  RISK FACTORS:")
        risk_factors = [
            ("Overtrading Tendency", behavior.overtrading_tendency),
            ("FOMO Susceptibility", behavior.fomo_susceptibility),
            ("Revenge Trading Risk", behavior.revenge_trading_risk)
        ]
        
        for factor, score in risk_factors:
            if score > 70:
                status = "🔴 HIGH RISK"
            elif score > 50:
                status = "🟠 MODERATE"
            elif score > 30:
                status = "🟡 LOW"
            else:
                status = "🟢 MINIMAL"
            
            print(f"   {factor}: {score:.1f}/100 {status}")
        
        print(f"\n💪 PRIMARY STRENGTHS:")
        if behavior.primary_strengths:
            for strength in behavior.primary_strengths:
                print(f"   ✅ {strength}")
        else:
            print("   📝 Areas for development identified")
        
        print(f"\n⚠️  PRIMARY WEAKNESSES:")
        if behavior.primary_weaknesses:
            for weakness in behavior.primary_weaknesses:
                print(f"   ❌ {weakness}")
        else:
            print("   🎉 No major weaknesses detected!")
        
        # Trading session analysis
        if analyzer.trading_sessions:
            print(f"\n📅 TRADING SESSION ANALYSIS:")
            total_sessions = len(analyzer.trading_sessions)
            profitable_sessions = sum(1 for s in analyzer.trading_sessions if s.pnl > 0)
            session_win_rate = profitable_sessions / total_sessions * 100
            
            print(f"   Total Sessions: {total_sessions}")
            print(f"   Session Win Rate: {session_win_rate:.1f}%")
            print(f"   Avg Trades/Session: {sum(s.trades_count for s in analyzer.trading_sessions) / total_sessions:.1f}")
            
            # Emotional analysis
            emotion_counts = {}
            for session in analyzer.trading_sessions:
                emotion_counts[session.emotions] = emotion_counts.get(session.emotions, 0) + 1
            
            print(f"\n😊 EMOTIONAL STATE DISTRIBUTION:")
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / total_sessions * 100
                print(f"   {emotion}: {count} sessions ({percentage:.1f}%)")
    
    return behavior


def demonstrate_ai_recommendations(analyzer):
    """Demonstrate AI-powered strategy recommendations"""
    print("\n🤖 AI-POWERED STRATEGY RECOMMENDATIONS")
    print("=" * 60)
    
    print("🔄 Generating personalized recommendations...")
    recommendations = analyzer.generate_ai_recommendations()
    
    if recommendations:
        print(f"✅ Generated {len(recommendations)} personalized recommendations!")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. 🎯 {rec.strategy_type.upper()} STRATEGY")
            print(f"   📊 Confidence: {rec.confidence:.1%}")
            print(f"   💭 Reasoning: {rec.reasoning}")
            
            print(f"   📈 Improvements:")
            for improvement in rec.improvements:
                print(f"      • {improvement}")
            
            print(f"   ⚖️  Risk Adjustments:")
            for adjustment in rec.risk_adjustments:
                print(f"      • {adjustment}")
            
            print(f"   🧠 Behavioral Modifications:")
            for modification in rec.behavioral_modifications:
                print(f"      • {modification}")
            
            print(f"   🎯 Expected Improvement: {rec.expected_improvement:.1f}%")
    else:
        print("❌ No recommendations generated - check analysis data")
    
    return recommendations


def demonstrate_trading_patterns(analyzer):
    """Demonstrate trading pattern analysis"""
    print("\n🔍 TRADING PATTERN ANALYSIS")
    print("=" * 60)
    
    if not analyzer.trades:
        print("❌ No trading data available")
        return
    
    print("📊 Analyzing trading patterns...")
    
    # Symbol analysis
    symbols = [t.symbol for t in analyzer.trades]
    symbol_counts = {}
    for symbol in symbols:
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    
    print(f"\n🪙 SYMBOL PREFERENCES:")
    for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(analyzer.trades) * 100
        print(f"   {symbol}: {count} trades ({percentage:.1f}%)")
    
    # Time analysis
    trade_hours = [t.timestamp.hour for t in analyzer.trades]
    hour_counts = {}
    for hour in trade_hours:
        hour_counts[hour] = hour_counts.get(hour, 0) + 1
    
    most_active_hour = max(hour_counts.items(), key=lambda x: x[1])
    print(f"\n🕐 TIME PATTERNS:")
    print(f"   Most Active Hour: {most_active_hour[0]:02d}:00 ({most_active_hour[1]} trades)")
    
    # Trade type analysis
    trade_types = [t.trade_type for t in analyzer.trades]
    type_counts = {}
    for trade_type in trade_types:
        type_counts[trade_type] = type_counts.get(trade_type, 0) + 1
    
    print(f"\n📋 TRADE TYPE DISTRIBUTION:")
    for trade_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(analyzer.trades) * 100
        print(f"   {trade_type}: {count} trades ({percentage:.1f}%)")
    
    # Size analysis
    trade_values = [t.value for t in analyzer.trades]
    print(f"\n💰 POSITION SIZE ANALYSIS:")
    print(f"   Average Trade Size: ${sum(trade_values) / len(trade_values):,.2f}")
    print(f"   Largest Trade: ${max(trade_values):,.2f}")
    print(f"   Smallest Trade: ${min(trade_values):,.2f}")
    
    # Exchange analysis
    exchanges = [t.exchange for t in analyzer.trades]
    exchange_counts = {}
    for exchange in exchanges:
        exchange_counts[exchange] = exchange_counts.get(exchange, 0) + 1
    
    print(f"\n🏢 EXCHANGE USAGE:")
    for exchange, count in sorted(exchange_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(analyzer.trades) * 100
        print(f"   {exchange}: {count} trades ({percentage:.1f}%)")


def demonstrate_integration_with_portfolio():
    """Demonstrate integration with portfolio management system"""
    print("\n🔗 INTEGRATION WITH PORTFOLIO MANAGEMENT")
    print("=" * 60)
    
    print("🔄 Initializing integrated systems...")
    
    # Initialize both systems
    trading_analyzer = TradingHistoryAnalyzer()
    portfolio_agent = PortfolioAgent()
    
    print("✅ Systems initialized:")
    print("   • Trading History Analyzer")
    print("   • Portfolio Management Agent")
    print("   • Multi-Chain Analyzer")
    
    # Generate trading data
    print("\n📊 Generating trading history...")
    trading_analyzer.import_trading_data("demo")
    
    # Analyze trading behavior
    print("🧠 Analyzing trading behavior...")
    behavior = trading_analyzer.analyze_trading_behavior()
    performance = trading_analyzer.analyze_performance_metrics()
    
    # Get portfolio analysis
    print("💰 Analyzing current portfolio...")
    portfolio_analysis = portfolio_agent.force_portfolio_analysis()
    
    print("\n🎯 INTEGRATED INSIGHTS:")
    
    if behavior and performance:
        print(f"📈 Trading Performance:")
        print(f"   Win Rate: {performance.win_rate:.1f}%")
        print(f"   Risk Management Score: {behavior.risk_management_score:.1f}/100")
        
        print(f"\n💼 Portfolio Status:")
        if portfolio_analysis:
            metrics = portfolio_analysis.get("metrics", {})
            print(f"   Total Value: ${metrics.get('total_value', 0):,.2f}")
            print(f"   Risk Level: {metrics.get('risk_level', 'N/A')}")
            print(f"   Diversification: {metrics.get('diversification_score', 0):.1f}/100")
        
        print(f"\n🤖 INTEGRATED RECOMMENDATIONS:")
        
        # Risk alignment
        portfolio_risk = portfolio_analysis.get("metrics", {}).get('risk_level', 'MEDIUM')
        trading_risk = behavior.risk_management_score
        
        if portfolio_risk == 'HIGH' and trading_risk < 60:
            print("   🔴 CRITICAL: High portfolio risk + poor trading risk management")
            print("      → Immediate risk reduction required")
            print("      → Consider reducing position sizes")
            print("      → Implement strict stop-losses")
        
        elif portfolio_risk == 'LOW' and trading_risk > 80:
            print("   🟢 OPTIMAL: Low portfolio risk + excellent risk management")
            print("      → Consider slightly increasing position sizes")
            print("      → Maintain current risk discipline")
        
        # Trading frequency recommendations
        if behavior.overtrading_tendency > 70:
            print("   ⚠️  OVERTRADING DETECTED:")
            print("      → Reduce trading frequency")
            print("      → Focus on higher-quality setups")
            print("      → Implement mandatory waiting periods")
        
        # FOMO protection
        if behavior.fomo_susceptibility > 70:
            print("   😰 FOMO RISK HIGH:")
            print("      → Implement systematic entry rules")
            print("      → Use limit orders only")
            print("      → Set up portfolio alerts instead of manual monitoring")
        
        # Patience recommendations
        if behavior.patience_score < 50:
            print("   ⏰ PATIENCE IMPROVEMENT:")
            print("      → Focus on longer-term holds")
            print("      → Set minimum holding periods")
            print("      → Use swing trading strategies")


def demonstrate_comprehensive_report(analyzer):
    """Demonstrate comprehensive report generation"""
    print("\n📋 COMPREHENSIVE TRADING ANALYSIS REPORT")
    print("=" * 60)
    
    print("🔄 Generating comprehensive report...")
    report = analyzer.generate_comprehensive_report()
    
    print("✅ Report generated!")
    print("\n" + "="*80)
    print(report)
    print("="*80)


def demonstrate_improvement_tracking():
    """Demonstrate how to track improvement over time"""
    print("\n📈 IMPROVEMENT TRACKING SYSTEM")
    print("=" * 60)
    
    print("🎯 TRACKING YOUR PROGRESS:")
    print("\n📊 Key Metrics to Monitor:")
    print("   • Win rate improvement")
    print("   • Risk management score")
    print("   • Emotional control score")
    print("   • Consistency score")
    print("   • Maximum drawdown reduction")
    
    print("\n📅 RECOMMENDED REVIEW SCHEDULE:")
    print("   • Weekly: Trading session review")
    print("   • Monthly: Comprehensive analysis")
    print("   • Quarterly: Strategy adjustments")
    print("   • Annually: Complete strategy overhaul")
    
    print("\n🎮 IMPROVEMENT STRATEGIES:")
    
    strategies = [
        {
            "area": "Discipline",
            "actions": [
                "Set daily trading rules and stick to them",
                "Use only limit orders for entries",
                "Implement position sizing rules",
                "Create trading checklists"
            ]
        },
        {
            "area": "Emotional Control",
            "actions": [
                "Keep a trading journal with emotions",
                "Practice meditation before trading",
                "Set maximum trades per day",
                "Take breaks after losses"
            ]
        },
        {
            "area": "Risk Management",
            "actions": [
                "Never risk more than 2% per trade",
                "Set stop-losses before entering",
                "Diversify across multiple assets",
                "Use position sizing calculators"
            ]
        },
        {
            "area": "Patience",
            "actions": [
                "Wait for high-probability setups",
                "Set minimum holding periods",
                "Focus on quality over quantity",
                "Use higher timeframes for analysis"
            ]
        }
    ]
    
    for strategy in strategies:
        print(f"\n💡 {strategy['area'].upper()} IMPROVEMENT:")
        for action in strategy['actions']:
            print(f"   • {action}")


def main():
    """Main demonstration function"""
    print_analysis_banner()
    
    try:
        # 1. Data Import Demo
        analyzer = demonstrate_trading_data_import()
        
        # 2. Performance Analysis
        performance = demonstrate_performance_analysis(analyzer)
        
        # 3. Behavioral Analysis
        behavior = demonstrate_behavioral_analysis(analyzer)
        
        # 4. Trading Patterns
        demonstrate_trading_patterns(analyzer)
        
        # 5. AI Recommendations
        recommendations = demonstrate_ai_recommendations(analyzer)
        
        # 6. Portfolio Integration
        demonstrate_integration_with_portfolio()
        
        # 7. Comprehensive Report
        demonstrate_comprehensive_report(analyzer)
        
        # 8. Improvement Tracking
        demonstrate_improvement_tracking()
        
        # Final Summary
        print("\n🎉 TRADING ANALYSIS DEMO COMPLETE!")
        print("=" * 80)
        print("✅ Trading history analysis")
        print("✅ Performance metrics evaluation")
        print("✅ Behavioral psychology assessment")
        print("✅ AI-powered recommendations")
        print("✅ Portfolio integration")
        print("✅ Comprehensive reporting")
        print("✅ Improvement tracking system")
        
        print(f"\n🚀 YOUR TRADING IS NOW UNDER COMPREHENSIVE AI ANALYSIS!")
        print("=" * 80)
        print("📊 Performance Metrics: Calculated")
        print("🧠 Trading Psychology: Analyzed")
        print("🎯 Personalized Strategies: Generated")
        print("🔗 Portfolio Integration: Active")
        print("📈 Improvement Tracking: Enabled")
        print("🤖 AI Recommendations: Ongoing")
        print("=" * 80)
        
        print(f"\n💡 NEXT STEPS FOR IMPROVEMENT:")
        if behavior and performance:
            if behavior.discipline_score < 60:
                print("   1. 🎯 Focus on improving trading discipline")
            if behavior.risk_management_score < 60:
                print("   2. ⚖️  Enhance risk management practices")
            if behavior.emotional_control_score < 60:
                print("   3. 🧠 Work on emotional control")
            if performance.win_rate < 50:
                print("   4. 📈 Improve trade selection process")
            if performance.max_drawdown > 20:
                print("   5. 🛡️  Reduce portfolio risk exposure")
        
        print("   6. 📋 Implement AI recommendations")
        print("   7. 🔄 Set up regular analysis schedule")
        print("   8. 📊 Track progress monthly")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Please ensure all dependencies are installed and configuration is correct.")


if __name__ == "__main__":
    main() 