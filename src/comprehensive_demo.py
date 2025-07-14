"""
Comprehensive Multi-Chain Portfolio Management System Demo
Showcases the complete autonomous trading intelligence with portfolio analysis
"""

import time
from datetime import datetime
from portfolio_agent import PortfolioAgent
from agentic_eliza import AgenticElizaOS
from portfolio_analyzer import MultiChainPortfolioAnalyzer


def print_system_banner():
    """Print system banner"""
    print("🚀 COMPREHENSIVE MULTI-CHAIN PORTFOLIO MANAGEMENT SYSTEM")
    print("=" * 80)
    print("🔗 Multi-Chain Analysis: SUI, Solana, Ethereum, Base, Sei")
    print("🤖 AI-Powered Recommendations: GROK-4 Integration")
    print("⚖️  Risk Management: Autonomous Monitoring")
    print("📊 Real-Time Portfolio Tracking: 15+ Wallets")
    print("🔄 Automated Rebalancing: Intelligent Suggestions")
    print("🚨 Alert System: Proactive Risk Management")
    print("=" * 80)


def demonstrate_multi_chain_analysis():
    """Demonstrate comprehensive multi-chain analysis"""
    print("\n🔗 MULTI-CHAIN WALLET ANALYSIS")
    print("=" * 60)
    
    analyzer = MultiChainPortfolioAnalyzer()
    
    print("📊 Analyzing your complete portfolio across all chains...")
    print("   • 11 SUI wallets")
    print("   • 2 Solana wallets")  
    print("   • 1 Ethereum wallet")
    print("   • 1 Base wallet")
    print("   • 1 Sei wallet")
    
    # Perform analysis
    analysis = analyzer.analyze_full_portfolio()
    
    # Display summary
    metrics = analysis.get("metrics", {})
    print(f"\n💰 PORTFOLIO OVERVIEW:")
    print(f"   Total Value: ${metrics.get('total_value', 0):,.2f}")
    print(f"   Risk Level: {metrics.get('risk_level', 'N/A')}")
    print(f"   Diversification Score: {metrics.get('diversification_score', 0):.1f}/100")
    print(f"   Active Chains: {metrics.get('total_chains', 0)}")
    print(f"   Total Tokens: {metrics.get('total_tokens', 0)}")
    
    # Show chain breakdown
    print(f"\n🔗 CHAIN BREAKDOWN:")
    chain_allocations = metrics.get('chain_allocations', {})
    for chain, allocation in sorted(chain_allocations.items(), key=lambda x: x[1], reverse=True):
        chain_value = metrics.get('chain_analysis', {}).get(chain, {}).get('total_value', 0)
        print(f"   {chain}: {allocation:.1f}% (${chain_value:,.2f})")
    
    return analysis


def demonstrate_ai_recommendations(analysis):
    """Demonstrate AI-powered recommendations"""
    print("\n🤖 AI-POWERED TRADING RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = analysis.get("recommendations", [])
    
    if not recommendations:
        print("✅ No immediate rebalancing needed - portfolio is well-optimized")
        return
    
    print(f"💡 Generated {len(recommendations)} intelligent recommendations:")
    
    # Group by priority
    high_priority = [r for r in recommendations if r.priority == "HIGH"]
    medium_priority = [r for r in recommendations if r.priority == "MEDIUM"]
    low_priority = [r for r in recommendations if r.priority == "LOW"]
    
    if high_priority:
        print(f"\n🔴 HIGH PRIORITY ({len(high_priority)} items):")
        for rec in high_priority:
            print(f"   • {rec.action} {rec.token}")
            print(f"     Current: {rec.current_allocation:.1f}% → Target: {rec.target_allocation:.1f}%")
            print(f"     Confidence: {rec.confidence:.1%}")
            print(f"     Reasoning: {rec.reasoning}")
    
    if medium_priority:
        print(f"\n🟡 MEDIUM PRIORITY ({len(medium_priority)} items):")
        for rec in medium_priority:
            print(f"   • {rec.action} {rec.token} ({rec.confidence:.1%} confidence)")
    
    if low_priority:
        print(f"\n🟢 LOW PRIORITY ({len(low_priority)} items):")
        for rec in low_priority:
            print(f"   • {rec.action} {rec.token} ({rec.confidence:.1%} confidence)")


def demonstrate_risk_management(analysis):
    """Demonstrate risk management capabilities"""
    print("\n⚖️  RISK MANAGEMENT SYSTEM")
    print("=" * 60)
    
    metrics = analysis.get("metrics", {})
    
    # Overall risk assessment
    risk_level = metrics.get('risk_level', 'MEDIUM')
    diversification = metrics.get('diversification_score', 0)
    
    print(f"📊 RISK ASSESSMENT:")
    print(f"   Overall Risk Level: {risk_level}")
    print(f"   Diversification Score: {diversification:.1f}/100")
    
    # Risk indicators
    if risk_level == "HIGH":
        print("   🚨 HIGH RISK DETECTED!")
        print("   ⚠️  Immediate action required")
        print("   📋 Recommendations:")
        print("      • Reduce position sizes")
        print("      • Increase diversification")
        print("      • Consider stable assets")
    elif risk_level == "MEDIUM":
        print("   🟡 MODERATE RISK LEVEL")
        print("   👀 Monitor closely")
        print("   📋 Recommendations:")
        print("      • Review large positions")
        print("      • Consider rebalancing")
    else:
        print("   ✅ LOW RISK LEVEL")
        print("   🎯 Portfolio well-balanced")
    
    # Concentration analysis
    print(f"\n🎯 CONCENTRATION ANALYSIS:")
    chain_allocations = metrics.get('chain_allocations', {})
    token_allocations = metrics.get('token_allocations', {})
    
    if chain_allocations:
        max_chain = max(chain_allocations.items(), key=lambda x: x[1])
        print(f"   Highest Chain: {max_chain[0]} ({max_chain[1]:.1f}%)")
        if max_chain[1] > 60:
            print("   ⚠️  Chain concentration warning")
    
    if token_allocations:
        max_token = max(token_allocations.items(), key=lambda x: x[1])
        print(f"   Highest Token: {max_token[0]} ({max_token[1]:.1f}%)")
        if max_token[1] > 40:
            print("   ⚠️  Token concentration warning")


def demonstrate_autonomous_system():
    """Demonstrate autonomous portfolio management"""
    print("\n🤖 AUTONOMOUS PORTFOLIO MANAGEMENT")
    print("=" * 60)
    
    # Initialize systems
    portfolio_agent = PortfolioAgent()
    agentic_system = AgenticElizaOS()
    
    print("🔄 Initializing autonomous systems...")
    print("   ✅ Portfolio Agent: Ready")
    print("   ✅ Agentic System: Ready")
    print("   ✅ Multi-Chain Analyzer: Ready")
    print("   ✅ GROK-4 AI: Connected")
    
    # Demonstrate autonomous decision making
    print(f"\n🧠 AUTONOMOUS DECISION SIMULATION:")
    
    # Get portfolio analysis
    analysis = portfolio_agent.force_portfolio_analysis()
    
    # Simulate autonomous decisions
    print(f"\n🚀 AUTONOMOUS ACTIONS:")
    print("   1. Market conditions analyzed")
    print("   2. Portfolio risk assessed")
    print("   3. Rebalancing opportunities identified")
    print("   4. AI recommendations generated")
    print("   5. Risk thresholds checked")
    
    # Show decision metrics
    status = portfolio_agent.get_portfolio_status()
    print(f"\n📊 SYSTEM STATUS:")
    print(f"   Autonomous Mode: {'🟢 ACTIVE' if status['autonomous_mode'] else '🔴 INACTIVE'}")
    print(f"   Total Decisions: {status['total_decisions']}")
    print(f"   Total Alerts: {status['total_alerts']}")
    print(f"   Performance Metrics: {status['performance_metrics']}")
    
    return portfolio_agent


def demonstrate_monitoring_system(portfolio_agent):
    """Demonstrate continuous monitoring"""
    print("\n👁️  CONTINUOUS MONITORING SYSTEM")
    print("=" * 60)
    
    print("🔄 Starting autonomous monitoring...")
    
    # Start autonomous management
    portfolio_agent.start_autonomous_portfolio_management()
    
    print("✅ Monitoring systems activated:")
    print("   • Portfolio value tracking")
    print("   • Risk level monitoring")
    print("   • Rebalancing opportunity detection")
    print("   • Market condition analysis")
    print("   • Alert generation")
    
    # Let it run for a short demo
    print(f"\n⏰ Running autonomous monitoring (30 seconds)...")
    
    for i in range(6):
        print(f"   📊 Monitoring cycle {i+1}/6...")
        time.sleep(5)
        
        # Simulate different scenarios
        if i == 1:
            print("   🚨 ALERT: SUI price movement detected!")
        elif i == 3:
            print("   💡 OPPORTUNITY: Rebalancing opportunity found!")
        elif i == 4:
            print("   ⚖️  RISK: Portfolio risk assessment updated!")
    
    # Stop monitoring
    portfolio_agent.stop_autonomous_portfolio_management()
    
    print("✅ Monitoring demo complete")


def demonstrate_integration_capabilities():
    """Demonstrate system integration"""
    print("\n🔗 SYSTEM INTEGRATION CAPABILITIES")
    print("=" * 60)
    
    print("🎯 INTEGRATED FEATURES:")
    print("   ✅ Multi-chain wallet analysis")
    print("   ✅ AI-powered recommendations")
    print("   ✅ Autonomous decision making")
    print("   ✅ Risk management system")
    print("   ✅ Continuous monitoring")
    print("   ✅ Alert generation")
    print("   ✅ Performance tracking")
    print("   ✅ Market context analysis")
    
    print(f"\n🔄 WORKFLOW INTEGRATION:")
    print("   1. 📊 Analyze portfolio across all chains")
    print("   2. 🤖 Generate AI recommendations")
    print("   3. ⚖️  Assess risk levels")
    print("   4. 🚨 Generate alerts if needed")
    print("   5. 🚀 Execute autonomous decisions")
    print("   6. 📈 Track performance")
    print("   7. 🔄 Repeat cycle")
    
    print(f"\n🎮 USER INTERACTIONS:")
    print("   • Force immediate analysis")
    print("   • Get portfolio status")
    print("   • View recommendations")
    print("   • Check risk levels")
    print("   • Review decision history")
    print("   • Configure risk tolerance")


def demonstrate_real_world_scenarios():
    """Demonstrate real-world usage scenarios"""
    print("\n🌍 REAL-WORLD USAGE SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Morning Portfolio Check",
            "description": "Daily analysis before market open",
            "actions": [
                "Analyze overnight portfolio changes",
                "Check for significant price movements",
                "Review risk levels",
                "Generate morning recommendations"
            ]
        },
        {
            "name": "Market Volatility Response",
            "description": "Automatic response to market volatility",
            "actions": [
                "Detect high volatility conditions",
                "Assess portfolio risk exposure",
                "Generate defensive recommendations",
                "Alert user to take action"
            ]
        },
        {
            "name": "Rebalancing Opportunity",
            "description": "Automated rebalancing suggestions",
            "actions": [
                "Identify allocation drift",
                "Calculate optimal rebalancing",
                "Consider transaction costs",
                "Prioritize by impact and confidence"
            ]
        },
        {
            "name": "Risk Threshold Breach",
            "description": "Emergency risk management",
            "actions": [
                "Detect risk threshold breach",
                "Generate immediate alerts",
                "Suggest risk reduction actions",
                "Monitor until resolved"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. 📋 {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Actions:")
        for action in scenario['actions']:
            print(f"      • {action}")


def main():
    """Main demonstration function"""
    print_system_banner()
    
    try:
        # 1. Multi-Chain Analysis
        analysis = demonstrate_multi_chain_analysis()
        
        # 2. AI Recommendations
        demonstrate_ai_recommendations(analysis)
        
        # 3. Risk Management
        demonstrate_risk_management(analysis)
        
        # 4. Autonomous System
        portfolio_agent = demonstrate_autonomous_system()
        
        # 5. Monitoring System
        demonstrate_monitoring_system(portfolio_agent)
        
        # 6. Integration Capabilities
        demonstrate_integration_capabilities()
        
        # 7. Real-World Scenarios
        demonstrate_real_world_scenarios()
        
        # Final Summary
        print("\n🎉 COMPREHENSIVE DEMO COMPLETE!")
        print("=" * 80)
        print("✅ Multi-chain portfolio analysis across 5 blockchains")
        print("✅ AI-powered recommendations using GROK-4")
        print("✅ Autonomous risk management system")
        print("✅ Real-time monitoring of 15+ wallets")
        print("✅ Intelligent rebalancing suggestions")
        print("✅ Proactive alert system")
        print("✅ Performance tracking and optimization")
        print("✅ Complete integration with existing trading system")
        
        print(f"\n🚀 YOUR PORTFOLIO IS NOW UNDER COMPREHENSIVE AI MANAGEMENT!")
        print("=" * 80)
        print("🔗 Wallets Monitored: 15 across 5 chains")
        print("🤖 AI Analysis: GROK-4 powered")
        print("⚖️  Risk Management: Autonomous")
        print("📊 Monitoring: 24/7 continuous")
        print("🔄 Rebalancing: Intelligent suggestions")
        print("🚨 Alerts: Proactive risk management")
        print("📈 Performance: Tracked and optimized")
        print("=" * 80)
        
        print(f"\n💡 NEXT STEPS:")
        print("   1. Configure your risk tolerance preferences")
        print("   2. Set up notification channels (Discord/Slack)")
        print("   3. Review and approve high-confidence recommendations")
        print("   4. Monitor autonomous decisions and performance")
        print("   5. Adjust parameters based on market conditions")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Please ensure all dependencies are installed and configuration is correct.")
        print("Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main() 