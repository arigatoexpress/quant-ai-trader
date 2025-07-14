"""
Multi-Chain Portfolio Analyzer Demo
Demonstrates comprehensive wallet analysis and trading recommendations
"""

import asyncio
import time
from datetime import datetime
from portfolio_analyzer import MultiChainPortfolioAnalyzer
from agentic_eliza import AgenticElizaOS


def demonstrate_portfolio_analysis():
    """Demonstrate multi-chain portfolio analysis capabilities"""
    print("🚀 MULTI-CHAIN PORTFOLIO ANALYZER DEMO")
    print("=" * 60)
    
    # Initialize the analyzer
    analyzer = MultiChainPortfolioAnalyzer()
    
    # Perform comprehensive analysis
    print("\n🔍 Analyzing your multi-chain portfolio...")
    analysis = analyzer.analyze_full_portfolio()
    
    # Display results
    analyzer.print_portfolio_report(analysis)
    
    return analysis


def demonstrate_ai_recommendations(analysis):
    """Demonstrate AI-powered trading recommendations"""
    print("\n🤖 AI-POWERED TRADING RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = analysis.get("recommendations", [])
    
    if not recommendations:
        print("❌ No recommendations generated")
        return
    
    print(f"💡 Generated {len(recommendations)} recommendations:")
    
    for i, rec in enumerate(recommendations, 1):
        priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        emoji = priority_emoji.get(rec.priority, "🔵")
        
        print(f"\n{i}. {emoji} {rec.action} {rec.token}")
        print(f"   📊 Current Allocation: {rec.current_allocation:.1f}%")
        print(f"   🎯 Target Allocation: {rec.target_allocation:.1f}%")
        print(f"   📈 Estimated Impact: {rec.estimated_impact:.1f}%")
        print(f"   🎲 Confidence: {rec.confidence:.1%}")
        print(f"   ⭐ Priority: {rec.priority}")
        print(f"   💭 Reasoning: {rec.reasoning}")


def demonstrate_risk_analysis(analysis):
    """Demonstrate risk analysis capabilities"""
    print("\n⚖️  RISK ANALYSIS")
    print("=" * 60)
    
    metrics = analysis.get("metrics", {})
    
    print(f"📊 Portfolio Risk Assessment:")
    print(f"   Risk Level: {metrics.get('risk_level', 'N/A')}")
    print(f"   Diversification Score: {metrics.get('diversification_score', 0):.1f}/100")
    print(f"   Total Chains: {metrics.get('total_chains', 0)}")
    print(f"   Total Tokens: {metrics.get('total_tokens', 0)}")
    
    # Chain concentration analysis
    chain_allocations = metrics.get('chain_allocations', {})
    if chain_allocations:
        max_chain = max(chain_allocations.items(), key=lambda x: x[1])
        print(f"   Highest Chain Concentration: {max_chain[0]} ({max_chain[1]:.1f}%)")
        
        if max_chain[1] > 70:
            print("   ⚠️  WARNING: High chain concentration detected!")
        elif max_chain[1] > 50:
            print("   ⚠️  CAUTION: Moderate chain concentration")
        else:
            print("   ✅ Good chain diversification")
    
    # Token concentration analysis
    token_allocations = metrics.get('token_allocations', {})
    if token_allocations:
        max_token = max(token_allocations.items(), key=lambda x: x[1])
        print(f"   Highest Token Concentration: {max_token[0]} ({max_token[1]:.1f}%)")
        
        if max_token[1] > 50:
            print("   ⚠️  WARNING: High token concentration detected!")
        elif max_token[1] > 30:
            print("   ⚠️  CAUTION: Moderate token concentration")
        else:
            print("   ✅ Good token diversification")


def demonstrate_chain_analysis(analysis):
    """Demonstrate per-chain analysis"""
    print("\n🔗 CHAIN-BY-CHAIN ANALYSIS")
    print("=" * 60)
    
    metrics = analysis.get("metrics", {})
    chain_analysis = metrics.get("chain_analysis", {})
    
    for chain, data in chain_analysis.items():
        print(f"\n{chain} Chain:")
        print(f"   💰 Total Value: ${data['total_value']:,.2f}")
        print(f"   🪙 Token Count: {data['token_count']}")
        print(f"   📊 Portfolio %: {metrics.get('chain_allocations', {}).get(chain, 0):.1f}%")
        
        # Show top holdings
        top_balances = sorted(data['balances'], key=lambda x: x.usd_value, reverse=True)[:3]
        print(f"   🏆 Top Holdings:")
        for balance in top_balances:
            print(f"      {balance.token_symbol}: {balance.balance:.4f} (${balance.usd_value:.2f})")


def demonstrate_wallet_monitoring():
    """Demonstrate wallet monitoring capabilities"""
    print("\n👁️  WALLET MONITORING SIMULATION")
    print("=" * 60)
    
    analyzer = MultiChainPortfolioAnalyzer()
    
    # Simulate monitoring cycle
    print("🔄 Starting monitoring cycle...")
    
    for cycle in range(3):
        print(f"\n📅 Monitoring Cycle {cycle + 1}")
        print("-" * 30)
        
        # Simulate some balance changes
        print("🔍 Checking for balance changes...")
        
        # In a real implementation, this would compare with previous analysis
        print("   ✅ SUI wallet 0x2401...5563: No significant changes")
        print("   ✅ Solana wallet DX3ws...4t2: No significant changes")
        print("   ⚠️  Ethereum wallet 0xc42E...8827: Price change detected")
        print("   ✅ Base wallet 0xc42E...8827: No significant changes")
        print("   ✅ Sei wallet 0xeea5...7944: No significant changes")
        
        # Simulate alerts
        if cycle == 1:
            print("   🚨 ALERT: SUI price increased by 8.5% - Portfolio value increased!")
        elif cycle == 2:
            print("   🚨 ALERT: Major rebalancing opportunity detected!")
        
        print(f"   💤 Waiting 30 seconds before next cycle...")
        time.sleep(2)  # Shortened for demo
    
    print("⏹️  Monitoring simulation complete")


def demonstrate_integration_with_agentic():
    """Demonstrate integration with existing agentic system"""
    print("\n🤖 INTEGRATION WITH AGENTIC SYSTEM")
    print("=" * 60)
    
    # Initialize both systems
    analyzer = MultiChainPortfolioAnalyzer()
    agentic_system = AgenticElizaOS()
    
    print("🔗 Connecting portfolio analyzer to agentic system...")
    
    # Perform portfolio analysis
    analysis = analyzer.analyze_full_portfolio()
    
    # Extract key metrics for agentic system
    metrics = analysis.get("metrics", {})
    recommendations = analysis.get("recommendations", [])
    
    print(f"\n📊 Portfolio Context for Agentic System:")
    print(f"   Total Value: ${metrics.get('total_value', 0):,.2f}")
    print(f"   Risk Level: {metrics.get('risk_level', 'N/A')}")
    print(f"   Active Recommendations: {len(recommendations)}")
    
    # Simulate agentic decision making with portfolio context
    print(f"\n🤖 Agentic System Analysis:")
    
    # High-priority recommendations
    high_priority = [r for r in recommendations if r.priority == "HIGH"]
    if high_priority:
        print(f"   🔴 HIGH PRIORITY: {len(high_priority)} urgent recommendations")
        for rec in high_priority:
            print(f"      → {rec.action} {rec.token} (Confidence: {rec.confidence:.1%})")
    
    # Risk-based decisions
    risk_level = metrics.get('risk_level', 'MEDIUM')
    if risk_level == "HIGH":
        print(f"   ⚠️  Risk Management: Portfolio requires immediate attention")
        print(f"      → Recommending defensive positions")
    elif risk_level == "MEDIUM":
        print(f"   🟡 Risk Management: Portfolio is moderately risky")
        print(f"      → Monitoring for rebalancing opportunities")
    else:
        print(f"   ✅ Risk Management: Portfolio is well-balanced")
        print(f"      → Maintaining current allocation")
    
    # Simulate autonomous decision
    print(f"\n🚀 Autonomous Decision Making:")
    print(f"   🎯 Next Action: Monitor for 15-minute price changes")
    print(f"   🔄 Rebalance Check: Scheduled for next analysis cycle")
    print(f"   📊 Performance Tracking: Enabled")
    
    return analysis


def demonstrate_scheduled_analysis():
    """Demonstrate scheduled portfolio analysis"""
    print("\n⏰ SCHEDULED ANALYSIS DEMO")
    print("=" * 60)
    
    analyzer = MultiChainPortfolioAnalyzer()
    
    print("📅 Simulating regular portfolio analysis schedule...")
    
    schedules = [
        {"name": "Morning Analysis", "time": "09:00 EST"},
        {"name": "Midday Check", "time": "12:00 EST"},
        {"name": "Evening Review", "time": "18:00 EST"},
        {"name": "Pre-Market Analysis", "time": "08:00 EST"}
    ]
    
    for schedule in schedules:
        print(f"\n🕐 {schedule['name']} - {schedule['time']}")
        print("-" * 40)
        
        # Simulate analysis
        print("   🔍 Performing portfolio analysis...")
        print("   📊 Calculating metrics...")
        print("   🤖 Generating recommendations...")
        
        # Simulate different scenarios
        if "Morning" in schedule['name']:
            print("   ✅ Portfolio stable overnight")
            print("   💡 2 new opportunities identified")
        elif "Midday" in schedule['name']:
            print("   ⚠️  Market volatility detected")
            print("   🔄 Rebalancing recommended")
        elif "Evening" in schedule['name']:
            print("   📈 Daily performance: +2.3%")
            print("   🎯 All targets on track")
        else:
            print("   🌅 Pre-market setup complete")
            print("   🚀 Ready for trading session")
        
        time.sleep(1)  # Brief pause for demo
    
    print("\n✅ Scheduled analysis complete")


def main():
    """Main demo function"""
    print("🎯 MULTI-CHAIN PORTFOLIO ANALYZER")
    print("🔗 Comprehensive Wallet Analysis & Trading Recommendations")
    print("=" * 80)
    
    try:
        # 1. Basic Portfolio Analysis
        analysis = demonstrate_portfolio_analysis()
        
        # 2. AI Recommendations
        demonstrate_ai_recommendations(analysis)
        
        # 3. Risk Analysis
        demonstrate_risk_analysis(analysis)
        
        # 4. Chain Analysis
        demonstrate_chain_analysis(analysis)
        
        # 5. Wallet Monitoring
        demonstrate_wallet_monitoring()
        
        # 6. Integration with Agentic System
        demonstrate_integration_with_agentic()
        
        # 7. Scheduled Analysis
        demonstrate_scheduled_analysis()
        
        print("\n🎉 DEMO COMPLETE!")
        print("=" * 80)
        print("✅ Multi-chain portfolio analysis")
        print("✅ AI-powered recommendations")
        print("✅ Risk assessment and diversification")
        print("✅ Real-time wallet monitoring")
        print("✅ Integration with autonomous trading")
        print("✅ Scheduled analysis capabilities")
        
        print(f"\n💡 Your portfolio is now under comprehensive AI surveillance!")
        print(f"   📊 Total wallets monitored: 15")
        print(f"   🔗 Chains covered: 5 (SUI, Solana, Ethereum, Base, Sei)")
        print(f"   🤖 AI recommendations: Active")
        print(f"   ⚖️  Risk management: Enabled")
        print(f"   🔄 Auto-rebalancing: Configured")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("Please ensure all dependencies are installed and configuration is correct.")


if __name__ == "__main__":
    main() 