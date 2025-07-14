#!/usr/bin/env python3
"""
Interactive Demo of Enhanced Agentic ElizaOS
"""

import time
import asyncio
from agentic_eliza import AgenticElizaOS, TradingDecision, MarketAlert

def demonstrate_natural_language_interface(agentic_system):
    """Demonstrate natural language command processing"""
    print("\n🗣️  NATURAL LANGUAGE INTERFACE DEMO")
    print("=" * 50)
    
    commands = [
        "Show me the current system status",
        "What's the market sentiment for BTC?",
        "Generate a trading report",
        "Start monitoring the markets",
        "What are the top performing assets?",
        "Stop all operations",
        "Analyze the risk for SUI trades"
    ]
    
    for command in commands:
        print(f"\n👤 User: {command}")
        response = agentic_system.process_natural_command(command)
        print(f"🤖 Agent: {response}")
        time.sleep(1)

def demonstrate_autonomous_decision_making(agentic_system):
    """Demonstrate autonomous trading decisions"""
    print("\n🧠 AUTONOMOUS DECISION MAKING DEMO")
    print("=" * 50)
    
    # Simulate some trading signals
    mock_signals = [
        {'asset': 'BTC', 'signal_type': 'BUY', 'confidence': 0.75, 'price': 118700},
        {'asset': 'SUI', 'signal_type': 'SELL', 'confidence': 0.65, 'price': 3.31},
        {'asset': 'SOL', 'signal_type': 'HOLD', 'confidence': 0.45, 'price': 161.75}
    ]
    
    print("🔄 Processing trading signals autonomously...")
    
    for signal in mock_signals:
        print(f"\n📊 Analyzing signal: {signal['signal_type']} {signal['asset']} @ ${signal['price']}")
        
        # Create a mock decision (in real system, this would be AI-generated)
        decision = TradingDecision(
            asset=signal['asset'],
            action=signal['signal_type'],
            confidence=signal['confidence'],
            reasoning=f"Technical analysis suggests {signal['signal_type']} based on market momentum",
            risk_level='MEDIUM' if signal['confidence'] > 0.6 else 'HIGH',
            target_price=signal['price'] * 1.05 if signal['signal_type'] == 'BUY' else None,
            stop_loss=signal['price'] * 0.95 if signal['signal_type'] == 'BUY' else None
        )
        
        # Execute autonomous decision
        executed = agentic_system.autonomous_trader.execute_trade(decision)
        
        if executed:
            print(f"✅ Trade executed successfully")
        else:
            print(f"⏸️  Trade held due to low confidence or risk management")
        
        time.sleep(2)

def demonstrate_real_time_monitoring(agentic_system):
    """Demonstrate real-time market monitoring"""
    print("\n📡 REAL-TIME MONITORING DEMO")
    print("=" * 50)
    
    print("🔍 Starting market monitoring for 30 seconds...")
    
    # Start monitoring
    agentic_system.market_monitor.start_monitoring()
    
    # Simulate monitoring for a short period
    print("   ⏱️  Monitoring active... (watching for price spikes > 5%)")
    
    # Let it run for a bit
    time.sleep(5)
    
    # Generate a mock alert
    mock_alert = MarketAlert(
        alert_type='PRICE_SPIKE',
        asset='SUI',
        message="SUI price spiked +7.2% in the last hour - potential momentum trade",
        severity='HIGH'
    )
    
    print(f"🚨 MOCK ALERT: {mock_alert.message}")
    
    # Stop monitoring
    agentic_system.market_monitor.stop_monitoring()
    print("⏹️  Monitoring stopped")

def demonstrate_learning_adaptation(agentic_system):
    """Demonstrate learning and strategy adaptation"""
    print("\n🧠 LEARNING & ADAPTATION DEMO")
    print("=" * 50)
    
    # Create some mock trading history
    mock_decisions = [
        TradingDecision('BTC', 'BUY', 0.8, 'Strong bullish signal', 'LOW'),
        TradingDecision('SOL', 'SELL', 0.6, 'Overbought conditions', 'MEDIUM'), 
        TradingDecision('SUI', 'HOLD', 0.4, 'Unclear market direction', 'HIGH'),
        TradingDecision('SEI', 'BUY', 0.7, 'Oversold bounce expected', 'MEDIUM'),
        TradingDecision('BTC', 'SELL', 0.5, 'Profit taking', 'HIGH')
    ]
    
    # Add to decision history
    agentic_system.decision_history.extend(mock_decisions)
    
    print(f"📊 Analyzing {len(mock_decisions)} recent trading decisions...")
    
    # Analyze performance
    performance = agentic_system.learning_agent.analyze_performance(mock_decisions)
    
    print("\n📈 Performance Analysis:")
    print(f"   Total Trades: {performance['total_trades']}")
    print(f"   Average Confidence: {performance['average_confidence']:.2%}")
    print(f"   Strategies Used: {len(performance['strategies_used'])}")
    
    if performance['recommendations']:
        print("\n💡 Learning Recommendations:")
        for rec in performance['recommendations']:
            print(f"   • {rec}")
    
    # Demonstrate adaptation
    adaptations = agentic_system.learning_agent.adapt_strategy(performance)
    
    if adaptations:
        print("\n🔄 Strategy Adaptations:")
        for adaptation in adaptations:
            print(f"   • {adaptation}")
    else:
        print("\n✅ Current strategy performing well - no adaptations needed")

def demonstrate_automated_reporting(agentic_system):
    """Demonstrate automated reporting"""
    print("\n📊 AUTOMATED REPORTING DEMO")
    print("=" * 50)
    
    print("🤖 Generating automated market report...")
    
    # Trigger an automated report
    agentic_system.send_autonomous_report()
    
    print("\n📧 This report would be automatically sent via:")
    print("   • Email notifications")
    print("   • Slack/Discord webhooks") 
    print("   • SMS alerts for critical events")
    print("   • Dashboard updates")

def demonstrate_risk_management(agentic_system):
    """Demonstrate intelligent risk management"""
    print("\n⚖️  RISK MANAGEMENT DEMO")
    print("=" * 50)
    
    trader = agentic_system.autonomous_trader
    
    print(f"📊 Current Risk Parameters:")
    print(f"   Risk Tolerance: {trader.risk_tolerance:.1%}")
    print(f"   Max Position Size: {trader.max_position_size:.1%}")
    print(f"   Active Positions: {len(trader.positions)}")
    
    # Simulate risk assessment
    print(f"\n🔍 Risk Assessment for Sample Portfolio:")
    
    sample_portfolio = {
        'BTC': {'size': 0.05, 'entry': 118000, 'current': 118700},
        'SUI': {'size': 0.08, 'entry': 3.50, 'current': 3.31},
        'SOL': {'size': 0.03, 'entry': 160, 'current': 161.75}
    }
    
    total_risk = 0
    for asset, position in sample_portfolio.items():
        pnl = ((position['current'] - position['entry']) / position['entry']) * 100
        risk_value = position['size'] * abs(pnl) if pnl < 0 else 0
        total_risk += risk_value
        
        status = "🟢 PROFIT" if pnl > 0 else "🔴 LOSS" if pnl < -5 else "🟡 WATCH"
        print(f"   {asset}: {position['size']:.1%} size, {pnl:+.1f}% PnL {status}")
    
    print(f"\n📈 Portfolio Risk Assessment:")
    risk_level = "🟢 LOW" if total_risk < 0.02 else "🟡 MEDIUM" if total_risk < 0.05 else "🔴 HIGH"
    print(f"   Total Risk Exposure: {total_risk:.1%} {risk_level}")

def main():
    """Main demo function"""
    print("🚀 ENHANCED AGENTIC ELIZAOS DEMO")
    print("=" * 60)
    print("This demo showcases the autonomous capabilities of the enhanced trading system")
    print()
    
    # Initialize the agentic system
    print("🤖 Initializing Enhanced Agentic ElizaOS...")
    agentic_system = AgenticElizaOS()
    
    # Show system status
    status = agentic_system.get_agent_status()
    print(f"\n🔧 System Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    
    # Run demonstrations
    demonstrations = [
        ("Natural Language Interface", demonstrate_natural_language_interface),
        ("Autonomous Decision Making", demonstrate_autonomous_decision_making),
        ("Real-Time Monitoring", demonstrate_real_time_monitoring),
        ("Learning & Adaptation", demonstrate_learning_adaptation),
        ("Automated Reporting", demonstrate_automated_reporting),
        ("Risk Management", demonstrate_risk_management)
    ]
    
    for title, demo_func in demonstrations:
        print(f"\n🎯 DEMO: {title}")
        try:
            demo_func(agentic_system)
        except Exception as e:
            print(f"❌ Demo error: {e}")
        
        print("\n" + "-"*50)
        time.sleep(2)
    
    # Final autonomous mode demonstration
    print("\n🚀 AUTONOMOUS MODE CAPABILITIES")
    print("=" * 50)
    print("The system can run fully autonomously with:")
    print("✅ Continuous market monitoring (24/7)")
    print("✅ Autonomous trading decisions")
    print("✅ Risk management and position sizing")
    print("✅ Learning from performance")
    print("✅ Automated reporting and alerts")
    print("✅ Natural language command processing")
    print("✅ Integration with external APIs and services")
    
    print("\n🎯 To start autonomous mode:")
    print("   agentic_system.start_autonomous_mode()")
    print("\n⏹️  To stop autonomous mode:")
    print("   agentic_system.stop_autonomous_mode()")
    
    print(f"\n🎉 Demo completed! Your Enhanced Agentic ElizaOS is ready for autonomous trading!")
    
    return agentic_system

if __name__ == "__main__":
    demo_system = main() 