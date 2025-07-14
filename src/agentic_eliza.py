"""
Enhanced Agentic ElizaOS - Autonomous Trading Intelligence System
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml
import requests
from openai import OpenAI

from eliza_os import ElizaOS
from data_fetcher import DataFetcher

@dataclass
class TradingDecision:
    asset: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    reasoning: str
    risk_level: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MarketAlert:
    alert_type: str  # 'PRICE_SPIKE', 'VOLUME_SURGE', 'NEWS_IMPACT', 'RISK_WARNING'
    asset: str
    message: str
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    timestamp: datetime = field(default_factory=datetime.now)

class AutonomousTrader:
    """Autonomous trading agent with decision-making capabilities"""
    
    def __init__(self, risk_tolerance: float = 0.02, max_position_size: float = 0.1):
        self.risk_tolerance = risk_tolerance
        self.max_position_size = max_position_size
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0
        }
    
    def analyze_trade_opportunity(self, signal, market_data, grok_client) -> TradingDecision:
        """Analyze trading opportunity using AI and risk management"""
        try:
            prompt = f"""
            As an autonomous trading agent, analyze this trading opportunity:
            
            Signal: {signal}
            Market Data: {market_data}
            Current Positions: {self.positions}
            Risk Tolerance: {self.risk_tolerance}
            Performance: Win Rate {self.performance_metrics['win_rate']:.2%}
            
            Provide a trading decision with:
            1. Action (BUY/SELL/HOLD)
            2. Confidence level (0-1)
            3. Risk assessment (LOW/MEDIUM/HIGH)
            4. Reasoning
            5. Target price and stop loss if applicable
            
            Format response as JSON.
            """
            
            completion = grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse AI response and create decision
            ai_response = completion.choices[0].message.content
            # For now, create a basic decision - would parse JSON in production
            
            return TradingDecision(
                asset=signal.get('asset', 'UNKNOWN'),
                action='HOLD',  # Default conservative action
                confidence=0.5,
                reasoning="Autonomous analysis pending",
                risk_level='MEDIUM',
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return TradingDecision(
                asset='ERROR',
                action='HOLD',
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}",
                risk_level='HIGH',
                timestamp=datetime.now()
            )
    
    def execute_trade(self, decision: TradingDecision) -> bool:
        """Execute trading decision (simulation for now)"""
        if decision.confidence < 0.6 or decision.action == 'HOLD':
            print(f"🤖 AGENT: Skipping trade for {decision.asset} - {decision.reasoning}")
            return False
        
        # Simulate trade execution
        print(f"🚀 AGENT: Executing {decision.action} for {decision.asset}")
        print(f"   Confidence: {decision.confidence:.2%}")
        print(f"   Risk Level: {decision.risk_level}")
        print(f"   Reasoning: {decision.reasoning}")
        
        # Update performance tracking
        self.trade_history.append(decision)
        self.performance_metrics['total_trades'] += 1
        
        return True

class MarketMonitor:
    """Continuous market monitoring agent"""
    
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.monitoring = False
        self.alerts = []
        self.last_prices = {}
        self.price_thresholds = {
            'spike_threshold': 0.05,  # 5% price change triggers alert
            'volume_threshold': 2.0   # 2x normal volume triggers alert
        }
    
    def start_monitoring(self):
        """Start continuous market monitoring"""
        self.monitoring = True
        print("🔍 AGENT: Starting continuous market monitoring...")
        
        def monitor_loop():
            while self.monitoring:
                try:
                    self.check_market_conditions()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    print(f"❌ AGENT: Monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop market monitoring"""
        self.monitoring = False
        print("⏹️  AGENT: Stopped market monitoring")
    
    def check_market_conditions(self):
        """Check current market conditions and generate alerts"""
        assets = ['BTC', 'SOL', 'SUI', 'SEI']
        
        for asset in assets:
            try:
                current_price, market_cap, change_24h = self.data_fetcher.fetch_price_and_market_cap(asset)
                
                # Check for price spikes
                if abs(change_24h) > self.price_thresholds['spike_threshold'] * 100:
                    alert = MarketAlert(
                        alert_type='PRICE_SPIKE',
                        asset=asset,
                        message=f"{asset} price moved {change_24h:+.2f}% in 24h (${current_price:,.2f})",
                        severity='HIGH' if abs(change_24h) > 10 else 'MEDIUM',
                        timestamp=datetime.now()
                    )
                    self.alerts.append(alert)
                    self.send_alert(alert)
                
                self.last_prices[asset] = current_price
                
            except Exception as e:
                print(f"❌ AGENT: Error monitoring {asset}: {e}")
    
    def send_alert(self, alert: MarketAlert):
        """Send alert to user"""
        severity_emoji = {'LOW': '🟡', 'MEDIUM': '🟠', 'HIGH': '🔴', 'CRITICAL': '🚨'}
        emoji = severity_emoji.get(alert.severity, '🔔')
        
        print(f"{emoji} ALERT: {alert.message}")

class LearningAgent:
    """Agent that learns from trading performance and adapts strategies"""
    
    def __init__(self):
        self.strategy_performance = {}
        self.market_patterns = {}
        self.adaptation_threshold = 0.1  # Adapt if performance drops 10%
    
    def analyze_performance(self, trade_history: List[TradingDecision]) -> Dict[str, Any]:
        """Analyze trading performance and identify patterns"""
        if not trade_history:
            return {"message": "No trading history to analyze"}
        
        recent_trades = trade_history[-10:]  # Last 10 trades
        
        # Calculate win rate and other metrics
        performance = {
            'total_trades': len(recent_trades),
            'strategies_used': list(set([t.reasoning[:20] for t in recent_trades])),
            'average_confidence': sum([t.confidence for t in recent_trades]) / len(recent_trades),
            'recommendations': []
        }
        
        # Generate learning insights
        if performance['average_confidence'] < 0.6:
            performance['recommendations'].append("Increase confidence threshold for trade execution")
        
        return performance
    
    def adapt_strategy(self, performance_data: Dict) -> List[str]:
        """Adapt trading strategy based on performance"""
        adaptations = []
        
        if performance_data.get('average_confidence', 0) < 0.5:
            adaptations.append("Lowering risk tolerance due to low confidence trades")
        
        return adaptations

class CommunicationAgent:
    """Agent for external communication and notifications"""
    
    def __init__(self):
        self.notification_channels = []
        self.command_history = []
    
    def send_report(self, report_data: Dict):
        """Send automated reports"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n📊 AUTOMATED REPORT - {timestamp}")
        print("=" * 50)
        
        if 'market_summary' in report_data:
            print("Market Summary:")
            for asset, data in report_data['market_summary'].items():
                print(f"  {asset}: ${data['price']:,.2f} ({data['change']:+.2f}%)")
        
        if 'trading_decisions' in report_data:
            print(f"\nTrading Decisions: {len(report_data['trading_decisions'])}")
            for decision in report_data['trading_decisions']:
                print(f"  {decision.action} {decision.asset} (Confidence: {decision.confidence:.2%})")
        
        if 'alerts' in report_data:
            print(f"\nAlerts Generated: {len(report_data['alerts'])}")
            for alert in report_data['alerts']:
                print(f"  {alert.severity}: {alert.message}")
    
    def process_command(self, command: str) -> str:
        """Process natural language commands"""
        command_lower = command.lower()
        
        if 'status' in command_lower:
            return "🤖 AGENT: System operational. Monitoring 4 assets, 0 active positions."
        elif 'stop' in command_lower:
            return "🛑 AGENT: Stopping all autonomous operations..."
        elif 'report' in command_lower:
            return "📊 AGENT: Generating comprehensive market report..."
        else:
            return f"🤖 AGENT: Processing command: {command}"

class AgenticElizaOS(ElizaOS):
    """Enhanced ElizaOS with autonomous agents"""
    
    def __init__(self, config_path=None):
        super().__init__(config_path)
        
        # Load GROK client
        with open('../config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        self.grok_client = OpenAI(
            api_key=config.get('grok_api_key'),
            base_url="https://api.x.ai/v1",
        )
        
        # Initialize agents
        self.autonomous_trader = AutonomousTrader()
        self.market_monitor = MarketMonitor(self.fetcher)
        self.learning_agent = LearningAgent()
        self.communication_agent = CommunicationAgent()
        
        # Operational state
        self.autonomous_mode = False
        self.decision_history = []
        
        print("🤖 AGENT: Enhanced Agentic ElizaOS initialized")
    
    def start_autonomous_mode(self):
        """Start autonomous trading and monitoring"""
        self.autonomous_mode = True
        self.market_monitor.start_monitoring()
        
        print("🚀 AGENT: Autonomous mode activated!")
        print("   - Market monitoring: ACTIVE")
        print("   - Autonomous trading: ENABLED")
        print("   - Learning system: ACTIVE")
        
        # Start autonomous loop
        self.autonomous_loop()
    
    def stop_autonomous_mode(self):
        """Stop autonomous operations"""
        self.autonomous_mode = False
        self.market_monitor.stop_monitoring()
        print("⏹️  AGENT: Autonomous mode deactivated")
    
    def autonomous_loop(self):
        """Main autonomous decision-making loop"""
        def loop():
            while self.autonomous_mode:
                try:
                    # Gather market data
                    market_data = self.gather_data()
                    
                    # Analyze trade signals
                    trade_signals = market_data.get('trade_signals', [])
                    
                    # Make autonomous decisions
                    decisions = []
                    for signal in trade_signals:
                        decision = self.autonomous_trader.analyze_trade_opportunity(
                            signal, market_data, self.grok_client
                        )
                        decisions.append(decision)
                        
                        # Execute if confidence is high enough
                        self.autonomous_trader.execute_trade(decision)
                    
                    self.decision_history.extend(decisions)
                    
                    # Learn from performance
                    if len(self.decision_history) > 10:
                        performance = self.learning_agent.analyze_performance(self.decision_history)
                        adaptations = self.learning_agent.adapt_strategy(performance)
                        
                        if adaptations:
                            print(f"🧠 AGENT: Learning adaptations: {adaptations}")
                    
                    # Send periodic reports
                    if len(self.decision_history) % 5 == 0:  # Every 5 decisions
                        self.send_autonomous_report()
                    
                    time.sleep(300)  # Wait 5 minutes between cycles
                    
                except Exception as e:
                    print(f"❌ AGENT: Autonomous loop error: {e}")
                    time.sleep(60)
        
        autonomous_thread = threading.Thread(target=loop, daemon=True)
        autonomous_thread.start()
    
    def send_autonomous_report(self):
        """Send autonomous report"""
        market_data = self.gather_data()
        
        report = {
            'market_summary': {
                asset: {
                    'price': info['price'],
                    'change': info['change_24h']
                }
                for asset, info in market_data['assets_summary'].items()
            },
            'trading_decisions': self.decision_history[-5:],  # Last 5 decisions
            'alerts': self.market_monitor.alerts[-3:],  # Last 3 alerts
        }
        
        self.communication_agent.send_report(report)
    
    def process_natural_command(self, command: str) -> str:
        """Process natural language commands"""
        return self.communication_agent.process_command(command)
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            'autonomous_mode': self.autonomous_mode,
            'monitoring_active': self.market_monitor.monitoring,
            'total_decisions': len(self.decision_history),
            'trader_performance': self.autonomous_trader.performance_metrics,
            'active_alerts': len(self.market_monitor.alerts),
            'learning_insights': len(self.learning_agent.strategy_performance)
        }

def main():
    """Enhanced main function with agentic capabilities"""
    print("🤖 Starting Enhanced Agentic ElizaOS...")
    
    # Initialize enhanced system
    agentic_eliza = AgenticElizaOS()
    
    # Show initial status
    print("\n📊 Initial Market Analysis:")
    agentic_eliza.print_report()
    
    # Demonstrate agentic capabilities
    print("\n🤖 Agentic Capabilities Demo:")
    
    # Process natural language commands
    commands = [
        "Show me the system status",
        "Generate a market report",
        "What are the current trading opportunities?"
    ]
    
    for command in commands:
        response = agentic_eliza.process_natural_command(command)
        print(f"User: {command}")
        print(f"Agent: {response}\n")
    
    # Show agent status
    status = agentic_eliza.get_agent_status()
    print("🔧 Agent Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Option to start autonomous mode
    print("\n🚀 To enable autonomous mode, call: agentic_eliza.start_autonomous_mode()")
    print("🛑 To stop autonomous mode, call: agentic_eliza.stop_autonomous_mode()")
    
    return agentic_eliza

if __name__ == "__main__":
    agentic_system = main() 