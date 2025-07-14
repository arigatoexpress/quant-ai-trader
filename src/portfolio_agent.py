"""
Portfolio Management Agent - Enhanced agentic system with multi-chain portfolio analysis
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import yaml
from openai import OpenAI

from portfolio_analyzer import MultiChainPortfolioAnalyzer, PortfolioRecommendation
from agentic_eliza import AgenticElizaOS, MarketAlert
from data_fetcher import DataFetcher


@dataclass
class PortfolioAlert:
    """Alert for portfolio-related events"""
    alert_type: str  # 'BALANCE_CHANGE', 'REBALANCE_NEEDED', 'RISK_THRESHOLD', 'OPPORTUNITY'
    message: str
    severity: str    # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    affected_wallets: List[str]
    recommended_action: str
    timestamp: datetime


@dataclass
class PortfolioDecision:
    """Autonomous portfolio management decision"""
    decision_type: str  # 'REBALANCE', 'RISK_REDUCE', 'OPPORTUNITY', 'HOLD'
    tokens_involved: List[str]
    confidence: float
    reasoning: str
    expected_impact: float
    risk_level: str
    execution_priority: str
    timestamp: datetime


class PortfolioManager:
    """Advanced portfolio management agent with AI-powered decisions"""
    
    def __init__(self, risk_tolerance: float = 0.15, rebalance_threshold: float = 0.05):
        self.risk_tolerance = risk_tolerance
        self.rebalance_threshold = rebalance_threshold
        self.portfolio_decisions = []
        self.portfolio_alerts = []
        self.last_analysis: Optional[Dict[str, Any]] = None
        self.monitoring_active = False
        
        # Performance tracking
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_rebalances': 0,
            'portfolio_value_change': 0.0,
            'risk_reduction_actions': 0,
            'opportunity_captures': 0
        }
    
    def analyze_portfolio_opportunity(self, analysis: Dict[str, Any], market_data: Dict[str, Any], grok_client) -> PortfolioDecision:
        """Analyze portfolio and make autonomous decisions"""
        try:
            metrics = analysis.get("metrics", {})
            recommendations = analysis.get("recommendations", [])
            
            prompt = f"""
            As an autonomous portfolio management agent, analyze this multi-chain portfolio situation:
            
            Portfolio Metrics:
            - Total Value: ${metrics.get('total_value', 0):,.2f}
            - Risk Level: {metrics.get('risk_level', 'N/A')}
            - Diversification Score: {metrics.get('diversification_score', 0):.1f}/100
            - Chain Allocations: {json.dumps(metrics.get('chain_allocations', {}), indent=2)}
            - Token Allocations: {json.dumps(metrics.get('token_allocations', {}), indent=2)}
            
            Current Recommendations: {len(recommendations)} active
            Risk Tolerance: {self.risk_tolerance}
            Rebalance Threshold: {self.rebalance_threshold}
            
            Market Context:
            {json.dumps(market_data.get('assets_summary', {}), indent=2)}
            
            Performance History:
            - Total Decisions: {self.performance_metrics['total_decisions']}
            - Successful Rebalances: {self.performance_metrics['successful_rebalances']}
            - Risk Reduction Actions: {self.performance_metrics['risk_reduction_actions']}
            
            Provide a portfolio management decision with:
            1. Decision Type (REBALANCE/RISK_REDUCE/OPPORTUNITY/HOLD)
            2. Tokens Involved
            3. Confidence Level (0-1)
            4. Reasoning
            5. Expected Impact
            6. Risk Level
            7. Execution Priority (HIGH/MEDIUM/LOW)
            
            Format response as JSON.
            """
            
            completion = grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.2
            )
            
            ai_response = completion.choices[0].message.content
            
            # Parse response and create decision
            decision = self.parse_portfolio_decision(ai_response, metrics, recommendations)
            
            return decision
            
        except Exception as e:
            return PortfolioDecision(
                decision_type='HOLD',
                tokens_involved=[],
                confidence=0.0,
                reasoning=f"Analysis error: {str(e)}",
                expected_impact=0.0,
                risk_level='HIGH',
                execution_priority='LOW',
                timestamp=datetime.now()
            )
    
    def parse_portfolio_decision(self, ai_response: str, metrics: Dict, recommendations: List) -> PortfolioDecision:
        """Parse AI response into structured portfolio decision"""
        try:
            # Try to parse as JSON first
            if '{' in ai_response and '}' in ai_response:
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                json_str = ai_response[json_start:json_end]
                decision_data = json.loads(json_str)
            else:
                # Fallback to text parsing
                decision_data = self.parse_decision_text(ai_response)
            
            return PortfolioDecision(
                decision_type=decision_data.get('decision_type', 'HOLD'),
                tokens_involved=decision_data.get('tokens_involved', []),
                confidence=float(decision_data.get('confidence', 0.5)),
                reasoning=decision_data.get('reasoning', 'AI analysis pending'),
                expected_impact=float(decision_data.get('expected_impact', 0.0)),
                risk_level=decision_data.get('risk_level', 'MEDIUM'),
                execution_priority=decision_data.get('execution_priority', 'MEDIUM'),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            # Fallback decision based on recommendations
            if recommendations:
                high_priority_recs = [r for r in recommendations if r.priority == "HIGH"]
                if high_priority_recs:
                    return PortfolioDecision(
                        decision_type='REBALANCE',
                        tokens_involved=[r.token for r in high_priority_recs],
                        confidence=0.7,
                        reasoning="High priority recommendations detected",
                        expected_impact=sum(r.estimated_impact for r in high_priority_recs),
                        risk_level=metrics.get('risk_level', 'MEDIUM'),
                        execution_priority='HIGH',
                        timestamp=datetime.now()
                    )
            
            return PortfolioDecision(
                decision_type='HOLD',
                tokens_involved=[],
                confidence=0.5,
                reasoning="No clear action needed",
                expected_impact=0.0,
                risk_level='MEDIUM',
                execution_priority='LOW',
                timestamp=datetime.now()
            )
    
    def parse_decision_text(self, text: str) -> Dict[str, Any]:
        """Parse decision from text format"""
        decision_data = {}
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if 'decision_type' in line.lower() or 'decision type' in line.lower():
                decision_data['decision_type'] = line.split(':')[-1].strip().upper()
            elif 'tokens_involved' in line.lower() or 'tokens involved' in line.lower():
                tokens_str = line.split(':')[-1].strip()
                decision_data['tokens_involved'] = [t.strip() for t in tokens_str.split(',')]
            elif 'confidence' in line.lower():
                try:
                    confidence_str = line.split(':')[-1].strip()
                    decision_data['confidence'] = float(confidence_str)
                except:
                    decision_data['confidence'] = 0.5
            elif 'reasoning' in line.lower():
                decision_data['reasoning'] = line.split(':')[-1].strip()
            elif 'expected_impact' in line.lower():
                try:
                    impact_str = line.split(':')[-1].strip()
                    decision_data['expected_impact'] = float(impact_str)
                except:
                    decision_data['expected_impact'] = 0.0
            elif 'risk_level' in line.lower():
                decision_data['risk_level'] = line.split(':')[-1].strip().upper()
            elif 'execution_priority' in line.lower():
                decision_data['execution_priority'] = line.split(':')[-1].strip().upper()
        
        return decision_data
    
    def execute_portfolio_decision(self, decision: PortfolioDecision) -> bool:
        """Execute portfolio management decision"""
        if decision.confidence < 0.6:
            print(f"🤖 PORTFOLIO: Skipping decision - confidence too low ({decision.confidence:.1%})")
            return False
        
        print(f"🚀 PORTFOLIO: Executing {decision.decision_type} decision")
        print(f"   📊 Tokens: {', '.join(decision.tokens_involved)}")
        print(f"   🎯 Confidence: {decision.confidence:.1%}")
        print(f"   📈 Expected Impact: {decision.expected_impact:.2f}%")
        print(f"   ⚖️  Risk Level: {decision.risk_level}")
        print(f"   ⭐ Priority: {decision.execution_priority}")
        print(f"   💭 Reasoning: {decision.reasoning}")
        
        # Update performance metrics
        self.performance_metrics['total_decisions'] += 1
        
        if decision.decision_type == 'REBALANCE':
            self.performance_metrics['successful_rebalances'] += 1
        elif decision.decision_type == 'RISK_REDUCE':
            self.performance_metrics['risk_reduction_actions'] += 1
        elif decision.decision_type == 'OPPORTUNITY':
            self.performance_metrics['opportunity_captures'] += 1
        
        # Add to decision history
        self.portfolio_decisions.append(decision)
        
        return True
    
    def generate_portfolio_alert(self, analysis: Dict[str, Any], previous_analysis: Optional[Dict[str, Any]] = None) -> List[PortfolioAlert]:
        """Generate portfolio alerts based on analysis"""
        alerts = []
        metrics = analysis.get("metrics", {})
        
        # Risk level alerts
        risk_level = metrics.get('risk_level', 'MEDIUM')
        if risk_level == 'HIGH':
            alerts.append(PortfolioAlert(
                alert_type='RISK_THRESHOLD',
                message=f"Portfolio risk level is HIGH - immediate attention required",
                severity='HIGH',
                affected_wallets=['ALL'],
                recommended_action='Reduce position sizes and increase diversification',
                timestamp=datetime.now()
            ))
        
        # Concentration alerts
        chain_allocations = metrics.get('chain_allocations', {})
        for chain, allocation in chain_allocations.items():
            if allocation > 70:
                alerts.append(PortfolioAlert(
                    alert_type='BALANCE_CHANGE',
                    message=f"{chain} chain concentration at {allocation:.1f}% - rebalancing recommended",
                    severity='MEDIUM',
                    affected_wallets=[chain],
                    recommended_action=f'Reduce {chain} allocation to <60%',
                    timestamp=datetime.now()
                ))
        
        # Opportunity alerts
        recommendations = analysis.get("recommendations", [])
        high_priority_recs = [r for r in recommendations if r.priority == "HIGH"]
        if high_priority_recs:
            alerts.append(PortfolioAlert(
                alert_type='OPPORTUNITY',
                message=f"{len(high_priority_recs)} high-priority opportunities detected",
                severity='MEDIUM',
                affected_wallets=['MULTIPLE'],
                recommended_action='Review and execute high-priority recommendations',
                timestamp=datetime.now()
            ))
        
        # Value change alerts (if previous analysis available)
        if previous_analysis:
            previous_value = previous_analysis.get("metrics", {}).get('total_value', 0)
            current_value = metrics.get('total_value', 0)
            
            if previous_value > 0:
                change_percent = ((current_value - previous_value) / previous_value) * 100
                
                if abs(change_percent) > 5:  # 5% threshold
                    severity = 'HIGH' if abs(change_percent) > 15 else 'MEDIUM'
                    direction = 'increased' if change_percent > 0 else 'decreased'
                    
                    alerts.append(PortfolioAlert(
                        alert_type='BALANCE_CHANGE',
                        message=f"Portfolio value {direction} by {abs(change_percent):.1f}% (${current_value:,.2f})",
                        severity=severity,
                        affected_wallets=['ALL'],
                        recommended_action='Review market conditions and adjust if needed',
                        timestamp=datetime.now()
                    ))
        
        return alerts
    
    def send_portfolio_alert(self, alert: PortfolioAlert):
        """Send portfolio alert to user"""
        severity_emoji = {'LOW': '🟡', 'MEDIUM': '🟠', 'HIGH': '🔴', 'CRITICAL': '🚨'}
        emoji = severity_emoji.get(alert.severity, '🔔')
        
        print(f"{emoji} PORTFOLIO ALERT: {alert.message}")
        print(f"   📍 Affected: {', '.join(alert.affected_wallets)}")
        print(f"   💡 Recommended: {alert.recommended_action}")


class PortfolioAgent:
    """Enhanced portfolio management agent with autonomous capabilities"""
    
    def __init__(self, config_path=None):
        # Initialize components
        self.portfolio_analyzer = MultiChainPortfolioAnalyzer(config_path)
        self.portfolio_manager = PortfolioManager()
        self.data_fetcher = DataFetcher()
        
        # Load GROK client
        if config_path is None:
            config_path = '../config/config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.grok_client = OpenAI(
            api_key=config.get('grok_api_key'),
            base_url="https://api.x.ai/v1",
        )
        
        # State management
        self.autonomous_mode = False
        self.monitoring_thread = None
        self.last_analysis_time = None
        self.analysis_interval = 1800  # 30 minutes
        
        print("🔗 Portfolio Agent initialized")
        print("   - Multi-chain analysis: Ready")
        print("   - AI recommendations: Active")
        print("   - Autonomous management: Standby")
    
    def start_autonomous_portfolio_management(self):
        """Start autonomous portfolio management"""
        self.autonomous_mode = True
        
        print("🚀 PORTFOLIO AGENT: Autonomous mode activated!")
        print("   - Portfolio monitoring: ACTIVE")
        print("   - AI decision making: ENABLED")
        print("   - Risk management: ACTIVE")
        print("   - Rebalancing alerts: ENABLED")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self.autonomous_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_autonomous_portfolio_management(self):
        """Stop autonomous portfolio management"""
        self.autonomous_mode = False
        print("⏹️  PORTFOLIO AGENT: Autonomous mode deactivated")
    
    def autonomous_monitoring_loop(self):
        """Main autonomous monitoring loop"""
        while self.autonomous_mode:
            try:
                print(f"\n📊 PORTFOLIO AGENT: Starting analysis cycle...")
                
                # Perform comprehensive portfolio analysis
                analysis = self.portfolio_analyzer.analyze_full_portfolio()
                
                # Generate alerts
                alerts = self.portfolio_manager.generate_portfolio_alert(
                    analysis, 
                    self.portfolio_manager.last_analysis
                )
                
                # Send alerts
                for alert in alerts:
                    self.portfolio_manager.send_portfolio_alert(alert)
                    self.portfolio_manager.portfolio_alerts.append(alert)
                
                # Get market data for context
                market_data = self.get_market_context()
                
                # Make autonomous decision
                decision = self.portfolio_manager.analyze_portfolio_opportunity(
                    analysis, market_data, self.grok_client
                )
                
                # Execute decision if confidence is high enough
                if decision.confidence > 0.6:
                    self.portfolio_manager.execute_portfolio_decision(decision)
                
                # Update state
                self.portfolio_manager.last_analysis = analysis
                self.last_analysis_time = datetime.now()
                
                # Send periodic report
                if len(self.portfolio_manager.portfolio_decisions) % 3 == 0:
                    self.send_portfolio_report()
                
                print(f"💤 PORTFOLIO AGENT: Sleeping for {self.analysis_interval} seconds...")
                time.sleep(self.analysis_interval)
                
            except Exception as e:
                print(f"❌ PORTFOLIO AGENT: Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def get_market_context(self) -> Dict[str, Any]:
        """Get current market context"""
        try:
            # Get current prices
            assets = ['BTC', 'ETH', 'SOL', 'SUI', 'SEI']
            market_data = {'assets_summary': {}}
            
            for asset in assets:
                price, market_cap, change_24h = self.data_fetcher.fetch_price_and_market_cap(asset)
                market_data['assets_summary'][asset] = {
                    'price': price,
                    'market_cap': market_cap,
                    'change_24h': change_24h
                }
            
            return market_data
            
        except Exception as e:
            print(f"❌ Error getting market context: {e}")
            return {'assets_summary': {}}
    
    def send_portfolio_report(self):
        """Send comprehensive portfolio report"""
        print(f"\n📊 PORTFOLIO AGENT REPORT")
        print("=" * 60)
        
        # Performance metrics
        metrics = self.portfolio_manager.performance_metrics
        print(f"🎯 Performance Metrics:")
        print(f"   Total Decisions: {metrics['total_decisions']}")
        print(f"   Successful Rebalances: {metrics['successful_rebalances']}")
        print(f"   Risk Reduction Actions: {metrics['risk_reduction_actions']}")
        print(f"   Opportunity Captures: {metrics['opportunity_captures']}")
        
        # Recent decisions
        recent_decisions = self.portfolio_manager.portfolio_decisions[-5:]
        if recent_decisions:
            print(f"\n🚀 Recent Decisions:")
            for i, decision in enumerate(recent_decisions, 1):
                print(f"   {i}. {decision.decision_type}: {', '.join(decision.tokens_involved)}")
                print(f"      Confidence: {decision.confidence:.1%}, Impact: {decision.expected_impact:.2f}%")
        
        # Active alerts
        recent_alerts = self.portfolio_manager.portfolio_alerts[-3:]
        if recent_alerts:
            print(f"\n🚨 Recent Alerts:")
            for alert in recent_alerts:
                print(f"   {alert.alert_type}: {alert.message}")
        
        # Current portfolio status
        if self.portfolio_manager.last_analysis:
            analysis = self.portfolio_manager.last_analysis
            metrics = analysis.get("metrics", {})
            
            print(f"\n💰 Current Portfolio Status:")
            print(f"   Total Value: ${metrics.get('total_value', 0):,.2f}")
            print(f"   Risk Level: {metrics.get('risk_level', 'N/A')}")
            print(f"   Diversification: {metrics.get('diversification_score', 0):.1f}/100")
            print(f"   Active Chains: {metrics.get('total_chains', 0)}")
            print(f"   Total Tokens: {metrics.get('total_tokens', 0)}")
        
        print("=" * 60)
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio status"""
        return {
            'autonomous_mode': self.autonomous_mode,
            'last_analysis_time': self.last_analysis_time,
            'total_decisions': len(self.portfolio_manager.portfolio_decisions),
            'total_alerts': len(self.portfolio_manager.portfolio_alerts),
            'performance_metrics': self.portfolio_manager.performance_metrics,
            'monitoring_active': self.autonomous_mode
        }
    
    def force_portfolio_analysis(self):
        """Force immediate portfolio analysis"""
        print("🔍 PORTFOLIO AGENT: Force analysis requested...")
        
        analysis = self.portfolio_analyzer.analyze_full_portfolio()
        self.portfolio_analyzer.print_portfolio_report(analysis)
        
        return analysis
    
    def get_portfolio_recommendations(self) -> List[PortfolioRecommendation]:
        """Get current portfolio recommendations"""
        if self.portfolio_manager.last_analysis:
            return self.portfolio_manager.last_analysis.get("recommendations", [])
        return []


def main():
    """Main function for testing portfolio agent"""
    print("🎯 PORTFOLIO AGENT DEMO")
    print("=" * 60)
    
    # Initialize portfolio agent
    agent = PortfolioAgent()
    
    # Start autonomous management
    agent.start_autonomous_portfolio_management()
    
    # Let it run for a bit
    try:
        time.sleep(120)  # Run for 2 minutes
    except KeyboardInterrupt:
        print("\n⏹️  Stopping portfolio agent...")
    
    # Stop autonomous management
    agent.stop_autonomous_portfolio_management()
    
    # Show final status
    status = agent.get_portfolio_status()
    print(f"\n📊 Final Status: {status}")


if __name__ == "__main__":
    main() 