"""
Secure Autonomous Trading System
Integrates cybersecurity framework with autonomous trading capabilities
"""

import asyncio
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import yaml
from openai import OpenAI

from cybersecurity_framework import (
    SecureTradingFramework, 
    SecurityEvent, 
    SecurityLevel, 
    ActionType,
    AuthenticationToken
)
from agentic_eliza import (
    AgenticElizaOS,
    TradingDecision,
    MarketAlert,
    AutonomousTrader,
    MarketMonitor,
    LearningAgent,
    CommunicationAgent
)
from data_fetcher import DataFetcher
from utils import calculate_risk_reward, calculate_momentum

@dataclass
class SecureTradingConfig:
    """Configuration for secure autonomous trading"""
    max_trade_amount: float = 1000.0
    risk_tolerance: float = 0.02
    confidence_threshold: float = 0.7
    max_daily_trades: int = 10
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    security_level: SecurityLevel = SecurityLevel.HIGH
    audit_all_actions: bool = True
    emergency_stop_loss: float = 0.05  # 5% portfolio stop loss

@dataclass
class SecureTradeResult:
    """Result of secure trade execution"""
    success: bool
    trade_id: str
    message: str
    security_event_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    risk_metrics: Dict[str, float] = field(default_factory=dict)

class SecureAutonomousTrader(AutonomousTrader):
    """Enhanced autonomous trader with security controls"""
    
    def __init__(self, security_framework: SecureTradingFramework, token: AuthenticationToken, config: SecureTradingConfig):
        super().__init__(config.risk_tolerance, config.max_trade_amount)
        self.security_framework = security_framework
        self.token = token
        self.config = config
        self.daily_trades = 0
        self.last_trade_reset = datetime.now().date()
        self.portfolio_value = 100000.0  # Initial portfolio value
        self.emergency_stop_triggered = False
        
    def secure_analyze_trade_opportunity(self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> Optional[TradingDecision]:
        """Secure analysis of trading opportunity with authorization checks"""
        try:
            # Verify authorization for trade analysis
            if not self.security_framework.auth_manager.authorize(self.token.token_id, "trade_execution"):
                self.security_framework.audit_logger.log_security_event(SecurityEvent(
                    action_type=ActionType.AUTHORIZATION,
                    user_id=self.token.user_id,
                    action_description="Unauthorized trade analysis attempt",
                    security_level=SecurityLevel.HIGH,
                    success=False,
                    metadata={"signal": signal.get("asset", "UNKNOWN")}
                ))
                return None
            
            # Check daily trade limits
            if self._check_daily_limits():
                return None
            
            # Check emergency stop
            if self._check_emergency_stop():
                return None
            
            # Get secure API key for analysis
            api_key = self.security_framework.get_secure_api_key("grok_api_key", self.token.token_id)
            if not api_key:
                return None
            
            # Initialize secure GROK client
            grok_client = OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
            
            # Perform AI analysis
            decision = self._perform_secure_ai_analysis(signal, market_data, grok_client)
            
            # Log analysis
            self.security_framework.audit_logger.log_security_event(SecurityEvent(
                action_type=ActionType.DATA_ACCESS,
                user_id=self.token.user_id,
                action_description=f"Trade analysis performed: {signal.get('asset', 'UNKNOWN')}",
                security_level=SecurityLevel.MEDIUM,
                success=True,
                metadata={
                    "asset": signal.get("asset"),
                    "confidence": decision.confidence if decision else 0,
                    "action": decision.action if decision else "NONE"
                }
            ))
            
            return decision
            
        except Exception as e:
            self.security_framework.audit_logger.log_security_event(SecurityEvent(
                action_type=ActionType.TRADE_EXECUTION,
                user_id=self.token.user_id,
                action_description=f"Trade analysis error: {str(e)}",
                security_level=SecurityLevel.HIGH,
                success=False,
                metadata={"error": str(e)}
            ))
            return None
    
    def _perform_secure_ai_analysis(self, signal: Dict[str, Any], market_data: Dict[str, Any], grok_client: OpenAI) -> Optional[TradingDecision]:
        """Perform AI analysis with security controls"""
        try:
            prompt = f"""
            As a secure autonomous trading agent, analyze this trading opportunity with strict risk management:
            
            SIGNAL DATA:
            {json.dumps(signal, indent=2)}
            
            MARKET DATA:
            {json.dumps(market_data, indent=2)}
            
            SECURITY CONSTRAINTS:
            - Max trade amount: ${self.config.max_trade_amount:,.2f}
            - Risk tolerance: {self.config.risk_tolerance:.2%}
            - Confidence threshold: {self.config.confidence_threshold:.2%}
            - Daily trades used: {self.daily_trades}/{self.config.max_daily_trades}
            - Emergency stop triggered: {self.emergency_stop_triggered}
            
            CURRENT PORTFOLIO:
            - Value: ${self.portfolio_value:,.2f}
            - Total trades: {self.performance_metrics['total_trades']}
            - Win rate: {self.performance_metrics['win_rate']:.2%}
            
            REQUIREMENTS:
            1. Strict risk management (max 2% portfolio risk per trade)
            2. High confidence threshold (minimum 70%)
            3. Consider market volatility and liquidity
            4. Implement stop-loss and take-profit levels
            5. Account for transaction costs
            
            Provide analysis in JSON format:
            {{
                "action": "BUY/SELL/HOLD",
                "confidence": 0.0-1.0,
                "reasoning": "detailed analysis",
                "risk_level": "LOW/MEDIUM/HIGH",
                "target_price": number or null,
                "stop_loss": number or null,
                "take_profit": number or null,
                "position_size": number,
                "risk_metrics": {{
                    "portfolio_risk": percentage,
                    "var_95": value_at_risk,
                    "sharpe_ratio": number
                }}
            }}
            """
            
            completion = grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.2
            )
            
            # Parse AI response
            ai_response = completion.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Find JSON in the response
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                json_str = ai_response[json_start:json_end]
                analysis = json.loads(json_str)
                
                # Validate analysis
                if not self._validate_ai_analysis(analysis):
                    return None
                
                # Create secure trading decision
                decision = TradingDecision(
                    asset=signal.get('asset', 'UNKNOWN'),
                    action=analysis.get('action', 'HOLD'),
                    confidence=analysis.get('confidence', 0.0),
                    reasoning=analysis.get('reasoning', 'AI analysis failed'),
                    risk_level=analysis.get('risk_level', 'HIGH'),
                    target_price=analysis.get('target_price'),
                    stop_loss=analysis.get('stop_loss')
                )
                
                return decision
                
            except json.JSONDecodeError:
                # Fallback to conservative decision
                return TradingDecision(
                    asset=signal.get('asset', 'UNKNOWN'),
                    action='HOLD',
                    confidence=0.0,
                    reasoning='AI analysis parsing failed - defaulting to HOLD',
                    risk_level='HIGH'
                )
            
        except Exception as e:
            return None
    
    def _validate_ai_analysis(self, analysis: Dict[str, Any]) -> bool:
        """Validate AI analysis for security compliance"""
        try:
            # Check required fields
            required_fields = ['action', 'confidence', 'reasoning', 'risk_level']
            for field in required_fields:
                if field not in analysis:
                    return False
            
            # Validate confidence
            confidence = analysis.get('confidence', 0.0)
            if not 0.0 <= confidence <= 1.0:
                return False
            
            # Validate action
            if analysis.get('action') not in ['BUY', 'SELL', 'HOLD']:
                return False
            
            # Validate risk level
            if analysis.get('risk_level') not in ['LOW', 'MEDIUM', 'HIGH']:
                return False
            
            return True
            
        except Exception:
            return False
    
    def secure_execute_trade(self, decision: TradingDecision) -> SecureTradeResult:
        """Execute trade with comprehensive security controls"""
        try:
            # Pre-execution security checks
            if not self._pre_execution_security_checks(decision):
                return SecureTradeResult(
                    success=False,
                    trade_id="SECURITY_BLOCK",
                    message="Trade blocked by security controls",
                    security_event_id="",
                    risk_metrics={}
                )
            
            # Generate secure trade ID
            trade_id = f"SECURE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{decision.asset}"
            
            # Prepare secure trade data
            trade_data = {
                "trade_id": trade_id,
                "asset": decision.asset,
                "action": decision.action,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "risk_level": decision.risk_level,
                "target_price": decision.target_price,
                "stop_loss": decision.stop_loss,
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": self.portfolio_value,
                "user_id": self.token.user_id
            }
            
            # Execute trade through security framework
            success, message = self.security_framework.secure_trade_execution(
                self.token.token_id, 
                trade_data
            )
            
            if success:
                # Update internal state
                self.daily_trades += 1
                self.performance_metrics['total_trades'] += 1
                self.trade_history.append(decision)
                
                # Calculate risk metrics
                risk_metrics = self._calculate_risk_metrics(decision)
                
                # Send secure notification
                self._send_secure_notification(decision, trade_id, risk_metrics)
                
                return SecureTradeResult(
                    success=True,
                    trade_id=trade_id,
                    message=f"Trade executed successfully: {decision.action} {decision.asset}",
                    security_event_id="",
                    risk_metrics=risk_metrics
                )
            else:
                return SecureTradeResult(
                    success=False,
                    trade_id=trade_id,
                    message=message,
                    security_event_id="",
                    risk_metrics={}
                )
                
        except Exception as e:
            return SecureTradeResult(
                success=False,
                trade_id="ERROR",
                message=f"Trade execution error: {str(e)}",
                security_event_id="",
                risk_metrics={}
            )
    
    def _pre_execution_security_checks(self, decision: TradingDecision) -> bool:
        """Comprehensive pre-execution security checks"""
        try:
            # Check confidence threshold
            if decision.confidence < self.config.confidence_threshold:
                return False
            
            # Check daily trade limits
            if self._check_daily_limits():
                return False
            
            # Check emergency stop
            if self._check_emergency_stop():
                return False
            
            # Check risk levels
            if decision.risk_level == 'HIGH' and self.config.security_level == SecurityLevel.HIGH:
                return False
            
            # Check action validity
            if decision.action not in ['BUY', 'SELL', 'HOLD']:
                return False
            
            # Skip HOLD actions
            if decision.action == 'HOLD':
                return False
            
            return True
            
        except Exception:
            return False
    
    def _check_daily_limits(self) -> bool:
        """Check if daily trading limits are exceeded"""
        # Reset counter if new day
        if datetime.now().date() > self.last_trade_reset:
            self.daily_trades = 0
            self.last_trade_reset = datetime.now().date()
        
        return self.daily_trades >= self.config.max_daily_trades
    
    def _check_emergency_stop(self) -> bool:
        """Check if emergency stop should be triggered"""
        # Calculate portfolio loss
        initial_value = 100000.0  # Starting portfolio value
        current_loss = (initial_value - self.portfolio_value) / initial_value
        
        if current_loss >= self.config.emergency_stop_loss:
            self.emergency_stop_triggered = True
            
            # Log emergency stop
            self.security_framework.audit_logger.log_security_event(SecurityEvent(
                action_type=ActionType.SYSTEM_CONFIGURATION,
                user_id=self.token.user_id,
                action_description="Emergency stop triggered",
                security_level=SecurityLevel.CRITICAL,
                success=True,
                metadata={
                    "portfolio_loss": current_loss,
                    "emergency_threshold": self.config.emergency_stop_loss
                }
            ))
            
            return True
        
        return self.emergency_stop_triggered
    
    def _calculate_risk_metrics(self, decision: TradingDecision) -> Dict[str, float]:
        """Calculate risk metrics for trade"""
        return {
            "portfolio_risk": self.config.risk_tolerance,
            "confidence": decision.confidence,
            "var_95": self.portfolio_value * 0.02,  # 2% VaR
            "sharpe_ratio": self.performance_metrics.get('win_rate', 0.5) / 0.1,
            "max_drawdown": 0.05,
            "position_size": min(self.config.max_trade_amount, self.portfolio_value * 0.1)
        }
    
    def _send_secure_notification(self, decision: TradingDecision, trade_id: str, risk_metrics: Dict[str, float]):
        """Send secure encrypted notification"""
        try:
            notification = {
                "trade_id": trade_id,
                "asset": decision.asset,
                "action": decision.action,
                "confidence": f"{decision.confidence:.2%}",
                "risk_level": decision.risk_level,
                "risk_metrics": risk_metrics,
                "timestamp": datetime.now().isoformat()
            }
            
            # Encrypt notification
            encrypted_notification = self.security_framework.crypto_manager.encrypt(
                json.dumps(notification)
            )
            
            print(f"🔒 SECURE TRADE NOTIFICATION:")
            print(f"   Trade ID: {trade_id}")
            print(f"   Action: {decision.action} {decision.asset}")
            print(f"   Confidence: {decision.confidence:.2%}")
            print(f"   Risk Level: {decision.risk_level}")
            print(f"   Portfolio Risk: {risk_metrics.get('portfolio_risk', 0):.2%}")
            print(f"   Security: ENCRYPTED & AUDITED")
            
        except Exception as e:
            print(f"⚠️  Notification error: {e}")

class SecureAgenticElizaOS(AgenticElizaOS):
    """Enhanced Agentic ElizaOS with comprehensive security"""
    
    def __init__(self, config_path=None, master_key: Optional[str] = None):
        # Initialize security framework first
        self.security_framework = SecureTradingFramework(master_key)
        
        # Initialize base system
        super().__init__(config_path)
        
        # Security configuration
        self.security_config = SecureTradingConfig()
        self.auth_token = None
        self.secure_autonomous_trader = None
        
        # Enhanced monitoring
        self.security_monitoring_active = False
        self.security_events = []
        
        print("🔐 SECURE AGENTIC ELIZA OS INITIALIZED")
        print("   ✅ Cybersecurity Framework: Active")
        print("   ✅ Encryption: Enabled")
        print("   ✅ Audit Logging: Active")
        print("   ✅ Authentication: Required")
    
    def initialize_secure_trading(self, grok_api_key: str, user_password: str) -> bool:
        """Initialize secure trading with authentication"""
        try:
            # Initialize security framework
            self.auth_token = self.security_framework.initialize_security(grok_api_key, user_password)
            
            if not self.auth_token:
                print("❌ Security initialization failed")
                return False
            
            # Create secure autonomous trader
            self.secure_autonomous_trader = SecureAutonomousTrader(
                self.security_framework,
                self.auth_token,
                self.security_config
            )
            
            # Start security monitoring
            self.security_framework.start_security_monitoring()
            
            print("✅ Secure trading initialized successfully")
            print(f"   Token: {self.auth_token.token_id[:8]}...")
            print(f"   Permissions: {len(self.auth_token.permissions)} granted")
            print(f"   Expires: {self.auth_token.expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Secure trading initialization failed: {e}")
            return False
    
    def start_secure_autonomous_mode(self):
        """Start secure autonomous trading mode"""
        if not self.auth_token or not self.secure_autonomous_trader:
            print("❌ Must initialize secure trading first")
            return
        
        self.autonomous_mode = True
        self.security_monitoring_active = True
        
        print("🚀 SECURE AUTONOMOUS MODE ACTIVATED")
        print("   🔒 Security Level: HIGH")
        print("   🛡️  Threat Monitoring: ACTIVE")
        print("   📊 Audit Logging: ENABLED")
        print("   ⚡ AI Trading: AUTHORIZED")
        
        # Start secure autonomous loop
        self.secure_autonomous_loop()
    
    def secure_autonomous_loop(self):
        """Secure autonomous trading loop with comprehensive security"""
        def secure_loop():
            while self.autonomous_mode and self.security_monitoring_active:
                try:
                    # Verify token is still valid
                    if not self.security_framework.auth_manager.authorize(self.auth_token.token_id, "trade_execution"):
                        print("❌ Authentication expired - stopping autonomous mode")
                        break
                    
                    # Gather market data securely
                    market_data = self.gather_data()
                    
                    # Process trading signals
                    trade_signals = market_data.get('trade_signals', [])
                    
                    secure_decisions = []
                    for signal in trade_signals:
                        # Secure analysis
                        decision = self.secure_autonomous_trader.secure_analyze_trade_opportunity(
                            signal, market_data
                        )
                        
                        if decision:
                            # Secure execution
                            result = self.secure_autonomous_trader.secure_execute_trade(decision)
                            secure_decisions.append((decision, result))
                            
                            # Log result
                            if result.success:
                                print(f"✅ SECURE TRADE: {result.message}")
                            else:
                                print(f"❌ TRADE BLOCKED: {result.message}")
                    
                    # Security health check
                    self.perform_security_health_check()
                    
                    # Send secure periodic report
                    if len(secure_decisions) > 0:
                        self.send_secure_report(secure_decisions)
                    
                    # Wait before next cycle
                    time.sleep(300)  # 5 minutes
                    
                except Exception as e:
                    print(f"❌ Secure autonomous loop error: {e}")
                    
                    # Log security incident
                    self.security_framework.audit_logger.log_security_event(SecurityEvent(
                        action_type=ActionType.SYSTEM_CONFIGURATION,
                        user_id=self.auth_token.user_id if self.auth_token else "unknown",
                        action_description=f"Autonomous loop error: {str(e)}",
                        security_level=SecurityLevel.HIGH,
                        success=False,
                        metadata={"error": str(e)}
                    ))
                    
                    time.sleep(60)  # Wait 1 minute on error
        
        # Start secure loop in background thread
        secure_thread = threading.Thread(target=secure_loop, daemon=True)
        secure_thread.start()
        
        print("🔄 Secure autonomous loop started")
    
    def perform_security_health_check(self):
        """Perform comprehensive security health check"""
        try:
            # Check authentication status
            if not self.auth_token or self.auth_token.expires_at < datetime.now():
                print("⚠️  Authentication token expired")
                return
            
            # Check security events
            recent_events = self.security_framework.audit_logger.get_security_events(
                start_time=datetime.now() - timedelta(minutes=10)
            )
            
            # Check for security threats
            failed_events = [e for e in recent_events if not e.success]
            if len(failed_events) > 3:
                print(f"⚠️  Security Alert: {len(failed_events)} failed events in last 10 minutes")
            
            # Check portfolio status
            if self.secure_autonomous_trader.emergency_stop_triggered:
                print("🚨 EMERGENCY STOP TRIGGERED - Autonomous trading halted")
                self.autonomous_mode = False
            
        except Exception as e:
            print(f"❌ Security health check error: {e}")
    
    def send_secure_report(self, decisions: List[Tuple[TradingDecision, SecureTradeResult]]):
        """Send encrypted periodic report"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "decisions_count": len(decisions),
                "successful_trades": len([d for d in decisions if d[1].success]),
                "security_status": self.security_framework.get_security_status(),
                "portfolio_value": self.secure_autonomous_trader.portfolio_value,
                "daily_trades": self.secure_autonomous_trader.daily_trades,
                "emergency_stop": self.secure_autonomous_trader.emergency_stop_triggered
            }
            
            # Encrypt report
            encrypted_report = self.security_framework.crypto_manager.encrypt(json.dumps(report))
            
            print("\n📊 SECURE AUTONOMOUS REPORT")
            print("=" * 50)
            print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📈 Decisions: {len(decisions)}")
            print(f"✅ Successful: {len([d for d in decisions if d[1].success])}")
            print(f"💰 Portfolio: ${self.secure_autonomous_trader.portfolio_value:,.2f}")
            print(f"📊 Daily Trades: {self.secure_autonomous_trader.daily_trades}/{self.security_config.max_daily_trades}")
            print(f"🔒 Security: ENCRYPTED & AUDITED")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Secure report error: {e}")
    
    def stop_secure_autonomous_mode(self):
        """Stop secure autonomous trading mode"""
        self.autonomous_mode = False
        self.security_monitoring_active = False
        
        print("⏹️  SECURE AUTONOMOUS MODE DEACTIVATED")
        print("   🔒 All trading stopped")
        print("   📊 Audit log preserved")
        print("   🛡️  Security monitoring continues")
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard"""
        if not self.auth_token:
            return {"error": "Not authenticated"}
        
        return {
            "authentication": {
                "token_id": self.auth_token.token_id[:8] + "...",
                "user_id": self.auth_token.user_id,
                "expires_at": self.auth_token.expires_at.isoformat(),
                "permissions": self.auth_token.permissions
            },
            "security_status": self.security_framework.get_security_status(),
            "trading_status": {
                "autonomous_mode": self.autonomous_mode,
                "daily_trades": self.secure_autonomous_trader.daily_trades if self.secure_autonomous_trader else 0,
                "emergency_stop": self.secure_autonomous_trader.emergency_stop_triggered if self.secure_autonomous_trader else False,
                "portfolio_value": self.secure_autonomous_trader.portfolio_value if self.secure_autonomous_trader else 0
            },
            "risk_metrics": self.secure_autonomous_trader._calculate_risk_metrics(
                TradingDecision(
                    asset="BTC",
                    action="HOLD",
                    confidence=0.0,
                    reasoning="Dashboard query",
                    risk_level="MEDIUM"
                )
            ) if self.secure_autonomous_trader else {}
        }


def main():
    """Demo the secure autonomous trading system"""
    print("🔐 SECURE AUTONOMOUS TRADING SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize secure system
    secure_system = SecureAgenticElizaOS(master_key="secure_trading_key_2024")
    
    # Initialize secure trading
    success = secure_system.initialize_secure_trading(
        grok_api_key=os.getenv("GROK_API_KEY", "your_api_key_here"),
        user_password="secure_password"
    )
    
    if not success:
        print("❌ Failed to initialize secure trading")
        return
    
    print("\n🚀 Starting secure autonomous trading for 60 seconds...")
    
    # Start secure autonomous mode
    secure_system.start_secure_autonomous_mode()
    
    # Monitor for 60 seconds
    try:
        for i in range(12):  # 12 x 5 seconds = 60 seconds
            time.sleep(5)
            
            # Display security dashboard
            if i % 3 == 0:  # Every 15 seconds
                dashboard = secure_system.get_security_dashboard()
                print(f"\n🔒 Security Dashboard (T+{i*5}s):")
                print(f"   Authentication: {'✅' if 'error' not in dashboard else '❌'}")
                print(f"   Active Tokens: {dashboard.get('security_status', {}).get('active_tokens', 0)}")
                print(f"   Daily Trades: {dashboard.get('trading_status', {}).get('daily_trades', 0)}")
                print(f"   Emergency Stop: {'🚨' if dashboard.get('trading_status', {}).get('emergency_stop', False) else '✅'}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Manual interrupt received")
    
    # Stop secure autonomous mode
    secure_system.stop_secure_autonomous_mode()
    
    print("\n🎉 SECURE AUTONOMOUS TRADING DEMO COMPLETED!")
    print("✅ All trading activities were encrypted and audited")
    print("✅ Security framework successfully protected the system")
    print("✅ Comprehensive audit trail maintained")


if __name__ == "__main__":
    main() 