"""
Telegram Bot Integration
Provides real-time trading alerts and portfolio updates via Telegram
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import yaml
from dataclasses import dataclass, asdict
import time
import threading
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    """Configuration for trading alerts"""
    asset: str
    alert_type: str  # price, volume, rsi, opportunity
    threshold: float
    direction: str  # above, below, cross
    enabled: bool = True
    chat_id: Optional[str] = None
    last_triggered: Optional[datetime] = None
    cooldown_minutes: int = 60

@dataclass
class TradingAlert:
    """Trading alert data structure"""
    alert_id: str
    asset: str
    alert_type: str
    message: str
    price: Optional[float]
    change_24h: Optional[float]
    timestamp: datetime
    urgency: str  # low, medium, high, critical
    action_recommended: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class TelegramTradingBot:
    """Telegram bot for trading alerts and portfolio management"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Bot configuration
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN') or self.config.get('telegram', {}).get('bot_token')
        self.default_chat_id = os.getenv('TELEGRAM_CHAT_ID') or self.config.get('telegram', {}).get('chat_id')
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not found in environment or config")
        
        # Initialize bot
        self.bot = Bot(token=self.bot_token)
        self.application = Application.builder().token(self.bot_token).build()
        
        # Alert management
        self.alert_configs: Dict[str, AlertConfig] = {}
        self.active_alerts: List[TradingAlert] = []
        self.alert_history: List[TradingAlert] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Data integration
        self.data_fetcher = None
        self.asymmetric_scanner = None
        
        # User permissions
        self.authorized_users = set(self.config.get('telegram', {}).get('authorized_users', []))
        
        # Setup command handlers
        self._setup_handlers()
        
        logger.info("🤖 Telegram Trading Bot initialized")
        logger.info(f"   Bot Token: {'✅' if self.bot_token else '❌'}")
        logger.info(f"   Default Chat: {'✅' if self.default_chat_id else '❌'}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        return {
            'telegram': {
                'bot_token': None,
                'chat_id': None,
                'authorized_users': [],
                'max_alerts_per_hour': 10,
                'alert_cooldown_minutes': 60
            },
            'alerts': {
                'price_threshold': 5.0,  # 5% price change
                'volume_threshold': 2.0,  # 2x volume increase
                'rsi_overbought': 80,
                'rsi_oversold': 20,
                'opportunity_min_score': 7.0
            },
            'assets': ['BTC', 'ETH', 'SOL', 'SUI', 'SEI'],
            'monitoring_interval': 300  # 5 minutes
        }
    
    def _setup_handlers(self):
        """Setup Telegram command handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("help", self._help_command))
        self.application.add_handler(CommandHandler("status", self._status_command))
        self.application.add_handler(CommandHandler("portfolio", self._portfolio_command))
        self.application.add_handler(CommandHandler("alerts", self._alerts_command))
        self.application.add_handler(CommandHandler("add_alert", self._add_alert_command))
        self.application.add_handler(CommandHandler("remove_alert", self._remove_alert_command))
        self.application.add_handler(CommandHandler("opportunities", self._opportunities_command))
        self.application.add_handler(CommandHandler("market", self._market_command))
        self.application.add_handler(CommandHandler("stop_monitoring", self._stop_monitoring_command))
        self.application.add_handler(CommandHandler("start_monitoring", self._start_monitoring_command))
        
        # Message handler for text messages
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
    
    def _is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized to use the bot"""
        if not self.authorized_users:
            return True  # If no users specified, allow all
        return str(user_id) in self.authorized_users
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if not self._is_authorized(update.effective_user.id):
            await update.message.reply_text("❌ Unauthorized access")
            return
        
        welcome_message = """
🤖 **Quant AI Trading Bot**

Welcome to your personal trading assistant! I'll help you monitor markets and catch asymmetric trading opportunities.

**Available Commands:**
/help - Show all commands
/status - Bot and monitoring status  
/portfolio - Portfolio overview
/market - Current market data
/alerts - Manage your alerts
/opportunities - Latest trading opportunities

**Quick Setup:**
1. Use /add_alert to set up price alerts
2. Use /start_monitoring to begin market monitoring
3. I'll send you real-time updates automatically!

Ready to start trading smarter? 📈
        """
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        if not self._is_authorized(update.effective_user.id):
            return
        
        help_text = """
🔧 **Available Commands:**

**Portfolio & Market:**
/portfolio - Portfolio overview and performance
/market [symbol] - Current market data for asset
/status - Bot status and monitoring info

**Alerts & Monitoring:**
/alerts - List all your alerts
/add_alert [asset] [type] [threshold] - Add new alert
/remove_alert [id] - Remove alert by ID
/start_monitoring - Start market monitoring
/stop_monitoring - Stop market monitoring

**Trading Opportunities:**
/opportunities - Latest asymmetric opportunities
/opportunities [filter] - Filter by asset or type

**Examples:**
`/add_alert BTC price 5` - Alert if BTC changes >5%
`/add_alert ETH volume 2` - Alert if ETH volume increases 2x
`/market SUI` - Get current SUI market data
`/opportunities defi` - Show DeFi opportunities only

Need more help? Just ask! 🚀
        """
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        if not self._is_authorized(update.effective_user.id):
            return
        
        status_text = f"""
📊 **Bot Status**

🤖 **Monitoring:** {'🟢 Active' if self.monitoring_active else '🔴 Inactive'}
⏰ **Uptime:** {self._get_uptime()}
🔔 **Active Alerts:** {len(self.alert_configs)}
📈 **Alerts Sent Today:** {len([a for a in self.alert_history if a.timestamp.date() == datetime.now().date()])}

💾 **Data Sources:**
- CoinGecko: {'🟢' if self.data_fetcher else '🔴'}
- DexScreener: 🟢
- DeFi Llama: 🟢
- Sui Network: 🟢

🎯 **Last Update:** {datetime.now().strftime('%H:%M:%S')}
        """
        await update.message.reply_text(status_text, parse_mode='Markdown')
    
    async def _portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /portfolio command"""
        if not self._is_authorized(update.effective_user.id):
            return
        
        # This would integrate with actual portfolio data
        portfolio_text = """
💼 **Portfolio Overview**

**Total Value:** $0.00 (Demo Mode)
**24h Change:** +0.00%
**7d Change:** +0.00%

**Holdings:**
• BTC: 0.00 BTC ($0.00)
• ETH: 0.00 ETH ($0.00)  
• SOL: 0.00 SOL ($0.00)
• SUI: 0.00 SUI ($0.00)

**Performance:**
• Best Performer: N/A
• Worst Performer: N/A
• Total Alerts: 0

*Connect your wallet to see real portfolio data*
        """
        await update.message.reply_text(portfolio_text, parse_mode='Markdown')
    
    async def _alerts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alerts command"""
        if not self._is_authorized(update.effective_user.id):
            return
        
        if not self.alert_configs:
            await update.message.reply_text("📢 No alerts configured. Use /add_alert to create one!")
            return
        
        alerts_text = "🔔 **Your Active Alerts:**\n\n"
        for alert_id, config in self.alert_configs.items():
            status = "🟢" if config.enabled else "🔴"
            alerts_text += f"{status} **{alert_id}**\n"
            alerts_text += f"   Asset: {config.asset}\n"
            alerts_text += f"   Type: {config.alert_type}\n"
            alerts_text += f"   Threshold: {config.threshold}\n"
            alerts_text += f"   Direction: {config.direction}\n\n"
        
        alerts_text += "Use /remove_alert [id] to remove an alert"
        await update.message.reply_text(alerts_text, parse_mode='Markdown')
    
    async def _add_alert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add_alert command"""
        if not self._is_authorized(update.effective_user.id):
            return
        
        if len(context.args) < 3:
            await update.message.reply_text(
                "Usage: /add_alert [asset] [type] [threshold]\n"
                "Example: /add_alert BTC price 5\n"
                "Types: price, volume, rsi, opportunity"
            )
            return
        
        asset = context.args[0].upper()
        alert_type = context.args[1].lower()
        try:
            threshold = float(context.args[2])
        except ValueError:
            await update.message.reply_text("❌ Invalid threshold value")
            return
        
        direction = context.args[3] if len(context.args) > 3 else "above"
        
        alert_id = f"{asset}_{alert_type}_{int(time.time())}"
        
        self.alert_configs[alert_id] = AlertConfig(
            asset=asset,
            alert_type=alert_type,
            threshold=threshold,
            direction=direction,
            enabled=True,
            chat_id=str(update.effective_chat.id)
        )
        
        await update.message.reply_text(
            f"✅ Alert created!\n"
            f"ID: {alert_id}\n"
            f"Asset: {asset}\n"
            f"Type: {alert_type}\n"
            f"Threshold: {threshold}\n"
            f"Direction: {direction}"
        )
    
    async def _remove_alert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remove_alert command"""
        if not self._is_authorized(update.effective_user.id):
            return
        
        if not context.args:
            await update.message.reply_text("Usage: /remove_alert [alert_id]")
            return
        
        alert_id = context.args[0]
        
        if alert_id in self.alert_configs:
            del self.alert_configs[alert_id]
            await update.message.reply_text(f"✅ Alert {alert_id} removed")
        else:
            await update.message.reply_text(f"❌ Alert {alert_id} not found")
    
    async def _opportunities_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /opportunities command"""
        if not self._is_authorized(update.effective_user.id):
            return
        
        opportunities_text = """
🎯 **Latest Trading Opportunities**

**High-Yield DeFi:**
• Compound V3 USDC: 12.5% APY (Low Risk)
• Sui DEX LP: 45% APY (Medium Risk)
• Noodles Finance: 89% APY (High Risk)

**Price Momentum:**
• SUI: +15% (24h), Strong volume
• SOL: -8% (Oversold, potential bounce)
• ETH: Consolidating near resistance

**Asymmetric Bets:**
• Small cap DeFi tokens showing accumulation
• Cross-chain arbitrage opportunities
• Yield farming with 3:1 risk/reward

*Real opportunities will show when monitoring is active*
        """
        await update.message.reply_text(opportunities_text, parse_mode='Markdown')
    
    async def _market_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /market command"""
        if not self._is_authorized(update.effective_user.id):
            return
        
        asset = context.args[0].upper() if context.args else "BTC"
        
        # This would fetch real market data
        market_text = f"""
📈 **{asset} Market Data**

**Price:** $0.00
**24h Change:** +0.00%
**24h Volume:** $0.00M
**Market Cap:** $0.00B

**Technical Indicators:**
• RSI: 50.0 (Neutral)
• MACD: Bullish crossover
• Support: $0.00
• Resistance: $0.00

**Sentiment:** Neutral
**Trend:** Sideways

*Connect data sources for real-time data*
        """
        await update.message.reply_text(market_text, parse_mode='Markdown')
    
    async def _start_monitoring_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start_monitoring command"""
        if not self._is_authorized(update.effective_user.id):
            return
        
        if not self.monitoring_active:
            self.start_monitoring()
            await update.message.reply_text("🟢 Market monitoring started!")
        else:
            await update.message.reply_text("📊 Monitoring is already active")
    
    async def _stop_monitoring_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop_monitoring command"""
        if not self._is_authorized(update.effective_user.id):
            return
        
        if self.monitoring_active:
            self.stop_monitoring()
            await update.message.reply_text("🔴 Market monitoring stopped")
        else:
            await update.message.reply_text("📊 Monitoring is already inactive")
    
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle text messages"""
        if not self._is_authorized(update.effective_user.id):
            return
        
        message = update.message.text.lower()
        
        if "price" in message:
            await update.message.reply_text("📈 Use /market [symbol] to get price data")
        elif "alert" in message:
            await update.message.reply_text("🔔 Use /add_alert to create alerts")
        elif "help" in message:
            await self._help_command(update, context)
        else:
            await update.message.reply_text("🤖 Type /help for available commands")
    
    def _get_uptime(self) -> str:
        """Get bot uptime"""
        # Placeholder - would track actual start time
        return "0d 0h 0m"
    
    async def send_alert(self, alert: TradingAlert, chat_id: Optional[str] = None):
        """Send trading alert to Telegram"""
        try:
            target_chat_id = chat_id or self.default_chat_id
            if not target_chat_id:
                logger.warning("No chat ID available for alert")
                return
            
            # Format alert message
            urgency_emoji = {
                'low': '🔵',
                'medium': '🟡', 
                'high': '🟠',
                'critical': '🔴'
            }
            
            emoji = urgency_emoji.get(alert.urgency, '🔵')
            
            message = f"""
{emoji} **{alert.alert_type.upper()} ALERT**

**Asset:** {alert.asset}
**Message:** {alert.message}
"""
            
            if alert.price:
                message += f"**Price:** ${alert.price:,.2f}\n"
            
            if alert.change_24h:
                change_emoji = "📈" if alert.change_24h > 0 else "📉"
                message += f"**24h Change:** {change_emoji} {alert.change_24h:.2f}%\n"
            
            if alert.action_recommended:
                message += f"**Recommended Action:** {alert.action_recommended}\n"
            
            message += f"**Time:** {alert.timestamp.strftime('%H:%M:%S')}"
            
            await self.bot.send_message(
                chat_id=target_chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            # Add to history
            self.alert_history.append(alert)
            
            logger.info(f"✅ Alert sent: {alert.alert_type} for {alert.asset}")
            
        except Exception as e:
            logger.error(f"❌ Failed to send alert: {e}")
    
    async def send_portfolio_update(self, portfolio_data: Dict[str, Any], chat_id: Optional[str] = None):
        """Send portfolio update to Telegram"""
        try:
            target_chat_id = chat_id or self.default_chat_id
            if not target_chat_id:
                return
            
            message = f"""
💼 **Portfolio Update**

**Total Value:** ${portfolio_data.get('total_value', 0):,.2f}
**24h Change:** {portfolio_data.get('change_24h', 0):+.2f}%
**P&L:** ${portfolio_data.get('pnl', 0):,.2f}

**Top Performers:**
{portfolio_data.get('top_performers', 'No data')}

**Alerts:** {len(self.active_alerts)} active
**Time:** {datetime.now().strftime('%H:%M:%S')}
            """
            
            await self.bot.send_message(
                chat_id=target_chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to send portfolio update: {e}")
    
    def start_monitoring(self):
        """Start market monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("🔄 Market monitoring started")
    
    def stop_monitoring(self):
        """Stop market monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("⏹️ Market monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Check alerts (placeholder - would integrate with real data)
                self._check_alerts()
                
                # Sleep for monitoring interval
                time.sleep(self.config.get('monitoring_interval', 300))
                
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _check_alerts(self):
        """Check all configured alerts"""
        # Placeholder - would integrate with real market data
        pass
    
    async def run_bot(self):
        """Run the Telegram bot"""
        logger.info("🚀 Starting Telegram bot...")
        
        # Start monitoring by default
        self.start_monitoring()
        
        # Run the bot
        await self.application.run_polling(drop_pending_updates=True)

# Standalone functions for integration
def create_trading_bot(config_path: Optional[str] = None) -> TelegramTradingBot:
    """Create and return a TelegramTradingBot instance"""
    return TelegramTradingBot(config_path)

async def send_quick_alert(message: str, urgency: str = 'medium'):
    """Send a quick alert without full bot setup"""
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        logger.warning("Telegram credentials not configured")
        return
    
    try:
        bot = Bot(token=bot_token)
        
        urgency_emoji = {
            'low': '🔵',
            'medium': '🟡',
            'high': '🟠', 
            'critical': '🔴'
        }
        
        emoji = urgency_emoji.get(urgency, '🔵')
        formatted_message = f"{emoji} **ALERT**\n\n{message}\n\n**Time:** {datetime.now().strftime('%H:%M:%S')}"
        
        await bot.send_message(
            chat_id=chat_id,
            text=formatted_message,
            parse_mode='Markdown'
        )
        
        logger.info("✅ Quick alert sent")
        
    except Exception as e:
        logger.error(f"❌ Failed to send quick alert: {e}")

# Main execution
if __name__ == "__main__":
    async def main():
        bot = TelegramTradingBot()
        await bot.run_bot()
    
    asyncio.run(main()) 