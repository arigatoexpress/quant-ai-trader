"""
Secure Configuration Manager
Handles environment variables and sensitive data management
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from dotenv import load_dotenv
from cryptography.fernet import Fernet
import base64
from datetime import datetime

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class WalletConfig:
    """Wallet configuration with security"""
    name: str
    address: str
    chain: str
    is_active: bool = True
    last_updated: str = ""
    balance_cache: float = 0.0

@dataclass
class SecurityConfig:
    """Security configuration"""
    encryption_key: str
    api_keys: Dict[str, str]
    master_password: str
    session_timeout: int = 3600
    max_login_attempts: int = 3

class SecureConfigManager:
    """Secure configuration manager with environment variable support"""
    
    def __init__(self, config_file: str = "config/config.yaml"):
        self.config_file = config_file
        self.config_data = {}
        self.wallet_configs: List[WalletConfig] = []
        self.security_config: Optional[SecurityConfig] = None
        
        # Initialize encryption
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Load configuration
        self._load_configuration()
        self._load_wallet_addresses()
        
        print("🔐 Secure Configuration Manager initialized")
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key"""
        key_file = Path("config/.encryption_key")
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Create new key
            key = Fernet.generate_key()
            
            # Ensure config directory exists
            key_file.parent.mkdir(exist_ok=True)
            
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            
            return key
    
    def _load_configuration(self):
        """Load configuration from file and environment variables"""
        try:
            # Load from YAML file
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config_data = yaml.safe_load(f) or {}
            
            # Override with environment variables
            self._load_from_environment()
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config_data = {}
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # API Keys
        if os.getenv('GROK_API_KEY'):
            self.config_data['grok_api_key'] = os.getenv('GROK_API_KEY')
        
        if os.getenv('OPENAI_API_KEY'):
            self.config_data['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        
        if os.getenv('COINGECKO_API_KEY'):
            self.config_data['coingecko_api_key'] = os.getenv('COINGECKO_API_KEY')
        
        # Security settings
        if os.getenv('MASTER_PASSWORD'):
            self.config_data['master_password'] = os.getenv('MASTER_PASSWORD')
        
        if os.getenv('SESSION_TIMEOUT'):
            self.config_data['session_timeout'] = int(os.getenv('SESSION_TIMEOUT'))
        
        # Database settings
        if os.getenv('DATABASE_URL'):
            self.config_data['database_url'] = os.getenv('DATABASE_URL')
        
        # Trading settings
        if os.getenv('MAX_TRADE_AMOUNT'):
            self.config_data['max_trade_amount'] = float(os.getenv('MAX_TRADE_AMOUNT'))
        
        if os.getenv('RISK_TOLERANCE'):
            self.config_data['risk_tolerance'] = float(os.getenv('RISK_TOLERANCE'))
    
    def _load_wallet_addresses(self):
        """Load wallet addresses from environment variables"""
        self.wallet_configs = []
        
        # SUI Wallets
        sui_wallets = [
            'SUI_WALLET_1', 'SUI_WALLET_2', 'SUI_WALLET_3', 'SUI_WALLET_4', 'SUI_WALLET_5',
            'SUI_WALLET_6', 'SUI_WALLET_7', 'SUI_WALLET_8', 'SUI_WALLET_9', 'SUI_WALLET_10',
            'SUI_WALLET_11'
        ]
        
        for i, wallet_var in enumerate(sui_wallets, 1):
            address = os.getenv(wallet_var)
            if address:
                self.wallet_configs.append(WalletConfig(
                    name=f"SUI_Wallet_{i}",
                    address=address,
                    chain="SUI",
                    is_active=True
                ))
        
        # Solana Wallets
        solana_wallets = ['SOLANA_WALLET_1', 'SOLANA_WALLET_2']
        for i, wallet_var in enumerate(solana_wallets, 1):
            address = os.getenv(wallet_var)
            if address:
                self.wallet_configs.append(WalletConfig(
                    name=f"Solana_Wallet_{i}",
                    address=address,
                    chain="SOLANA",
                    is_active=True
                ))
        
        # Ethereum Wallets
        eth_address = os.getenv('ETHEREUM_WALLET_1')
        if eth_address:
            self.wallet_configs.append(WalletConfig(
                name="Ethereum_Wallet_1",
                address=eth_address,
                chain="ETHEREUM",
                is_active=True
            ))
        
        # Base Wallets
        base_address = os.getenv('BASE_WALLET_1')
        if base_address:
            self.wallet_configs.append(WalletConfig(
                name="Base_Wallet_1",
                address=base_address,
                chain="BASE",
                is_active=True
            ))
        
        # Sei Wallets
        sei_address = os.getenv('SEI_WALLET_1')
        if sei_address:
            self.wallet_configs.append(WalletConfig(
                name="Sei_Wallet_1",
                address=sei_address,
                chain="SEI",
                is_active=True
            ))
        
        print(f"📱 Loaded {len(self.wallet_configs)} wallet configurations")
    
    def get_wallet_addresses(self, chain: Optional[str] = None) -> List[WalletConfig]:
        """Get wallet addresses by chain"""
        if chain:
            return [w for w in self.wallet_configs if w.chain == chain and w.is_active]
        return [w for w in self.wallet_configs if w.is_active]
    
    def get_sui_wallets(self) -> List[str]:
        """Get SUI wallet addresses"""
        return [w.address for w in self.wallet_configs if w.chain == "SUI" and w.is_active]
    
    def get_solana_wallets(self) -> List[str]:
        """Get Solana wallet addresses"""
        return [w.address for w in self.wallet_configs if w.chain == "SOLANA" and w.is_active]
    
    def get_ethereum_wallets(self) -> List[str]:
        """Get Ethereum wallet addresses"""
        return [w.address for w in self.wallet_configs if w.chain == "ETHEREUM" and w.is_active]
    
    def get_base_wallets(self) -> List[str]:
        """Get Base wallet addresses"""
        return [w.address for w in self.wallet_configs if w.chain == "BASE" and w.is_active]
    
    def get_sei_wallets(self) -> List[str]:
        """Get Sei wallet addresses"""
        return [w.address for w in self.wallet_configs if w.chain == "SEI" and w.is_active]
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key securely"""
        return self.config_data.get(key_name)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        encrypted = self.cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode()
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        return {
            'max_trade_amount': self.config_data.get('max_trade_amount', 1000.0),
            'risk_tolerance': self.config_data.get('risk_tolerance', 0.02),
            'confidence_threshold': self.config_data.get('confidence_threshold', 0.7),
            'max_daily_trades': self.config_data.get('max_daily_trades', 10),
            'emergency_stop_loss': self.config_data.get('emergency_stop_loss', 0.05),
            'position_size_limit': self.config_data.get('position_size_limit', 0.1),
            'stop_loss_pct': self.config_data.get('stop_loss_pct', 0.02),
            'take_profit_pct': self.config_data.get('take_profit_pct', 0.04)
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            'master_password': self.config_data.get('master_password', 'secure_trading_password_2024'),
            'session_timeout': self.config_data.get('session_timeout', 3600),
            'max_login_attempts': self.config_data.get('max_login_attempts', 3),
            'encryption_enabled': True,
            'audit_logging': True,
            'security_monitoring': True,
            'threat_detection': True
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            'url': self.config_data.get('database_url', 'sqlite:///trading_data.db'),
            'echo': self.config_data.get('database_echo', False),
            'pool_size': self.config_data.get('database_pool_size', 5),
            'max_overflow': self.config_data.get('database_max_overflow', 10)
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'wallet_count': len(self.wallet_configs),
            'api_keys_configured': 0,
            'security_level': 'HIGH'
        }
        
        # Check API keys
        required_keys = ['grok_api_key']
        for key in required_keys:
            if self.get_api_key(key):
                validation_results['api_keys_configured'] += 1
            else:
                validation_results['errors'].append(f"Missing required API key: {key}")
                validation_results['valid'] = False
        
        # Check wallet addresses
        if len(self.wallet_configs) == 0:
            validation_results['warnings'].append("No wallet addresses configured")
            validation_results['security_level'] = 'MEDIUM'
        
        # Check security settings
        if not self.config_data.get('master_password'):
            validation_results['warnings'].append("Master password not configured")
            validation_results['security_level'] = 'LOW'
        
        return validation_results
    
    def create_env_template(self) -> str:
        """Create environment variable template"""
        template = """# Secure Trading System Environment Variables
# Copy this file to .env and fill in your values

# =============================================================================
# API KEYS (Required)
# =============================================================================
GROK_API_KEY=your_grok_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
COINGECKO_API_KEY=your_coingecko_api_key_here

# =============================================================================
# WALLET ADDRESSES (Update with your actual addresses)
# =============================================================================

# SUI Wallets (11 wallets)
SUI_WALLET_1=your_sui_wallet_address_1
SUI_WALLET_2=your_sui_wallet_address_2
SUI_WALLET_3=your_sui_wallet_address_3
SUI_WALLET_4=your_sui_wallet_address_4
SUI_WALLET_5=your_sui_wallet_address_5
SUI_WALLET_6=your_sui_wallet_address_6
SUI_WALLET_7=your_sui_wallet_address_7
SUI_WALLET_8=your_sui_wallet_address_8
SUI_WALLET_9=your_sui_wallet_address_9
SUI_WALLET_10=your_sui_wallet_address_10
SUI_WALLET_11=your_sui_wallet_address_11

# Solana Wallets (2 wallets)
SOLANA_WALLET_1=your_solana_wallet_address_1
SOLANA_WALLET_2=your_solana_wallet_address_2

# Ethereum Wallets (1 wallet)
ETHEREUM_WALLET_1=your_ethereum_wallet_address_1

# Base Wallets (1 wallet)
BASE_WALLET_1=your_base_wallet_address_1

# Sei Wallets (1 wallet)
SEI_WALLET_1=your_sei_wallet_address_1

# =============================================================================
# SECURITY SETTINGS
# =============================================================================
"YOUR_PASSWORD_HERE"
SESSION_TIMEOUT=3600
MAX_LOGIN_ATTEMPTS=3

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================
MAX_TRADE_AMOUNT=1000.0
RISK_TOLERANCE=0.02
CONFIDENCE_THRESHOLD=0.7
MAX_DAILY_TRADES=10
EMERGENCY_STOP_LOSS=0.05
POSITION_SIZE_LIMIT=0.1
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DATABASE_URL=sqlite:///trading_data.db
DATABASE_ECHO=false
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================
LOG_LEVEL=INFO
DEBUG_MODE=false
PERFORMANCE_MONITORING=true
SECURITY_MONITORING=true
AUDIT_LOGGING=true
BACKUP_ENABLED=true
BACKUP_INTERVAL=3600

# =============================================================================
# NOTIFICATION SETTINGS
# =============================================================================
ENABLE_NOTIFICATIONS=true
NOTIFICATION_WEBHOOK=your_webhook_url_here
EMAIL_NOTIFICATIONS=false
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
"YOUR_PASSWORD_HERE"
"""
        
        # Write template to file
        with open('.env.template', 'w') as f:
            f.write(template)
        
        return template
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        validation = self.validate_configuration()
        
        summary = {
            'configuration_status': 'VALID' if validation['valid'] else 'INVALID',
            'security_level': validation['security_level'],
            'wallet_addresses': {
                'total': len(self.wallet_configs),
                'by_chain': {
                    'SUI': len(self.get_sui_wallets()),
                    'SOLANA': len(self.get_solana_wallets()),
                    'ETHEREUM': len(self.get_ethereum_wallets()),
                    'BASE': len(self.get_base_wallets()),
                    'SEI': len(self.get_sei_wallets())
                }
            },
            'api_keys_configured': validation['api_keys_configured'],
            'security_features': {
                'encryption': True,
                'audit_logging': True,
                'session_management': True,
                'secure_storage': True
            },
            'warnings': validation['warnings'],
            'errors': validation['errors']
        }
        
        return summary
    
    def update_wallet_balance(self, wallet_name: str, balance: float):
        """Update wallet balance cache"""
        for wallet in self.wallet_configs:
            if wallet.name == wallet_name:
                wallet.balance_cache = balance
                wallet.last_updated = datetime.now().isoformat()
                break
    
    def get_total_portfolio_value(self) -> float:
        """Get total portfolio value from cached balances"""
        return sum(wallet.balance_cache for wallet in self.wallet_configs)


def main():
    """Demo the secure configuration manager"""
    print("🔐 SECURE CONFIGURATION MANAGER DEMO")
    print("=" * 60)
    
    # Initialize manager
    config_manager = SecureConfigManager()
    
    # Create environment template
    print("📝 Creating environment variable template...")
    template = config_manager.create_env_template()
    print("✅ Template created: .env.template")
    
    # Get configuration summary
    print("\n📊 Configuration Summary:")
    summary = config_manager.get_configuration_summary()
    
    print(f"   Status: {summary['configuration_status']}")
    print(f"   Security Level: {summary['security_level']}")
    print(f"   Total Wallets: {summary['wallet_addresses']['total']}")
    print(f"   API Keys Configured: {summary['api_keys_configured']}")
    
    print("\n🔗 Wallet Distribution:")
    for chain, count in summary['wallet_addresses']['by_chain'].items():
        print(f"   {chain}: {count} wallets")
    
    if summary['warnings']:
        print("\n⚠️  Warnings:")
        for warning in summary['warnings']:
            print(f"   • {warning}")
    
    if summary['errors']:
        print("\n❌ Errors:")
        for error in summary['errors']:
            print(f"   • {error}")
    
    # Test encryption
    print("\n🔒 Testing Encryption:")
    test_data = "Sensitive wallet data: 0x1234567890abcdef"
    encrypted = config_manager.encrypt_sensitive_data(test_data)
    decrypted = config_manager.decrypt_sensitive_data(encrypted)
    
    print(f"   Original: {test_data}")
    print(f"   Encrypted: {encrypted[:50]}...")
    print(f"   Decrypted: {decrypted}")
    print(f"   Success: {'✅' if test_data == decrypted else '❌'}")
    
    print("\n🎉 Configuration manager demo completed!")
    print("💡 Next steps:")
    print("   1. Copy .env.template to .env")
    print("   2. Fill in your actual wallet addresses and API keys")
    print("   3. Set secure passwords and configuration values")
    print("   4. Run the application with secure configuration")


if __name__ == "__main__":
    main() 