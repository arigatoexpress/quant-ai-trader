"""
Cybersecurity Framework for Autonomous Trading
Provides comprehensive security measures for AI-powered trading systems
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import threading
import sqlite3
import uuid
from enum import Enum

# Configure secure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('secure_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ActionType(Enum):
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    TRADE_EXECUTION = "TRADE_EXECUTION"
    DATA_ACCESS = "DATA_ACCESS"
    SYSTEM_CONFIGURATION = "SYSTEM_CONFIGURATION"
    ALERT_GENERATION = "ALERT_GENERATION"

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    action_type: ActionType = ActionType.DATA_ACCESS
    user_id: str = "system"
    source_ip: str = "127.0.0.1"
    action_description: str = ""
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuthenticationToken:
    """Secure authentication token"""
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "autonomous_trader"
    issued_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=24))
    permissions: List[str] = field(default_factory=list)
    session_data: Dict[str, Any] = field(default_factory=dict)

class CryptoManager:
    """Secure encryption and decryption manager"""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.key = self._derive_key(master_key)
        else:
            self.key = Fernet.generate_key()
        self.cipher = Fernet(self.key)
        
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        password_bytes = password.encode('utf-8')
        salt = b'trading_salt_2024'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted = self.cipher.encrypt(data.encode('utf-8'))
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def create_secure_hash(self, data: str) -> str:
        """Create secure hash for data integrity"""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def verify_signature(self, data: str, signature: str, key: str) -> bool:
        """Verify HMAC signature"""
        expected_signature = hmac.new(
            key.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(signature, expected_signature)

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, db_path: str = "security_audit.db"):
        self.db_path = db_path
        self.crypto_manager = CryptoManager()
        self._init_database()
        
    def _init_database(self):
        """Initialize audit database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                action_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                source_ip TEXT NOT NULL,
                action_description TEXT NOT NULL,
                security_level TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                metadata TEXT NOT NULL,
                hash TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_security_event(self, event: SecurityEvent):
        """Log security event with integrity protection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create integrity hash
            event_data = f"{event.event_id}{event.timestamp}{event.action_type.value}{event.user_id}{event.action_description}"
            integrity_hash = self.crypto_manager.create_secure_hash(event_data)
            
            cursor.execute('''
                INSERT INTO security_events 
                (event_id, timestamp, action_type, user_id, source_ip, action_description, 
                 security_level, success, metadata, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.timestamp.isoformat(),
                event.action_type.value,
                event.user_id,
                event.source_ip,
                event.action_description,
                event.security_level.value,
                event.success,
                json.dumps(event.metadata),
                integrity_hash
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Security event logged: {event.action_type.value} - {event.action_description}")
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    def get_security_events(self, 
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          action_type: Optional[ActionType] = None,
                          user_id: Optional[str] = None) -> List[SecurityEvent]:
        """Retrieve security events with filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM security_events WHERE 1=1"
            params = []
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            if action_type:
                query += " AND action_type = ?"
                params.append(action_type.value)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            events = []
            for row in rows:
                event = SecurityEvent(
                    event_id=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    action_type=ActionType(row[3]),
                    user_id=row[4],
                    source_ip=row[5],
                    action_description=row[6],
                    security_level=SecurityLevel(row[7]),
                    success=bool(row[8]),
                    metadata=json.loads(row[9])
                )
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to retrieve security events: {e}")
            return []

class AuthenticationManager:
    """Secure authentication and authorization system"""
    
    def __init__(self, crypto_manager: CryptoManager):
        self.crypto_manager = crypto_manager
        self.active_tokens: Dict[str, AuthenticationToken] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 3
        self.lockout_duration = timedelta(minutes=15)
        
    def authenticate(self, user_id: str, password: str, permissions: List[str]) -> Optional[AuthenticationToken]:
        """Authenticate user and create secure token"""
        try:
            # Check for account lockout
            if self._is_account_locked(user_id):
                logger.warning(f"Authentication attempt on locked account: {user_id}")
                return None
            
            # Verify password (in production, use secure password hashing)
            if not self._verify_password(user_id, password):
                self._record_failed_attempt(user_id)
                logger.warning(f"Failed authentication attempt: {user_id}")
                return None
            
            # Create authentication token
            token = AuthenticationToken(
                user_id=user_id,
                permissions=permissions,
                session_data={"authenticated_at": datetime.now().isoformat()}
            )
            
            self.active_tokens[token.token_id] = token
            
            logger.info(f"User authenticated successfully: {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def authorize(self, token_id: str, required_permission: str) -> bool:
        """Check if token has required permission"""
        try:
            token = self.active_tokens.get(token_id)
            if not token:
                logger.warning(f"Authorization failed: Invalid token {token_id}")
                return False
            
            if token.expires_at < datetime.now():
                logger.warning(f"Authorization failed: Expired token {token_id}")
                self._revoke_token(token_id)
                return False
            
            if required_permission not in token.permissions:
                logger.warning(f"Authorization failed: Missing permission {required_permission}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            return False
    
    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if user_id not in self.failed_attempts:
            return False
        
        recent_attempts = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > datetime.now() - self.lockout_duration
        ]
        
        return len(recent_attempts) >= self.max_failed_attempts
    
    def _record_failed_attempt(self, user_id: str):
        """Record failed authentication attempt"""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        self.failed_attempts[user_id].append(datetime.now())
    
    def _verify_password(self, user_id: str, password: str) -> bool:
        """Verify password (simplified for demo)"""
        # In production, use secure password hashing
        return password == "secure_trading_password_2024"
    
    def _revoke_token(self, token_id: str):
        """Revoke authentication token"""
        if token_id in self.active_tokens:
            del self.active_tokens[token_id]
            logger.info(f"Token revoked: {token_id}")

class SecureKeyManager:
    """Secure management of API keys and secrets"""
    
    def __init__(self, crypto_manager: CryptoManager):
        self.crypto_manager = crypto_manager
        self.keys_db = "secure_keys.db"
        self._init_key_database()
    
    def _init_key_database(self):
        """Initialize secure key storage database"""
        conn = sqlite3.connect(self.keys_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key_name TEXT UNIQUE NOT NULL,
                encrypted_key TEXT NOT NULL,
                key_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used TEXT,
                usage_count INTEGER DEFAULT 0,
                max_usage INTEGER DEFAULT -1,
                expires_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_key(self, key_name: str, key_value: str, key_type: str = "api_key", 
                  max_usage: int = -1, expires_at: Optional[datetime] = None):
        """Store API key securely"""
        try:
            encrypted_key = self.crypto_manager.encrypt(key_value)
            
            conn = sqlite3.connect(self.keys_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO api_keys 
                (key_name, encrypted_key, key_type, created_at, max_usage, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                key_name,
                encrypted_key,
                key_type,
                datetime.now().isoformat(),
                max_usage,
                expires_at.isoformat() if expires_at else None
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"API key stored securely: {key_name}")
            
        except Exception as e:
            logger.error(f"Failed to store API key: {e}")
            raise
    
    def get_key(self, key_name: str) -> Optional[str]:
        """Retrieve and decrypt API key"""
        try:
            conn = sqlite3.connect(self.keys_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT encrypted_key, expires_at, usage_count, max_usage 
                FROM api_keys WHERE key_name = ?
            ''', (key_name,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"API key not found: {key_name}")
                return None
            
            encrypted_key, expires_at, usage_count, max_usage = row
            
            # Check expiration
            if expires_at and datetime.fromisoformat(expires_at) < datetime.now():
                logger.warning(f"API key expired: {key_name}")
                return None
            
            # Check usage limit
            if max_usage > 0 and usage_count >= max_usage:
                logger.warning(f"API key usage limit exceeded: {key_name}")
                return None
            
            # Update usage count
            cursor.execute('''
                UPDATE api_keys SET usage_count = usage_count + 1, last_used = ?
                WHERE key_name = ?
            ''', (datetime.now().isoformat(), key_name))
            
            conn.commit()
            conn.close()
            
            # Decrypt and return key
            decrypted_key = self.crypto_manager.decrypt(encrypted_key)
            return decrypted_key
            
        except Exception as e:
            logger.error(f"Failed to retrieve API key: {e}")
            return None

class SecureTradingFramework:
    """Main cybersecurity framework for autonomous trading"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.crypto_manager = CryptoManager(master_key)
        self.audit_logger = AuditLogger()
        self.auth_manager = AuthenticationManager(self.crypto_manager)
        self.key_manager = SecureKeyManager(self.crypto_manager)
        
        # Security monitoring
        self.security_alerts = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("Secure Trading Framework initialized")
    
    def initialize_security(self, grok_api_key: str, user_password: str) -> AuthenticationToken:
        """Initialize security framework with API keys and authentication"""
        try:
            # Store API key securely
            self.key_manager.store_key("grok_api_key", grok_api_key, "api_key")
            
            # Create authentication token for autonomous trading
            token = self.auth_manager.authenticate(
                user_id="autonomous_trader",
                password=user_password,
                permissions=[
                    "trade_execution",
                    "market_data_access",
                    "portfolio_management",
                    "risk_management",
                    "alert_generation"
                ]
            )
            
            if not token:
                raise Exception("Authentication failed")
            
            # Log security event
            self.audit_logger.log_security_event(SecurityEvent(
                action_type=ActionType.AUTHENTICATION,
                user_id="autonomous_trader",
                action_description="Autonomous trading system initialized",
                security_level=SecurityLevel.HIGH,
                success=True,
                metadata={"permissions": token.permissions}
            ))
            
            return token
            
        except Exception as e:
            logger.error(f"Security initialization failed: {e}")
            raise
    
    def secure_trade_execution(self, token_id: str, trade_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute trade with security controls"""
        try:
            # Verify authorization
            if not self.auth_manager.authorize(token_id, "trade_execution"):
                return False, "Unauthorized trade execution attempt"
            
            # Encrypt sensitive trade data
            encrypted_data = self.crypto_manager.encrypt(json.dumps(trade_data))
            
            # Log trade execution
            self.audit_logger.log_security_event(SecurityEvent(
                action_type=ActionType.TRADE_EXECUTION,
                user_id=self.auth_manager.active_tokens[token_id].user_id,
                action_description=f"Trade executed: {trade_data.get('action', 'UNKNOWN')} {trade_data.get('asset', 'UNKNOWN')}",
                security_level=SecurityLevel.CRITICAL,
                success=True,
                metadata={
                    "trade_id": trade_data.get("trade_id"),
                    "asset": trade_data.get("asset"),
                    "action": trade_data.get("action"),
                    "amount": trade_data.get("amount"),
                    "encrypted_data_hash": self.crypto_manager.create_secure_hash(encrypted_data)
                }
            ))
            
            return True, "Trade executed securely"
            
        except Exception as e:
            logger.error(f"Secure trade execution failed: {e}")
            return False, f"Trade execution error: {str(e)}"
    
    def get_secure_api_key(self, key_name: str, token_id: str) -> Optional[str]:
        """Retrieve API key with authorization check"""
        try:
            # Verify authorization
            if not self.auth_manager.authorize(token_id, "market_data_access"):
                logger.warning(f"Unauthorized API key access attempt: {key_name}")
                return None
            
            # Get API key
            api_key = self.key_manager.get_key(key_name)
            
            if api_key:
                # Log API key usage
                self.audit_logger.log_security_event(SecurityEvent(
                    action_type=ActionType.DATA_ACCESS,
                    user_id=self.auth_manager.active_tokens[token_id].user_id,
                    action_description=f"API key accessed: {key_name}",
                    security_level=SecurityLevel.MEDIUM,
                    success=True,
                    metadata={"key_name": key_name}
                ))
            
            return api_key
            
        except Exception as e:
            logger.error(f"API key retrieval failed: {e}")
            return None
    
    def start_security_monitoring(self):
        """Start continuous security monitoring"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._security_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Security monitoring started")
    
    def _security_monitoring_loop(self):
        """Continuous security monitoring loop"""
        while self.monitoring_active:
            try:
                # Monitor for suspicious activities
                recent_events = self.audit_logger.get_security_events(
                    start_time=datetime.now() - timedelta(minutes=5)
                )
                
                # Analyze for security threats
                self._analyze_security_events(recent_events)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Security monitoring error: {e}")
                time.sleep(60)
    
    def _analyze_security_events(self, events: List[SecurityEvent]):
        """Analyze security events for threats"""
        # Check for repeated failures
        failed_events = [e for e in events if not e.success]
        
        if len(failed_events) > 5:  # More than 5 failures in 5 minutes
            self._generate_security_alert(
                "High failure rate detected",
                SecurityLevel.HIGH,
                {"failed_events": len(failed_events)}
            )
    
    def _generate_security_alert(self, message: str, level: SecurityLevel, metadata: Dict[str, Any]):
        """Generate security alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": level.value,
            "metadata": metadata
        }
        
        self.security_alerts.append(alert)
        logger.warning(f"SECURITY ALERT: {message}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        return {
            "active_tokens": len(self.auth_manager.active_tokens),
            "monitoring_active": self.monitoring_active,
            "recent_alerts": len(self.security_alerts),
            "last_alert": self.security_alerts[-1] if self.security_alerts else None,
            "security_events_today": len(self.audit_logger.get_security_events(
                start_time=datetime.now() - timedelta(days=1)
            ))
        }


def main():
    """Demo the cybersecurity framework"""
    print("🔐 CYBERSECURITY FRAMEWORK DEMO")
    print("=" * 50)
    
    # Initialize framework
    framework = SecureTradingFramework()
    
    # Initialize security with API key
    token = framework.initialize_security(
        grok_api_key="demo_api_key_12345",
        user_password="secure_trading_password_2024"
    )
    
    print(f"✅ Security initialized. Token: {token.token_id[:8]}...")
    
    # Start security monitoring
    framework.start_security_monitoring()
    
    # Test secure trade execution
    trade_data = {
        "trade_id": "TRADE_001",
        "asset": "BTC",
        "action": "BUY",
        "amount": 0.1,
        "price": 118000
    }
    
    success, message = framework.secure_trade_execution(token.token_id, trade_data)
    print(f"🔒 Secure trade: {success} - {message}")
    
    # Test API key retrieval
    api_key = framework.get_secure_api_key("grok_api_key", token.token_id)
    print(f"🔑 API key retrieved: {'✅' if api_key else '❌'}")
    
    # Display security status
    status = framework.get_security_status()
    print(f"\n📊 Security Status: {status}")
    
    print("\n🎉 Cybersecurity framework demo completed!")


if __name__ == "__main__":
    main() 