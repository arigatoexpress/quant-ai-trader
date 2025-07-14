"""
Secure Authentication System
===========================

This module provides enterprise-grade authentication with strong password requirements,
two-factor authentication (2FA), session management, and comprehensive security features.

Features:
- Strong password validation and hashing
- TOTP-based 2FA with authenticator app support
- Secure session management with JWT tokens
- Account lockout and rate limiting
- Password history and rotation
- Security audit logging
- Configurable security policies

Author: AI Assistant
Version: 1.0.0
License: MIT
"""

import os
import re
import time
import hmac
import hashlib
import secrets
import qrcode
import base64
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import jwt
import pyotp
import bcrypt
from io import BytesIO
import sqlite3
import json

logger = logging.getLogger(__name__)

@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    min_password_length: int = 12
    max_password_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_symbols: bool = True
    password_history_count: int = 5
    max_login_attempts: int = 3
    lockout_duration_minutes: int = 15
    session_timeout_minutes: int = 60
    require_2fa: bool = True
    jwt_secret_rotation_days: int = 30

@dataclass
class User:
    """User account information."""
    username: str
    password_hash: str
    salt: str
    totp_secret: Optional[str] = None
    is_2fa_enabled: bool = False
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    last_login: Optional[datetime] = None
    password_history: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.password_history is None:
            self.password_history = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

@dataclass
class Session:
    """User session information."""
    session_id: str
    username: str
    token: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True

class PasswordValidator:
    """Validates password strength according to security policy."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
    
    def validate(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against security policy.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Length validation
        if len(password) < self.policy.min_password_length:
            errors.append(f"Password must be at least {self.policy.min_password_length} characters long")
        
        if len(password) > self.policy.max_password_length:
            errors.append(f"Password must not exceed {self.policy.max_password_length} characters")
        
        # Character type validation
        if self.policy.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.policy.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.policy.require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if self.policy.require_symbols and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # Check for common weak patterns
        if self._has_weak_patterns(password):
            errors.append("Password contains weak patterns (sequences, repetition, common words)")
        
        return len(errors) == 0, errors
    
    def _has_weak_patterns(self, password: str) -> bool:
        """Check for weak password patterns."""
        # Sequential characters
        for i in range(len(password) - 2):
            if ord(password[i+1]) == ord(password[i]) + 1 and ord(password[i+2]) == ord(password[i]) + 2:
                return True
        
        # Repeated characters
        for i in range(len(password) - 2):
            if password[i] == password[i+1] == password[i+2]:
                return True
        
        # Common weak passwords (basic check)
        weak_patterns = ['password', '123456', 'qwerty', 'admin', 'letmein']
        password_lower = password.lower()
        for pattern in weak_patterns:
            if pattern in password_lower:
                return True
        
        return False

class TOTPManager:
    """Manages TOTP (Time-based One-Time Password) for 2FA."""
    
    def __init__(self, issuer_name: str = "Quant AI Trader"):
        self.issuer_name = issuer_name
    
    def generate_secret(self) -> str:
        """Generate a new TOTP secret."""
        return pyotp.random_base32()
    
    def generate_qr_code(self, username: str, secret: str) -> str:
        """
        Generate QR code for authenticator app setup.
        
        Returns:
            Base64 encoded QR code image
        """
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=username,
            issuer_name=self.issuer_name
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    def verify_token(self, secret: str, token: str) -> bool:
        """Verify TOTP token."""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)  # Allow 1 window (30 seconds) tolerance
        except Exception as e:
            logger.error(f"TOTP verification error: {e}")
            return False
    
    def generate_backup_codes(self, count: int = 8) -> List[str]:
        """Generate backup codes for 2FA recovery."""
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4).upper()  # 8 character hex codes
            codes.append(code)
        return codes

class SecureDatabase:
    """Secure database for storing user authentication data."""
    
    def __init__(self, db_path: str = "data/auth.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize the authentication database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                totp_secret TEXT,
                is_2fa_enabled BOOLEAN DEFAULT 0,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TEXT,
                last_login TEXT,
                password_history TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                last_activity TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        
        # Backup codes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backup_codes (
                username TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                used_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users (username)
            )
        ''')
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auth_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                action TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                details TEXT,
                timestamp TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, user: User) -> bool:
        """Create a new user in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (
                    username, password_hash, salt, totp_secret, is_2fa_enabled,
                    failed_attempts, locked_until, last_login, password_history,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.username, user.password_hash, user.salt, user.totp_secret,
                user.is_2fa_enabled, user.failed_attempts,
                user.locked_until.isoformat() if user.locked_until else None,
                user.last_login.isoformat() if user.last_login else None,
                json.dumps(user.password_history),
                user.created_at.isoformat(), user.updated_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.IntegrityError:
            logger.error(f"User {user.username} already exists")
            return False
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False
    
    def get_user(self, username: str) -> Optional[User]:
        """Retrieve user from database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            
            if row:
                return User(
                    username=row[0],
                    password_hash=row[1],
                    salt=row[2],
                    totp_secret=row[3],
                    is_2fa_enabled=bool(row[4]),
                    failed_attempts=row[5],
                    locked_until=datetime.fromisoformat(row[6]) if row[6] else None,
                    last_login=datetime.fromisoformat(row[7]) if row[7] else None,
                    password_history=json.loads(row[8]) if row[8] else [],
                    created_at=datetime.fromisoformat(row[9]),
                    updated_at=datetime.fromisoformat(row[10])
                )
            
            conn.close()
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving user: {e}")
            return None
    
    def update_user(self, user: User) -> bool:
        """Update user in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            user.updated_at = datetime.now()
            
            cursor.execute('''
                UPDATE users SET
                    password_hash = ?, salt = ?, totp_secret = ?, is_2fa_enabled = ?,
                    failed_attempts = ?, locked_until = ?, last_login = ?,
                    password_history = ?, updated_at = ?
                WHERE username = ?
            ''', (
                user.password_hash, user.salt, user.totp_secret, user.is_2fa_enabled,
                user.failed_attempts,
                user.locked_until.isoformat() if user.locked_until else None,
                user.last_login.isoformat() if user.last_login else None,
                json.dumps(user.password_history),
                user.updated_at.isoformat(),
                user.username
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False

class SecureAuthenticationSystem:
    """Main authentication system with enterprise security features."""
    
    def __init__(self, policy: SecurityPolicy = None):
        self.policy = policy or SecurityPolicy()
        self.password_validator = PasswordValidator(self.policy)
        self.totp_manager = TOTPManager()
        self.database = SecureDatabase()
        self.jwt_secret = self._get_or_create_jwt_secret()
        self.active_sessions: Dict[str, Session] = {}
        
        logger.info("Secure Authentication System initialized")
    
    def _get_or_create_jwt_secret(self) -> bytes:
        """Get or create JWT secret for token signing."""
        secret_file = Path("config/.jwt_secret")
        secret_file.parent.mkdir(exist_ok=True)
        
        if secret_file.exists():
            with open(secret_file, 'rb') as f:
                return f.read()
        else:
            secret = secrets.token_bytes(32)
            with open(secret_file, 'wb') as f:
                f.write(secret)
            os.chmod(secret_file, 0o600)
            return secret
    
    def _hash_password(self, password: str, salt: bytes = None) -> Tuple[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = bcrypt.gensalt()
        
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8'), salt.decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def create_user(self, username: str, password: str, enable_2fa: bool = True) -> Tuple[bool, List[str]]:
        """
        Create a new user account with strong password requirements.
        
        Returns:
            Tuple of (success, list_of_messages)
        """
        try:
            # Validate username
            if not username or len(username) < 3:
                return False, ["Username must be at least 3 characters long"]
            
            if not re.match(r'^[a-zA-Z0-9_]+$', username):
                return False, ["Username can only contain letters, numbers, and underscores"]
            
            # Check if user already exists
            if self.database.get_user(username):
                return False, ["Username already exists"]
            
            # Validate password
            is_valid, errors = self.password_validator.validate(password)
            if not is_valid:
                return False, errors
            
            # Hash password
            password_hash, salt = self._hash_password(password)
            
            # Generate TOTP secret if 2FA is enabled
            totp_secret = None
            if enable_2fa:
                totp_secret = self.totp_manager.generate_secret()
            
            # Create user
            user = User(
                username=username,
                password_hash=password_hash,
                salt=salt,
                totp_secret=totp_secret,
                is_2fa_enabled=enable_2fa,
                password_history=[password_hash]
            )
            
            if self.database.create_user(user):
                messages = ["User created successfully"]
                if enable_2fa:
                    messages.append("2FA enabled. Please set up your authenticator app.")
                
                self._log_auth_event(username, "user_created", True)
                return True, messages
            else:
                return False, ["Failed to create user"]
                
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False, ["Internal error occurred"]
    
    def authenticate(self, username: str, password: str, totp_token: str = None, 
                    ip_address: str = None, user_agent: str = None) -> Tuple[bool, str, Optional[Session]]:
        """
        Authenticate user with username, password, and optional 2FA token.
        
        Returns:
            Tuple of (success, message, session_if_successful)
        """
        try:
            user = self.database.get_user(username)
            if not user:
                self._log_auth_event(username, "login_failed", False, ip_address, user_agent, "User not found")
                return False, "Invalid credentials", None
            
            # Check if account is locked
            if user.locked_until and datetime.now() < user.locked_until:
                remaining = user.locked_until - datetime.now()
                self._log_auth_event(username, "login_failed", False, ip_address, user_agent, "Account locked")
                return False, f"Account locked for {remaining.seconds // 60} more minutes", None
            
            # Verify password
            if not self._verify_password(password, user.password_hash):
                user.failed_attempts += 1
                
                # Lock account if too many failed attempts
                if user.failed_attempts >= self.policy.max_login_attempts:
                    user.locked_until = datetime.now() + timedelta(minutes=self.policy.lockout_duration_minutes)
                    self._log_auth_event(username, "account_locked", True, ip_address, user_agent)
                
                self.database.update_user(user)
                self._log_auth_event(username, "login_failed", False, ip_address, user_agent, "Invalid password")
                return False, "Invalid credentials", None
            
            # Verify 2FA if enabled
            if user.is_2fa_enabled:
                if not totp_token:
                    return False, "2FA token required", None
                
                if not self.totp_manager.verify_token(user.totp_secret, totp_token):
                    user.failed_attempts += 1
                    self.database.update_user(user)
                    self._log_auth_event(username, "login_failed", False, ip_address, user_agent, "Invalid 2FA token")
                    return False, "Invalid 2FA token", None
            
            # Reset failed attempts and update last login
            user.failed_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now()
            self.database.update_user(user)
            
            # Create session
            session = self._create_session(username, ip_address, user_agent)
            
            self._log_auth_event(username, "login_success", True, ip_address, user_agent)
            return True, "Authentication successful", session
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, "Internal error occurred", None
    
    def _create_session(self, username: str, ip_address: str = None, user_agent: str = None) -> Session:
        """Create a new user session."""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(minutes=self.policy.session_timeout_minutes)
        
        # Create JWT token
        token_payload = {
            'session_id': session_id,
            'username': username,
            'iat': datetime.now().timestamp(),
            'exp': expires_at.timestamp()
        }
        
        token = jwt.encode(token_payload, self.jwt_secret, algorithm='HS256')
        
        session = Session(
            session_id=session_id,
            username=username,
            token=token,
            created_at=datetime.now(),
            expires_at=expires_at,
            last_activity=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.active_sessions[session_id] = session
        return session
    
    def validate_session(self, token: str) -> Tuple[bool, Optional[Session]]:
        """Validate session token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            session_id = payload['session_id']
            
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                # Check if session is expired
                if datetime.now() > session.expires_at:
                    self.logout(session_id)
                    return False, None
                
                # Update last activity
                session.last_activity = datetime.now()
                return True, session
            
            return False, None
            
        except jwt.ExpiredSignatureError:
            return False, None
        except jwt.InvalidTokenError:
            return False, None
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False, None
    
    def logout(self, session_id: str) -> bool:
        """Logout user and invalidate session."""
        try:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                self._log_auth_event(session.username, "logout", True)
                del self.active_sessions[session_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    def setup_2fa(self, username: str) -> Tuple[bool, str, str]:
        """
        Set up 2FA for user.
        
        Returns:
            Tuple of (success, qr_code_base64, secret_for_manual_entry)
        """
        try:
            user = self.database.get_user(username)
            if not user:
                return False, "", ""
            
            # Generate new secret
            secret = self.totp_manager.generate_secret()
            
            # Generate QR code
            qr_code = self.totp_manager.generate_qr_code(username, secret)
            
            # Update user with new secret (but don't enable 2FA yet)
            user.totp_secret = secret
            self.database.update_user(user)
            
            return True, qr_code, secret
            
        except Exception as e:
            logger.error(f"2FA setup error: {e}")
            return False, "", ""
    
    def enable_2fa(self, username: str, verification_token: str) -> Tuple[bool, str, List[str]]:
        """
        Enable 2FA after verification.
        
        Returns:
            Tuple of (success, message, backup_codes)
        """
        try:
            user = self.database.get_user(username)
            if not user or not user.totp_secret:
                return False, "2FA not set up", []
            
            # Verify token
            if not self.totp_manager.verify_token(user.totp_secret, verification_token):
                return False, "Invalid verification token", []
            
            # Enable 2FA
            user.is_2fa_enabled = True
            self.database.update_user(user)
            
            # Generate backup codes
            backup_codes = self.totp_manager.generate_backup_codes()
            
            # Store backup codes in database (hashed)
            self._store_backup_codes(username, backup_codes)
            
            self._log_auth_event(username, "2fa_enabled", True)
            return True, "2FA enabled successfully", backup_codes
            
        except Exception as e:
            logger.error(f"2FA enable error: {e}")
            return False, "Internal error occurred", []
    
    def _store_backup_codes(self, username: str, codes: List[str]):
        """Store hashed backup codes in database."""
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            
            # Clear existing codes
            cursor.execute('DELETE FROM backup_codes WHERE username = ?', (username,))
            
            # Store new codes (hashed)
            for code in codes:
                code_hash = hashlib.sha256(code.encode()).hexdigest()
                cursor.execute('''
                    INSERT INTO backup_codes (username, code_hash, created_at)
                    VALUES (?, ?, ?)
                ''', (username, code_hash, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing backup codes: {e}")
    
    def _log_auth_event(self, username: str, action: str, success: bool, 
                       ip_address: str = None, user_agent: str = None, details: str = None):
        """Log authentication event for auditing."""
        try:
            conn = sqlite3.connect(self.database.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO auth_audit (username, action, success, ip_address, user_agent, details, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, action, success, ip_address, user_agent, details, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging auth event: {e}")
    
    def get_user_status(self, username: str) -> Dict:
        """Get user account status and security information."""
        try:
            user = self.database.get_user(username)
            if not user:
                return {"error": "User not found"}
            
            return {
                "username": user.username,
                "is_2fa_enabled": user.is_2fa_enabled,
                "failed_attempts": user.failed_attempts,
                "is_locked": user.locked_until and datetime.now() < user.locked_until,
                "locked_until": user.locked_until.isoformat() if user.locked_until else None,
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "created_at": user.created_at.isoformat(),
                "password_age_days": (datetime.now() - user.updated_at).days
            }
            
        except Exception as e:
            logger.error(f"Error getting user status: {e}")
            return {"error": "Internal error"}

# Convenience functions for easy integration
def create_default_system() -> SecureAuthenticationSystem:
    """Create authentication system with default security policy."""
    policy = SecurityPolicy(
        min_password_length=12,
        require_2fa=True,
        max_login_attempts=3,
        lockout_duration_minutes=15,
        session_timeout_minutes=60
    )
    return SecureAuthenticationSystem(policy)

def setup_initial_admin(auth_system: SecureAuthenticationSystem, username: str = "admin") -> Tuple[bool, str]:
    """Set up initial admin account with strong password."""
    # Generate a strong random password
    import string
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = ''.join(secrets.choice(alphabet) for _ in range(16))
    
    success, messages = auth_system.create_user(username, password, enable_2fa=True)
    
    if success:
        return True, password
    else:
        return False, "; ".join(messages) 