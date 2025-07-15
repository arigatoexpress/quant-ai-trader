"""
Security Audit and Cleanup System
Scans for exposed secrets, validates code security, and ensures safe GitHub commits
"""

import os
import re
import json
import hashlib
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
from datetime import datetime
import subprocess
import tempfile
import shutil
from pathlib import Path
import yaml
import base64
import secrets
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityIssue:
    """Represents a security issue found during audit"""
    severity: str  # HIGH, MEDIUM, LOW
    category: str  # SECRET, VULNERABILITY, CONFIGURATION, DEPENDENCY
    file_path: str
    line_number: int
    description: str
    recommendation: str
    code_snippet: str
    timestamp: datetime

@dataclass
class SecretPattern:
    """Pattern for detecting secrets in code"""
    name: str
    pattern: str
    description: str
    severity: str
    examples: List[str]

class SecretScanner:
    """Scans for exposed secrets and sensitive information"""
    
    def __init__(self):
        self.secret_patterns = self._initialize_secret_patterns()
        self.ignored_files = {
            '.git', '.gitignore', '.env.example', 'README.md', 
            'requirements.txt', '*.log', '*.db', '__pycache__'
        }
        self.ignored_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe'}
        
    def _initialize_secret_patterns(self) -> List[SecretPattern]:
        """Initialize patterns for detecting secrets"""
        patterns = [
            SecretPattern(
                name="API Keys",
                pattern=r'(?i)(api[_-]?key|apikey|access[_-]?key|secret[_-]?key)\s*[:=]\s*["\']?([a-zA-Z0-9]{20,})["\']?',
                description="API keys and access tokens",
                severity="HIGH",
                examples=["api_key: SECRET_REMOVED"]
            ),
            SecretPattern(
                name="Private Keys",
                pattern=r'(?i)(private[_-]?key|secret[_-]?key|privatekey)\s*[:=]\s*["\']?([a-zA-Z0-9]{40,})["\']?',
                description="Private keys and cryptographic secrets",
                severity="CRITICAL",
                examples=["private_key: SECRET_REMOVED"]
            ),
            SecretPattern(
                name="Database Credentials",
                pattern=r'(?i)(database[_-]?url|db[_-]?url|connection[_-]?string)\s*[:=]\s*["\']?([a-zA-Z0-9://@._-]+)["\']?',
                description="Database connection strings and credentials",
                severity="HIGH",
                examples=["database_url: postgresql://user:password@localhost:5432/dbname"]
            ),
            SecretPattern(
                name="Passwords",
                pattern=r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']?([a-zA-Z0-9@#$%^&*()_+-=]{8,})["\']?',
                description="Plain text passwords",
                severity="HIGH",
                examples=["password: secure_password"]
            ),
            SecretPattern(
                name="JWT Tokens",
                pattern=r'(?i)(jwt[_-]?token|bearer[_-]?token)\s*[:=]\s*["\']?([a-zA-Z0-9._-]{50,})["\']?',
                description="JWT tokens and bearer tokens",
                severity="HIGH",
                examples=["jwt_token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."]
            ),
            SecretPattern(
                name="AWS Credentials",
                pattern=r'(?i)(aws[_-]?access[_-]?key[_-]?id|aws[_-]?secret[_-]?access[_-]?key)\s*[:=]\s*["\']?([A-Z0-9]{20,})["\']?',
                description="AWS access keys and credentials",
                severity="CRITICAL",
                examples=["aws_access_key_id: SECRET_REMOVED"]
            ),
            SecretPattern(
                name="SSH Keys",
                pattern=r'(?i)(ssh[_-]?private[_-]?key|rsa[_-]?private[_-]?key)\s*[:=]\s*["\']?-----BEGIN[^-]+-----',
                description="SSH private keys",
                severity="CRITICAL",
                examples=["ssh_private_key: -----BEGIN RSA PRIVATE KEY-----"]
            ),
            SecretPattern(
                name="OAuth Tokens",
                pattern=r'(?i)(oauth[_-]?token|access[_-]?token)\s*[:=]\s*["\']?([a-zA-Z0-9._-]{30,})["\']?',
                description="OAuth tokens and access tokens",
                severity="HIGH",
                examples=["oauth_token: 1/fFAGRNJru1FTz70BzhT3Zg"]
            ),
            SecretPattern(
                name="Encryption Keys",
                pattern=r'(?i)(encryption[_-]?key|cipher[_-]?key|aes[_-]?key)\s*[:=]\s*["\']?([a-zA-Z0-9]{32,})["\']?',
                description="Encryption keys and cipher keys",
                severity="CRITICAL",
                examples=["encryption_key: SECRET_REMOVED"]
            ),
            SecretPattern(
                name="Webhook URLs",
                pattern=r'(?i)(webhook[_-]?url|callback[_-]?url)\s*[:=]\s*["\']?https?://[^\s"\']+["\']?',
                description="Webhook URLs that might contain tokens",
                severity="MEDIUM",
                examples=["webhook_url: https://api.github.com/webhooks/1234567890abcdef"]
            )
        ]
        return patterns
    
    def scan_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan a single file for secrets"""
        issues = []
        
        # Check if file should be ignored
        if self._should_ignore_file(file_path):
            return issues
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.secret_patterns:
                        matches = re.finditer(pattern.pattern, line)
                        for match in matches:
                            issue = SecurityIssue(
                                severity=pattern.severity,
                                category="SECRET",
                                file_path=file_path,
                                line_number=line_num,
                                description=f"Potential {pattern.name} found",
                                recommendation=f"Remove or encrypt the {pattern.name.lower()}",
                                code_snippet=line.strip(),
                                timestamp=datetime.now()
                            )
                            issues.append(issue)
                            
        except Exception as e:
            logger.warning(f"Error scanning file {file_path}: {e}")
        
        return issues
    
    def scan_directory(self, directory_path: str) -> List[SecurityIssue]:
        """Scan entire directory for secrets"""
        all_issues = []
        
        for root, dirs, files in os.walk(directory_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignored_files]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip ignored files
                if self._should_ignore_file(file_path):
                    continue
                
                issues = self.scan_file(file_path)
                all_issues.extend(issues)
        
        return all_issues
    
    def _should_ignore_file(self, file_path: str) -> bool:
        """Check if file should be ignored during scanning"""
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1]
        
        # Check ignored files
        if file_name in self.ignored_files:
            return True
        
        # Check ignored extensions
        if file_ext in self.ignored_extensions:
            return True
        
        # Check if file is in ignored directories
        for ignored_dir in self.ignored_files:
            if ignored_dir in file_path:
                return True
        
        return False

class VulnerabilityScanner:
    """Scans for common security vulnerabilities in code"""
    
    def __init__(self):
        self.vulnerability_patterns = self._initialize_vulnerability_patterns()
        
    def _initialize_vulnerability_patterns(self) -> List[Dict[str, Any]]:
        """Initialize patterns for detecting vulnerabilities"""
        patterns = [
            {
                "name": "SQL Injection",
                "pattern": r'(?i)(execute|executemany|query)\s*\(\s*["\']?\s*\$\{.*\}\s*["\']?\s*\)',
                "description": "Potential SQL injection vulnerability",
                "severity": "HIGH",
                "examples": ["cursor.execute(f\"SELECT * FROM users WHERE id = {user_id}\")"]
            },
            {
                "name": "Command Injection",
                "pattern": r'(?i)(subprocess\.|os\.system|eval\(|exec\()',
                "description": "Potential command injection vulnerability",
                "severity": "HIGH",
                "examples": ["os.system(f\"rm {user_input}\")", "eval(user_input)"]
            },
            {
                "name": "Hardcoded Credentials",
                "pattern": r'(?i)(username|user|login|email)\s*[:=]\s*["\']?[^"\']+["\']?\s*[,;]?\s*(password|passwd|pwd)\s*[:=]\s*["\']?[^"\']+["\']?',
                "description": "Hardcoded credentials in code",
                "severity": "HIGH",
                "examples": ["username = \"admin\", password = \"secret123\""]
            },
            {
                "name": "Insecure Random",
                "pattern": r'(?i)(random\.|math\.random)',
                "description": "Use of insecure random number generation",
                "severity": "MEDIUM",
                "examples": ["random.randint(1, 100)", "math.random()"]
            },
            {
                "name": "Debug Code",
                "pattern": r'(?i)(print\s*\(|console\.log|debugger|breakpoint)',
                "description": "Debug code left in production",
                "severity": "LOW",
                "examples": ["print(\"DEBUG: user_id = \", user_id)", "debugger;"]
            },
            {
                "name": "Insecure File Operations",
                "pattern": r'(?i)(open\s*\([^)]*\)|file\s*\([^)]*\))',
                "description": "Potential insecure file operations",
                "severity": "MEDIUM",
                "examples": ["open(user_input, 'r')"]
            },
            {
                "name": "Weak Cryptography",
                "pattern": r'(?i)(md5|sha1)\s*\(',
                "description": "Use of weak cryptographic algorithms",
                "severity": "HIGH",
                "examples": ["hashlib.md5(password.encode())", "hashlib.sha1(data)"]
            }
        ]
        return patterns
    
    def scan_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan a single file for vulnerabilities"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.vulnerability_patterns:
                        matches = re.finditer(pattern["pattern"], line)
                        for match in matches:
                            issue = SecurityIssue(
                                severity=pattern["severity"],
                                category="VULNERABILITY",
                                file_path=file_path,
                                line_number=line_num,
                                description=pattern["description"],
                                recommendation=f"Review and fix the {pattern['name'].lower()} vulnerability",
                                code_snippet=line.strip(),
                                timestamp=datetime.now()
                            )
                            issues.append(issue)
                            
        except Exception as e:
            logger.warning(f"Error scanning file {file_path} for vulnerabilities: {e}")
        
        return issues

class ConfigurationValidator:
    """Validates configuration files for security issues"""
    
    def __init__(self):
        self.required_env_vars = [
            'DATABASE_URL', 'API_KEY', 'SECRET_KEY', 'JWT_SECRET'
        ]
        self.sensitive_config_keys = [
            'password', 'secret', 'key', 'token', 'credential'
        ]
    
    def validate_config_file(self, file_path: str) -> List[SecurityIssue]:
        """Validate a configuration file for security issues"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Check for hardcoded secrets
                for key in self.sensitive_config_keys:
                    pattern = rf'(?i){key}\s*[:=]\s*["\']?[^"\']+["\']?'
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        issue = SecurityIssue(
                            severity="HIGH",
                            category="CONFIGURATION",
                            file_path=file_path,
                            line_number=0,
                            description=f"Hardcoded {key} in configuration",
                            recommendation=f"Use environment variables for {key}",
                            code_snippet=match.group(),
                            timestamp=datetime.now()
                        )
                        issues.append(issue)
                
                # Check for missing required environment variables
                if 'config.yaml' in file_path or 'config.yml' in file_path:
                    try:
                        config = yaml.safe_load(content)
                        for env_var in self.required_env_vars:
                            if not self._check_env_var_usage(content, env_var):
                                issue = SecurityIssue(
                                    severity="MEDIUM",
                                    category="CONFIGURATION",
                                    file_path=file_path,
                                    line_number=0,
                                    description=f"Missing environment variable: {env_var}",
                                    recommendation=f"Use {env_var} environment variable",
                                    code_snippet="",
                                    timestamp=datetime.now()
                                )
                                issues.append(issue)
                    except yaml.YAMLError:
                        pass
                        
        except Exception as e:
            logger.warning(f"Error validating config file {file_path}: {e}")
        
        return issues
    
    def _check_env_var_usage(self, content: str, env_var: str) -> bool:
        """Check if environment variable is properly used"""
        patterns = [
            rf'os\.environ\[["\']{env_var}["\']\]',
            rf'os\.getenv\(["\']{env_var}["\']\)',
            rf'\$\{{{env_var}\}}',
            rf'\${env_var}'
        ]
        
        for pattern in patterns:
            if re.search(pattern, content):
                return True
        
        return False

class DependencyScanner:
    """Scans for vulnerable dependencies"""
    
    def __init__(self):
        self.known_vulnerabilities = {
            'requests': ['2.25.0', '2.26.0'],
            'urllib3': ['1.26.0', '1.26.1'],
            'cryptography': ['3.3.0', '3.3.1']
        }
    
    def scan_requirements_file(self, file_path: str) -> List[SecurityIssue]:
        """Scan requirements.txt for vulnerable dependencies"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse package and version
                        parts = line.split('==')
                        if len(parts) == 2:
                            package = parts[0].strip()
                            version = parts[1].strip()
                            
                            if package in self.known_vulnerabilities:
                                vulnerable_versions = self.known_vulnerabilities[package]
                                if version in vulnerable_versions:
                                    issue = SecurityIssue(
                                        severity="HIGH",
                                        category="DEPENDENCY",
                                        file_path=file_path,
                                        line_number=line_num,
                                        description=f"Vulnerable dependency: {package} {version}",
                                        recommendation=f"Update {package} to a secure version",
                                        code_snippet=line,
                                        timestamp=datetime.now()
                                    )
                                    issues.append(issue)
                                    
        except Exception as e:
            logger.warning(f"Error scanning requirements file {file_path}: {e}")
        
        return issues

class SecurityCleanup:
    """Performs security cleanup operations"""
    
    def __init__(self):
        self.backup_dir: Optional[str] = None
        
    def create_backup(self, directory_path: str) -> Optional[str]:
        """Create a backup of the directory before cleanup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = os.path.join(os.path.dirname(directory_path), backup_name)
        
        try:
            shutil.copytree(directory_path, backup_path)
            self.backup_dir = backup_path
            logger.info(f"Created backup at: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def remove_secrets(self, issues: List[SecurityIssue]) -> List[str]:
        """Remove or replace secrets found in the codebase"""
        modified_files = []
        
        for issue in issues:
            if issue.category == "SECRET" and issue.severity in ["HIGH", "CRITICAL"]:
                try:
                    # Read file
                    with open(issue.file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    lines = content.split('\n')
                    
                    # Replace the problematic line
                    if 0 < issue.line_number <= len(lines):
                        original_line = lines[issue.line_number - 1]
                        
                        # Replace with placeholder
                        if 'api_key' in original_line.lower() or 'apikey' in original_line.lower():
                            match = re.search(r'["\']?[a-zA-Z0-9]{20,}["\']?', original_line)
                            if match:
                                new_line = original_line.replace(match.group(), '"YOUR_API_KEY_HERE"')
                            else:
                                new_line = original_line # Or handle error appropriately
                        elif 'password' in original_line.lower():
                            match = re.search(r'["\']?[^"\']+["\']?', original_line)
                            if match:
                                new_line = original_line.replace(match.group(), os.getenv("AUDIT_PASSWORD", "audit_password"))
                            else:
                                new_line = original_line
                        else:
                            match = re.search(r'["\']?[a-zA-Z0-9]{20,}["\']?', original_line)
                            if match:
                                new_line = original_line.replace(match.group(), '"SECRET_REMOVED"')
                            else:
                                new_line = original_line
                        
                        lines[issue.line_number - 1] = new_line
                        
                        # Write back to file
                        with open(issue.file_path, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(lines))
                        
                        modified_files.append(issue.file_path)
                        logger.info(f"Removed secret from {issue.file_path}:{issue.line_number}")
                        
                except Exception as e:
                    logger.warning(f"Failed to remove secret from {issue.file_path}: {e}")
        
        return modified_files
    
    def create_env_template(self, issues: List[SecurityIssue]) -> str:
        """Create .env.template file with placeholders"""
        template_content = "# Environment Variables Template\n"
        template_content += "# Copy this file to .env and fill in your actual values\n\n"
        
        # Extract unique environment variables from issues
        env_vars = set()
        for issue in issues:
            if issue.category == "SECRET":
                # Extract potential env var names
                matches = re.findall(r'(?i)(api[_-]?key|password|secret[_-]?key|token)', issue.code_snippet)
                for match in matches:
                    env_vars.add(match.upper())
        
        for env_var in sorted(env_vars):
            template_content += f"{env_var}=your_{env_var.lower()}_here\n"
        
        template_path = ".env.template"
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        logger.info(f"Created environment template: {template_path}")
        return template_path
    
    def update_gitignore(self) -> bool:
        """Update .gitignore to exclude sensitive files"""
        gitignore_path = ".gitignore"
        sensitive_patterns = [
            "# Security - Sensitive Files",
            ".env",
            ".env.local",
            ".env.production",
            "*.key",
            "*.pem",
            "*.p12",
            "*.pfx",
            "secrets/",
            "keys/",
            "credentials/",
            "*.log",
            "*.db",
            "*.sqlite",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            ".pytest_cache/",
            ".coverage",
            "htmlcov/",
            ".DS_Store",
            "Thumbs.db"
        ]
        
        try:
            # Read existing .gitignore
            existing_content = ""
            if os.path.exists(gitignore_path):
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
            
            # Add new patterns if they don't exist
            new_patterns = []
            for pattern in sensitive_patterns:
                if pattern not in existing_content:
                    new_patterns.append(pattern)
            
            if new_patterns:
                with open(gitignore_path, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(new_patterns) + '\n')
                logger.info(f"Updated .gitignore with {len(new_patterns)} new patterns")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update .gitignore: {e}")
            return False

class SecurityAuditor:
    """Main security auditor that coordinates all scanning and cleanup"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.secret_scanner = SecretScanner()
        self.vulnerability_scanner = VulnerabilityScanner()
        self.config_validator = ConfigurationValidator()
        self.dependency_scanner = DependencyScanner()
        self.cleanup = SecurityCleanup()
        
        self.all_issues = []
        self.scan_results = {}
        
    async def run_full_audit(self) -> Dict[str, Any]:
        """Run a comprehensive security audit"""
        logger.info("Starting comprehensive security audit...")
        
        # Create backup
        backup_path = self.cleanup.create_backup(self.project_path)
        
        # Scan for secrets
        logger.info("Scanning for secrets...")
        secret_issues = self.secret_scanner.scan_directory(self.project_path)
        self.all_issues.extend(secret_issues)
        
        # Scan for vulnerabilities
        logger.info("Scanning for vulnerabilities...")
        vulnerability_issues = []
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    issues = self.vulnerability_scanner.scan_file(file_path)
                    vulnerability_issues.extend(issues)
        self.all_issues.extend(vulnerability_issues)
        
        # Validate configuration files
        logger.info("Validating configuration files...")
        config_issues = []
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file.endswith(('.yaml', '.yml', '.json', '.ini', '.cfg')):
                    file_path = os.path.join(root, file)
                    issues = self.config_validator.validate_config_file(file_path)
                    config_issues.extend(issues)
        self.all_issues.extend(config_issues)
        
        # Scan dependencies
        logger.info("Scanning dependencies...")
        requirements_path = os.path.join(self.project_path, 'requirements.txt')
        if os.path.exists(requirements_path):
            dependency_issues = self.dependency_scanner.scan_requirements_file(requirements_path)
            self.all_issues.extend(dependency_issues)
        
        # Categorize issues
        categorized_issues = self._categorize_issues()
        
        # Generate report
        report = {}
        if backup_path:
            report = self._generate_report(categorized_issues, backup_path)
        
        self.scan_results = report
        return report
    
    def _categorize_issues(self) -> Dict[str, List[SecurityIssue]]:
        """Categorize issues by severity and type"""
        categorized = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for issue in self.all_issues:
            if issue.severity == 'CRITICAL':
                categorized['critical'].append(issue)
            elif issue.severity == 'HIGH':
                categorized['high'].append(issue)
            elif issue.severity == 'MEDIUM':
                categorized['medium'].append(issue)
            else:
                categorized['low'].append(issue)
        
        return categorized
    
    def _generate_report(self, categorized_issues: Dict[str, List[SecurityIssue]], backup_path: Optional[str]) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        total_issues = len(self.all_issues)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_path': self.project_path,
            'backup_path': backup_path,
            'summary': {
                'total_issues': total_issues,
                'critical': len(categorized_issues['critical']),
                'high': len(categorized_issues['high']),
                'medium': len(categorized_issues['medium']),
                'low': len(categorized_issues['low'])
            },
            'issues': categorized_issues,
            'recommendations': self._generate_recommendations(categorized_issues),
            'security_score': self._calculate_security_score(categorized_issues)
        }
        
        return report
    
    def _generate_recommendations(self, categorized_issues: Dict[str, List[SecurityIssue]]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if categorized_issues['critical']:
            recommendations.append("CRITICAL: Address all critical issues immediately before deployment")
        
        if categorized_issues['high']:
            recommendations.append("HIGH: Review and fix high-severity issues before production deployment")
        
        if categorized_issues['medium']:
            recommendations.append("MEDIUM: Consider addressing medium-severity issues for better security")
        
        if categorized_issues['low']:
            recommendations.append("LOW: Low-severity issues can be addressed in future updates")
        
        recommendations.extend([
            "Use environment variables for all sensitive configuration",
            "Implement proper input validation and sanitization",
            "Use secure random number generation (secrets module)",
            "Keep dependencies updated to latest secure versions",
            "Implement proper logging without sensitive data",
            "Use HTTPS for all external communications",
            "Implement proper authentication and authorization",
            "Regular security audits and penetration testing"
        ])
        
        return recommendations
    
    def _calculate_security_score(self, categorized_issues: Dict[str, List[SecurityIssue]]) -> float:
        """Calculate overall security score (0-100)"""
        total_issues = len(self.all_issues)
        if total_issues == 0:
            return 100.0
        
        # Weight issues by severity
        weighted_score = (
            len(categorized_issues['critical']) * 10 +
            len(categorized_issues['high']) * 5 +
            len(categorized_issues['medium']) * 2 +
            len(categorized_issues['low']) * 1
        )
        
        # Calculate score (higher is better)
        max_possible_score = total_issues * 10
        score = max(0, 100 - (weighted_score / max_possible_score * 100))
        
        return round(score, 2)
    
    async def perform_cleanup(self) -> Dict[str, Any]:
        """Perform security cleanup operations"""
        logger.info("Performing security cleanup...")
        
        cleanup_results = {
            'modified_files': [],
            'created_files': [],
            'backup_path': None
        }
        
        # Remove secrets
        critical_high_issues = [
            issue for issue in self.all_issues 
            if issue.category == "SECRET" and issue.severity in ["HIGH", "CRITICAL"]
        ]
        
        if critical_high_issues:
            modified_files = self.cleanup.remove_secrets(critical_high_issues)
            cleanup_results['modified_files'] = modified_files
        
        # Create environment template
        if any(issue.category == "SECRET" for issue in self.all_issues):
            template_path = self.cleanup.create_env_template(self.all_issues)
            cleanup_results['created_files'].append(template_path)
        
        # Update .gitignore
        if self.cleanup.update_gitignore():
            cleanup_results['created_files'].append('.gitignore (updated)')
        
        return cleanup_results
    
    def save_report(self, report: Dict[str, Any], filename: str = "security_audit_report.json") -> str:
        """Save audit report to file"""
        report_path = os.path.join(self.project_path, filename)
        
        # Convert datetime objects to strings for JSON serialization
        serializable_report = self._make_json_serializable(report)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2)
        
        logger.info(f"Security audit report saved to: {report_path}")
        return report_path
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def is_safe_for_github(self) -> bool:
        """Check if the codebase is safe for GitHub commit"""
        critical_high_secrets = [
            issue for issue in self.all_issues 
            if issue.category == "SECRET" and issue.severity in ["HIGH", "CRITICAL"]
        ]
        
        return len(critical_high_secrets) == 0

# Example usage and testing
async def test_security_audit():
    """Test the security audit system"""
    
    # Initialize auditor
    auditor = SecurityAuditor(".")
    
    # Run full audit
    report = await auditor.run_full_audit()
    
    print("Security Audit Results:")
    print(f"Total Issues: {report['summary']['total_issues']}")
    print(f"Critical: {report['summary']['critical']}")
    print(f"High: {report['summary']['high']}")
    print(f"Medium: {report['summary']['medium']}")
    print(f"Low: {report['summary']['low']}")
    print(f"Security Score: {report['security_score']}/100")
    
    # Check if safe for GitHub
    safe_for_github = auditor.is_safe_for_github()
    print(f"Safe for GitHub: {safe_for_github}")
    
    if not safe_for_github:
        print("Performing cleanup...")
        cleanup_results = await auditor.perform_cleanup()
        print(f"Modified files: {len(cleanup_results['modified_files'])}")
        print(f"Created files: {len(cleanup_results['created_files'])}")
    
    # Save report
    report_path = auditor.save_report(report)
    print(f"Report saved to: {report_path}")
    
    return report

if __name__ == "__main__":
    asyncio.run(test_security_audit()) 