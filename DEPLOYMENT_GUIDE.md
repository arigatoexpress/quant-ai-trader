# Quant AI Trader Deployment Guide 🚀

This comprehensive guide covers initial setup, security configuration, and Google Cloud deployment for the Quant AI Trader system.

## 📋 Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Node.js**: 16+ (for web dashboard)
- **Docker**: Latest version
- **Git**: Latest version
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: Minimum 10GB free space

### Required Accounts & API Keys

1. **xAI Grok API**: [Get API Key](https://x.ai/)
2. **CoinGecko Pro**: [Get API Key](https://www.coingecko.com/en/api)
3. **DexScreener**: [Get API Key](https://docs.dexscreener.com/)
4. **DeFi Llama**: [Get API Key](https://defillama.com/docs/api)
5. **Google Cloud**: [Create Account](https://cloud.google.com/)

## 🚀 Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/quant-ai-trader.git
cd quant-ai-trader
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

```bash
# Copy environment template
cp .env.template .env

# Edit with your configuration
nano .env  # or use your preferred editor
```

#### Required Environment Variables

```bash
# Core Configuration
GROK_API_KEY=your_grok_api_key_here
INITIAL_CAPITAL=100000
PAPER_TRADING_MODE=true
ENABLE_AUTONOMOUS_TRADING=false

# Risk Management
MAX_POSITION_SIZE=0.25
RISK_TOLERANCE=0.15
KELLY_MULTIPLIER=0.5

# Data Sources
COINGECKO_API_KEY=your_coingecko_pro_key
DEXSCREENER_API_KEY=your_dexscreener_key
DEFILLAMA_API_KEY=your_defillama_key

# Security
MASTER_PASSWORD=your_strong_password_here
ENABLE_2FA=true
SESSION_TIMEOUT=3600

# Deployment
ENVIRONMENT=development
LOG_LEVEL=INFO
WEB_PORT=8080
```

### 5. Database Setup

```bash
# Create data directory
mkdir -p data

# Initialize databases (automatic on first run)
python src/secure_authentication.py
```

## 🔐 Security Configuration

### 1. Strong Password Setup

The system requires a strong master password with:
- Minimum 12 characters
- Uppercase and lowercase letters
- Numbers and special characters
- No common patterns or dictionary words

```bash
# Generate secure password (optional)
python -c "import secrets, string; print(''.join(secrets.choice(string.ascii_letters + string.digits + '!@#$%^&*') for _ in range(16)))"
```

### 2. Two-Factor Authentication (2FA)

#### Setup Steps:

1. **Install Authenticator App** (Choose one):
   - Google Authenticator (iOS/Android)
   - Authy (iOS/Android/Desktop)
   - Microsoft Authenticator (iOS/Android)
   - 1Password (Premium)

2. **Initialize 2FA**:
```bash
python -c "
from src.secure_authentication import create_default_system
auth = create_default_system()
success, qr_code, secret = auth.setup_2fa('your_username')
print(f'Manual entry secret: {secret}')
"
```

3. **Verify Setup**:
   - Scan QR code or enter secret manually in your authenticator app
   - Enter 6-digit code when logging in

### 3. API Key Security

#### Best Practices:

1. **Never commit API keys to version control**
2. **Use environment variables only**
3. **Rotate keys regularly (monthly)**
4. **Monitor API usage and limits**
5. **Restrict key permissions to minimum required**

#### Key Storage:
```bash
# Store in .env file (never commit this file)
echo "GROK_API_KEY=your_actual_key_here" >> .env

# Verify permissions
chmod 600 .env
```

### 4. Network Security

#### Firewall Configuration:
```bash
# Ubuntu/Debian
sudo ufw enable
sudo ufw allow 22    # SSH
sudo ufw allow 8080  # Web dashboard
sudo ufw deny 5432   # Block direct database access

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

## 🧪 Testing & Validation

### 1. Quick Validation

```bash
# Run basic tests
python src/simple_test.py

# Test authentication system
python -c "
from src.secure_authentication import create_default_system
auth = create_default_system()
print('Authentication system ready')
"

# Test singleton manager
python src/singleton_manager.py
```

### 2. Comprehensive Testing

```bash
# Run full test suite
python src/comprehensive_testing_framework.py

# Test real data integrations
python src/test_real_data_integrations.py

# Validate deployment
python src/main.py --validate
```

### 3. Security Audit

```bash
# Run security audit
python src/security_audit_cleanup.py

# Check for vulnerabilities
pip audit

# Validate configuration
python -c "
from src.secure_config_manager import SecureConfigManager
config = SecureConfigManager()
status = config.validate_configuration()
print(f'Config status: {status}')
"
```

## ☁️ Google Cloud Deployment

### 1. Prerequisites

#### Install Google Cloud CLI:
```bash
# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# macOS
brew install google-cloud-sdk

# Windows
# Download from: https://cloud.google.com/sdk/docs/install
```

#### Authenticate:
```bash
gcloud auth login
gcloud auth application-default login
```

### 2. Project Setup

```bash
# Create new project
gcloud projects create quant-ai-trader --name="Quant AI Trader"

# Set project
gcloud config set project quant-ai-trader

# Enable billing (required)
# Visit: https://console.cloud.google.com/billing
```

### 3. Deploy to Google Cloud

#### Automated Deployment:
```bash
# Set environment variables
export GOOGLE_CLOUD_PROJECT=quant-ai-trader
export GOOGLE_CLOUD_REGION=us-central1

# Run deployment script
chmod +x scripts/deploy_gcp.sh
./scripts/deploy_gcp.sh
```

#### Manual Deployment Steps:

1. **Enable APIs**:
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable sql-component.googleapis.com
gcloud services enable secretmanager.googleapis.com
```

2. **Build and Push Image**:
```bash
gcloud builds submit --tag gcr.io/quant-ai-trader/quant-ai-trader .
```

3. **Create Cloud SQL**:
```bash
gcloud sql instances create quantai-db \
    --database-version=POSTGRES_14 \
    --tier=db-f1-micro \
    --region=us-central1
```

4. **Deploy Cloud Run**:
```bash
gcloud run deploy quant-ai-trader \
    --image gcr.io/quant-ai-trader/quant-ai-trader \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8080 \
    --memory 2Gi \
    --cpu 1
```

### 4. Configure Secrets

#### Using Secret Manager:
```bash
# Create secrets
echo -n "your_grok_api_key" | gcloud secrets create grok-api-key --data-file=-
echo -n "your_coingecko_key" | gcloud secrets create coingecko-api-key --data-file=-
echo -n "secure_master_password" | gcloud secrets create master-password --data-file=-

# Grant access to Cloud Run
gcloud secrets add-iam-policy-binding grok-api-key \
    --member="serviceAccount:your-service-account@quant-ai-trader.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### 5. Monitoring & Logging

#### Set up monitoring:
```bash
# Run monitoring setup script
chmod +x scripts/setup_monitoring.sh
./scripts/setup_monitoring.sh
```

#### Access logs:
```bash
# View application logs
gcloud logs read "resource.type=cloud_run_revision AND resource.labels.service_name=quant-ai-trader"

# Stream real-time logs
gcloud logs tail "resource.type=cloud_run_revision AND resource.labels.service_name=quant-ai-trader"
```

## 🌐 Web Dashboard Access

### 1. Local Access

```bash
# Start the application
python src/main.py

# Access dashboard
open http://localhost:8080
```

### 2. Production Access

After Google Cloud deployment:
- Dashboard URL: `https://quant-ai-trader-xxxxx-uc.a.run.app`
- Admin panel: `https://quant-ai-trader-xxxxx-uc.a.run.app/admin`
- API docs: `https://quant-ai-trader-xxxxx-uc.a.run.app/docs`

### 3. Initial Login

1. **Default Admin Account**:
   - Username: `admin`
   - Password: Generated during setup (check logs)

2. **Setup 2FA**:
   - Scan QR code with authenticator app
   - Enter verification code
   - Save backup codes securely

## 🔧 Configuration Management

### 1. Application Configuration

Edit `config/config.yaml`:
```yaml
# Trading Configuration
trading:
  paper_mode: true
  max_position_size: 0.25
  risk_tolerance: 0.15
  
# Data Sources
data_sources:
  coingecko:
    enabled: true
    rate_limit: 500
  dexscreener:
    enabled: true
    rate_limit: 300
    
# Security
security:
  require_2fa: true
  session_timeout: 3600
  max_login_attempts: 3
```

### 2. Wallet Configuration

For multi-chain portfolio tracking:
```bash
# Add wallet addresses to .env
SUI_WALLET_1=0x1234...
SUI_WALLET_2=0x5678...
ETH_WALLET_1=0xabcd...
SOL_WALLET_1=ABC123...
```

### 3. Trading Parameters

Risk management settings:
```yaml
risk_management:
  max_daily_trades: 10
  max_drawdown: 0.15
  stop_loss_level: 0.1
  kelly_multiplier: 0.5
```

## 📊 Monitoring & Maintenance

### 1. Health Checks

```bash
# Check system health
curl http://localhost:8080/health

# Check singleton status
python -c "
from src.singleton_manager import get_global_singleton
singleton = get_global_singleton()
print(f'Singleton status: {singleton.get_status() if singleton else \"Not running\"}')
"
```

### 2. Performance Monitoring

Access monitoring dashboards:
- **Application**: `http://localhost:8080/dashboard`
- **System metrics**: `http://localhost:9090` (Prometheus)
- **Visualizations**: `http://localhost:3000` (Grafana)

### 3. Log Management

```bash
# View application logs
tail -f logs/quant_ai_trader.log

# View error logs
tail -f logs/error.log

# View security audit logs
tail -f logs/security_audit.log
```

### 4. Backup & Recovery

#### Automated Backups:
```bash
# Create backup script
cat > scripts/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p backups/$DATE
cp -r data/ backups/$DATE/
cp -r config/ backups/$DATE/
cp -r logs/ backups/$DATE/
tar -czf backups/backup_$DATE.tar.gz backups/$DATE/
rm -rf backups/$DATE/
echo "Backup created: backup_$DATE.tar.gz"
EOF

chmod +x scripts/backup.sh
```

#### Schedule Backups:
```bash
# Add to crontab
crontab -e

# Add line for daily backups at 2 AM
0 2 * * * /path/to/quant-ai-trader/scripts/backup.sh
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Authentication Errors
```bash
# Reset authentication database
rm data/auth.db
python src/secure_authentication.py

# Check 2FA setup
python -c "
from src.secure_authentication import create_default_system
auth = create_default_system()
print(auth.get_user_status('your_username'))
"
```

#### 2. API Connection Issues
```bash
# Test API connections
python src/test_real_data_integrations.py

# Check rate limits
python -c "
from src.real_data_integrations import DataAggregationEngine
engine = DataAggregationEngine()
print('Testing connections...')
"
```

#### 3. Memory Issues
```bash
# Monitor memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Available: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"

# Restart application if needed
pkill -f "python src/main.py"
python src/main.py
```

#### 4. Database Issues
```bash
# Check database integrity
sqlite3 data/auth.db "PRAGMA integrity_check;"

# Rebuild database if corrupted
mv data/auth.db data/auth.db.backup
python src/secure_authentication.py
```

### Support Resources

- **Documentation**: [GitHub Wiki](https://github.com/yourusername/quant-ai-trader/wiki)
- **Issues**: [GitHub Issues](https://github.com/yourusername/quant-ai-trader/issues)
- **Community**: [Discord Server](https://discord.gg/quantaitrader)
- **Email**: support@quantaitrader.com

## 🔄 Updates & Maintenance

### 1. Update Application

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run migrations (if any)
python scripts/migrate.py

# Restart application
sudo systemctl restart quant-ai-trader
```

### 2. Security Updates

```bash
# Update security patches
pip install --upgrade cryptography bcrypt pyjwt

# Rotate JWT secret
rm config/.jwt_secret
python src/secure_authentication.py

# Update API keys (monthly)
# Edit .env file with new keys
```

### 3. Monitoring Alerts

Set up alerts for:
- High memory usage (>80%)
- Failed authentication attempts (>10/hour)
- API rate limit breaches
- Unexpected downtime
- Security events

## ✅ Production Checklist

Before going live:

- [ ] All API keys configured and tested
- [ ] Strong master password set
- [ ] 2FA enabled and tested
- [ ] SSL/TLS certificates configured
- [ ] Firewall rules configured
- [ ] Backup system tested
- [ ] Monitoring alerts configured
- [ ] Security audit completed
- [ ] Performance testing completed
- [ ] Disaster recovery plan documented

## 📞 Support

For deployment assistance:
1. Check this guide thoroughly
2. Search existing GitHub issues
3. Join our Discord community
4. Create a new GitHub issue with detailed logs
5. Contact support for enterprise assistance

---

**⚠️ Security Notice**: This system handles real financial data and can execute trades. Always use strong security practices, keep software updated, and monitor system activity regularly. 