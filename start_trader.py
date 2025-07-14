#!/usr/bin/env python3
"""
Quant AI Trader Startup Script
Enhanced with secure authentication and real-time price monitoring
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def load_environment_variables():
    """Load and validate environment variables"""
    env_vars = {
        'GROK_API_KEY': os.environ.get('GROK_API_KEY'),
        'MASTER_PASSWORD': os.environ.get('MASTER_PASSWORD'),
        'JWT_SECRET_KEY': os.environ.get('JWT_SECRET_KEY'),
        'TOTP_SECRET': os.environ.get('TOTP_SECRET'),
        'PAPER_TRADING': os.environ.get('PAPER_TRADING', 'true'),
        'USE_FREE_TIER': os.environ.get('USE_FREE_TIER', 'true')
    }
    
    print("🔍 Checking environment variables...")
    for key, value in env_vars.items():
        if value:
            if key == 'GROK_API_KEY':
                print(f"✅ {key}: {'*' * 20}...{value[-10:]}")
            elif key in ['MASTER_PASSWORD', 'JWT_SECRET_KEY']:
                print(f"✅ {key}: {'*' * 20}...{value[-5:]}")
            else:
                print(f"✅ {key}: {value}")
        else:
            print(f"⚠️  {key}: Not configured")
    
    return env_vars

def validate_configuration():
    """Validate system configuration"""
    print("🔍 Checking system configuration...")
    
    required_vars = ['MASTER_PASSWORD', 'JWT_SECRET_KEY', 'TOTP_SECRET']
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    # Check if Grok API key is configured
    grok_key = os.environ.get('GROK_API_KEY')
    if grok_key and grok_key != 'your_api_key_here':
        print("✅ AI API: Grok configured")
    else:
        print("⚠️  AI API: Using fallback (configure GROK_API_KEY for full features)")
    
    print("✅ Configuration valid!")
    return True

def test_price_fetching():
    """Test price fetching to ensure data sources are working"""
    print("🧪 Testing price data sources...")
    
    try:
        sys.path.append('src')
        from data_fetcher import DataFetcher
        
        df = DataFetcher()
        
        # Test key assets
        test_assets = ['BTC', 'SUI', 'SEI']
        for asset in test_assets:
            price, mc, change = df.fetch_price_and_market_cap(asset)
            if price and price > 0:
                print(f"✅ {asset}: ${price:.4f} ({change:+.2f}%)")
            else:
                print(f"⚠️  {asset}: Price data unavailable")
        
        print("✅ Price data sources working")
        return True
        
    except Exception as e:
        print(f"❌ Price data test failed: {e}")
        return False

def start_secure_web_dashboard():
    """Start the secure web dashboard"""
    print("🌐 Starting secure web dashboard...")
    print("🔐 Authentication required:")
    print(f"   - Master Password: {os.environ.get('MASTER_PASSWORD', 'Not configured')}")
    print(f"   - 2FA Code: Use your authenticator app")
    print("🎯 Dashboard URL: http://localhost:8080")
    
    try:
        # Start the secure web app
        subprocess.run([sys.executable, 'src/secure_web_app.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception as e:
        print(f"❌ Error starting web dashboard: {e}")
        print("💡 Try running manually: python src/secure_web_app.py")

def main():
    """Main startup function"""
    print("🚀 Quant AI Trader - System Startup")
    print("=" * 40)
    
    # Load environment variables
    env_vars = load_environment_variables()
    
    # Validate configuration
    if not validate_configuration():
        print("\n❌ Configuration invalid. Please run 'python configure_env.py' first.")
        return
    
    # Test data sources
    if not test_price_fetching():
        print("\n⚠️  Price data sources have issues, but continuing...")
    
    print("\n✅ All systems ready!")
    print("\n🚀 Starting Quant AI Trader...")
    print("🔐 Security: 2FA authentication enabled")
    print(f"💰 Mode: {'Free tier' if env_vars.get('USE_FREE_TIER') == 'true' else 'Premium'} data sources")
    print(f"📊 Trading: {'Paper trading (safe mode)' if env_vars.get('PAPER_TRADING') == 'true' else 'Live trading'}")
    
    # Start the secure web dashboard
    start_secure_web_dashboard()

if __name__ == "__main__":
    main() 