#!/usr/bin/env python3
"""
Secure Web Application for Quant AI Trader
Includes 2FA authentication and real-time price updates
"""

import os
import sys
import time
from datetime import datetime, timedelta
from flask import Flask, render_template_string, request, session, redirect, url_for, jsonify, flash
import pyotp
import jwt
from functools import wraps

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import DataFetcher
from macro_analyzer import MacroAnalyzer
from onchain_analyzer import OnchainAnalyzer
from trading_agent import TradingAgent
from news_fetcher import NewsFetcher
from technical_analyzer import TechnicalAnalyzer

app = Flask(__name__)
app.secret_key = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-here')

# Initialize components
fetcher = DataFetcher()
news_fetcher = NewsFetcher()

# Security configuration
MASTER_PASSWORD = os.environ.get('MASTER_PASSWORD', '')
TOTP_SECRET = os.environ.get('TOTP_SECRET', '')
JWT_SECRET = os.environ.get('JWT_SECRET_KEY', '')

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = session.get('auth_token')
        if not token:
            return redirect(url_for('login'))
        
        try:
            jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            return f(*args, **kwargs)
        except jwt.ExpiredSignatureError:
            session.pop('auth_token', None)
            flash('Session expired. Please login again.', 'warning')
            return redirect(url_for('login'))
        except jwt.InvalidTokenError:
            session.pop('auth_token', None)
            flash('Invalid session. Please login again.', 'error')
            return redirect(url_for('login'))
    
    return decorated_function

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant AI Trader - Secure Login</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 400px; margin: 100px auto; padding: 20px; }
        .login-form { background: #f5f5f5; padding: 30px; border-radius: 10px; }
        input { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
        button { width: 100%; padding: 12px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .error { color: red; margin-top: 10px; }
        .success { color: green; margin-top: 10px; }
        .info { color: #007bff; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="login-form">
        <h2>🔐 Quant AI Trader</h2>
        <p>Secure Authentication Required</p>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="{{ category }}">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <form method="POST">
            <input type="password" name="master_password" placeholder="Master Password" required>
            <input type="text" name="totp_code" placeholder="2FA Code (6 digits)" required maxlength="6">
            <button type="submit">🚀 Login</button>
        </form>
        
        <div class="info">
            <p><strong>Setup Instructions:</strong></p>
            <p>1. Use your master password from .env file</p>
            <p>2. Get 2FA code from your authenticator app</p>
            <p>3. Prices update every 30 seconds automatically</p>
        </div>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant AI Trader Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f8f9fa; }
        .header { background: #007bff; color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .price-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .price-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .price { font-size: 24px; font-weight: bold; color: #007bff; }
        .change-positive { color: #28a745; }
        .change-negative { color: #dc3545; }
        .market-cap { color: #6c757d; font-size: 14px; }
        .section { background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .logout { float: right; background: #dc3545; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
        .refresh-time { color: #6c757d; font-size: 12px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #dee2e6; }
        th { background: #f8f9fa; }
    </style>
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            window.location.reload();
        }, 30000);
    </script>
</head>
<body>
    <div class="header">
        <h1>🚀 Quant AI Trader Dashboard</h1>
        <p>Real-time Crypto Market Intelligence | Last Updated: {{ update_time }}</p>
        <a href="{{ url_for('logout') }}" class="logout">🔓 Logout</a>
    </div>

    <div class="price-grid">
        {% for asset, data in assets.items() %}
        <div class="price-card">
            <h3>{{ asset }}</h3>
            <div class="price">${{ data.price }}</div>
            <div class="market-cap">Market Cap: ${{ data.market_cap }}</div>
            <div class="{% if data.change_24h_raw >= 0 %}change-positive{% else %}change-negative{% endif %}">
                24h: {{ data.change_24h }}
            </div>
            <div class="{% if data.change_7d_raw >= 0 %}change-positive{% else %}change-negative{% endif %}">
                7d: {{ data.change_7d }}
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="section">
        <h2>📊 Market Highlights</h2>
        <ul>
            {% for highlight in highlights %}
            <li>{{ highlight }}</li>
            {% endfor %}
        </ul>
    </div>

    <div class="section">
        <h2>🔍 Technical Analysis</h2>
        <ul>
            {% for line in technical %}
            <li>{{ line }}</li>
            {% endfor %}
        </ul>
    </div>

    <div class="section">
        <h2>📈 Trading Signals</h2>
        <table>
            <tr><th>Asset</th><th>Signal</th><th>Current Price</th><th>Target</th><th>Confidence</th></tr>
            {% for pair, sig in signals.items() %}
            <tr>
                <td>{{ pair }}</td>
                <td><strong>{{ sig.signal }}</strong></td>
                <td>${{ "%.4f"|format(sig.current_price) }}</td>
                <td>${{ "%.4f"|format(sig.predicted_price) }}</td>
                <td>{{ "%.1f"|format(sig.risk_reward * 100) }}%</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="refresh-time">
        Auto-refresh in 30 seconds | Free tier data sources active
    </div>
</body>
</html>
"""

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        master_password = request.form.get('master_password')
        totp_code = request.form.get('totp_code')
        
        # Verify master password
        if master_password != MASTER_PASSWORD:
            flash('Invalid master password', 'error')
            return render_template_string(LOGIN_TEMPLATE)
        
        # Verify TOTP
        totp = pyotp.TOTP(TOTP_SECRET)
        if not totp.verify(totp_code, valid_window=1):
            flash('Invalid 2FA code', 'error')
            return render_template_string(LOGIN_TEMPLATE)
        
        # Create JWT token
        payload = {
            'user': 'trader',
            'exp': datetime.utcnow() + timedelta(hours=1),
            'iat': datetime.utcnow()
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
        session['auth_token'] = token
        
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
def logout():
    session.pop('auth_token', None)
    flash('Logged out successfully', 'info')
    return redirect(url_for('login'))

@app.route('/')
@require_auth
def dashboard():
    """Main dashboard with real-time data"""
    assets_data = {}
    highlights_calc = []
    
    # Get real-time prices for configured assets
    for asset in fetcher.config["assets"]:
        print(f"Fetching data for {asset}...")
        price, market_cap, change_24h = fetcher.fetch_price_and_market_cap(asset)
        week_change = fetcher.fetch_week_change(asset)
        
        highlights_calc.append({
            "asset": asset, 
            "change_24h": change_24h if change_24h is not None else 0, 
            "week": week_change if week_change is not None else 0
        })
        
        assets_data[asset] = {
            "price": f"{price:,.4f}" if price else "N/A",
            "market_cap": f"{market_cap:,.0f}" if market_cap else "N/A",
            "change_24h": f"{change_24h:+.2f}%" if change_24h is not None else "N/A",
            "change_24h_raw": change_24h if change_24h is not None else 0,
            "change_7d": f"{week_change:+.2f}%" if week_change is not None else "N/A",
            "change_7d_raw": week_change if week_change is not None else 0,
        }
    
    # Calculate highlights
    if highlights_calc:
        best_24h = max(highlights_calc, key=lambda x: x["change_24h"])
        best_week = max(highlights_calc, key=lambda x: x["week"])
        highlights = [
            f"🚀 Top 24h Gainer: {best_24h['asset']} ({best_24h['change_24h']:+.2f}%)",
            f"📈 Top 7d Gainer: {best_week['asset']} ({best_week['week']:+.2f}%)",
            f"💰 Free tier providing live data for {len(assets_data)} assets",
            f"🔄 Data updated from CoinGecko API at {datetime.now().strftime('%H:%M:%S')}"
        ]
    else:
        highlights = ["No data available"]
    
    # Technical analysis
    technical_lines = []
    for asset in fetcher.config["assets"]:
        df = fetcher.fetch_market_data(asset, "1d")
        if df is not None and not df.empty:
            try:
                for t in TechnicalAnalyzer(df).analyze():
                    technical_lines.append(f"{asset}: {t}")
            except Exception as e:
                technical_lines.append(f"{asset}: Analysis error - {str(e)}")
    
    if not technical_lines:
        technical_lines = ["Technical analysis data loading..."]
    
    # Trading signals
    try:
        agent = TradingAgent(fetcher.config, fetcher)
        signals = agent.generate_trade_signals()
    except Exception as e:
        signals = {"Error": {"signal": "N/A", "current_price": 0, "predicted_price": 0, "risk_reward": 0}}
    
    return render_template_string(
        DASHBOARD_TEMPLATE,
        assets=assets_data,
        highlights=highlights,
        technical=technical_lines,
        signals=signals,
        update_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    )

@app.route('/api/prices')
@require_auth
def api_prices():
    """API endpoint for price data"""
    prices = {}
    for asset in fetcher.config["assets"]:
        price, market_cap, change_24h = fetcher.fetch_price_and_market_cap(asset)
        prices[asset] = {
            "price": price,
            "market_cap": market_cap,
            "change_24h": change_24h,
            "timestamp": datetime.now().isoformat()
        }
    return jsonify(prices)

if __name__ == "__main__":
    print("🔐 Starting Secure Quant AI Trader Web Dashboard")
    print(f"🌐 Access at: http://localhost:8080")
    print(f"🔑 Master Password: {MASTER_PASSWORD}")
    print(f"📱 2FA Secret: {TOTP_SECRET}")
    print(f"🚀 Paper Trading Mode: {os.environ.get('PAPER_TRADING', 'true')}")
    
    app.run(host="127.0.0.1", port=8080, debug=False) 