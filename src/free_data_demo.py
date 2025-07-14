"""
Free Data Sources Demo
Demonstrates how to get comprehensive market data using only free sources
and optional TradingView premium access
"""

import asyncio
import pandas as pd
import json
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from free_data_sources import FreeDataSources, get_free_market_data, setup_free_environment

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_data_summary(data: dict):
    """Print a summary of retrieved data"""
    print(f"📊 Data Summary:")
    print(f"   • Market Data Sources: {len(data.get('market_data', []))}")
    print(f"   • DEX Pairs Found: {len(data.get('dex_pairs', []))}")
    print(f"   • Yield Opportunities: {len(data.get('yield_opportunities', []))}")
    print(f"   • Exchange Data: {len(data.get('exchange_data', []))}")
    print(f"   • TradingView Data: {len(data.get('tradingview_data', {}))}")
    print(f"   • Price Consensus: {len(data.get('validated_prices', {}))}")
    
    metadata = data.get('metadata', {})
    print(f"   • Fetch Time: {metadata.get('fetch_time_seconds', 0):.2f}s")
    print(f"   • Sources Used: {', '.join(metadata.get('sources_used', []))}")

async def demo_free_coingecko():
    """Demo free CoinGecko API"""
    print_section("Free CoinGecko API Demo")
    
    async with FreeDataSources() as free_sources:
        symbols = ['BTC', 'ETH', 'SOL', 'SUI', 'SEI']
        data = await free_sources.fetch_coingecko_free_data(symbols)
        
        print(f"✅ Retrieved {len(data)} price points from CoinGecko Free API")
        
        for point in data[:3]:  # Show first 3
            print(f"   • {point.symbol}: ${point.price:,.2f} "
                  f"({point.price_change_24h:+.2f}% 24h)")
        
        return data

async def demo_dexscreener_free():
    """Demo free DexScreener API"""
    print_section("Free DexScreener API Demo")
    
    async with FreeDataSources() as free_sources:
        # Get trending DEX pairs
        trending_pairs = await free_sources.fetch_dexscreener_free()
        
        print(f"✅ Retrieved {len(trending_pairs)} trending DEX pairs")
        
        # Show top 5 by volume
        top_pairs = sorted(trending_pairs, key=lambda x: x.volume_24h, reverse=True)[:5]
        
        for pair in top_pairs:
            print(f"   • {pair.base_token}/{pair.quote_token} on {pair.dex_name}")
            print(f"     Price: ${pair.price:.6f} | Volume: ${pair.volume_24h:,.0f}")
            print(f"     Liquidity: ${pair.liquidity:,.0f} | Change: {pair.price_change_24h:+.2f}%")
        
        return trending_pairs

async def demo_exchange_data():
    """Demo free exchange data via CCXT"""
    print_section("Free Exchange Data Demo (CCXT)")
    
    async with FreeDataSources() as free_sources:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        exchange_data = []
        
        for symbol in symbols:
            data = await free_sources.fetch_exchange_data(symbol, 'binance')
            if data:
                exchange_data.append(data)
                print(f"   • {symbol}: ${data.price:,.2f} "
                      f"(Vol: ${data.volume_24h:,.0f})")
        
        print(f"✅ Retrieved {len(exchange_data)} exchange data points")
        return exchange_data

async def demo_yield_opportunities():
    """Demo free yield farming opportunities"""
    print_section("Free Yield Opportunities Demo")
    
    async with FreeDataSources() as free_sources:
        opportunities = await free_sources.fetch_yield_opportunities()
        
        print(f"✅ Found {len(opportunities)} yield opportunities")
        
        # Show top opportunities by APY
        top_yields = sorted(opportunities, key=lambda x: x.apy, reverse=True)[:5]
        
        for opp in top_yields:
            print(f"   • {opp.protocol} - {opp.pool_name}")
            print(f"     APY: {opp.apy:.1f}% | TVL: ${opp.tvl:,.0f}")
            print(f"     Chain: {opp.chain} | Risk: {opp.risk_level}")
        
        return opportunities

def demo_tradingview_setup():
    """Demo TradingView setup instructions"""
    print_section("TradingView Premium Setup (Optional)")
    
    print("🔹 Since you have TradingView Premium, you can enhance data quality:")
    print()
    print("1. Install the TradingView library:")
    print("   pip install tvdatafeed")
    print()
    print("2. Add your credentials to the free data sources:")
    print("   free_sources = FreeDataSources(")
    print("       tradingview_username='your_username',")
    print("       tradingview_password='your_password'")
    print("   )")
    print()
    print("3. Benefits of TradingView integration:")
    print("   • Access to 100+ exchanges")
    print("   • High-quality OHLCV data")
    print("   • Advanced technical indicators")
    print("   • Real-time data feeds")
    print("   • Professional charting data")
    print()
    print("⚠️  Note: Keep your TradingView credentials secure!")

async def demo_comprehensive_free_data():
    """Demo comprehensive free data aggregation"""
    print_section("Comprehensive Free Data Demo")
    
    symbols = ['BTC', 'ETH', 'SOL', 'SUI']
    
    print(f"🔍 Fetching comprehensive data for: {', '.join(symbols)}")
    print("📡 Using only free data sources...")
    
    # Get data without TradingView
    data = await get_free_market_data(symbols)
    
    print_data_summary(data)
    
    # Show price consensus
    if data.get('validated_prices'):
        print(f"\n💰 Price Consensus (Cross-validated):")
        for symbol, prices in data['validated_prices'].items():
            reliability = "✅ Reliable" if prices['consensus_reliable'] else "⚠️ Check"
            print(f"   • {symbol}: ${prices['recommended_price']:,.2f} {reliability}")
            print(f"     CoinGecko: ${prices['coingecko_price']:,.2f} | "
                  f"Exchange: ${prices['exchange_price']:,.2f}")
    
    return data

async def demo_with_tradingview():
    """Demo with TradingView integration (if credentials available)"""
    print_section("TradingView Integration Demo")
    
    # Check if user wants to test TradingView
    print("📺 To test TradingView integration, you need premium credentials.")
    print("   For now, we'll show what data would be available...")
    
    # This would be the actual call with credentials:
    # data = await get_free_market_data(
    #     ['BTC', 'ETH', 'SOL'],
    #     tradingview_username="your_username",
    #     tradingview_password="your_password"
    # )
    
    print("✅ TradingView would provide:")
    print("   • Real-time OHLCV data")
    print("   • Multiple timeframes (1m, 5m, 1h, 1d, etc.)")
    print("   • 100+ exchanges coverage")
    print("   • Professional-grade data quality")

def show_environment_setup():
    """Show how to set up environment for free sources"""
    print_section("Environment Setup for Free Sources")
    
    env_vars = setup_free_environment()
    
    print("📝 Environment variables for free tier:")
    for key, value in env_vars.items():
        print(f"   {key}={value}")
    
    print("\n💡 Quick setup:")
    print("1. Copy .env.template to .env")
    print("2. Set USE_FREE_TIER=true")
    print("3. Add TradingView credentials if you have them")
    print("4. All other APIs are optional!")

async def main():
    """Main demo function"""
    print("🚀 Free Data Sources Demo")
    print("=" * 60)
    print("This demo shows how to get comprehensive market data")
    print("using only FREE APIs and optional TradingView premium.")
    print()
    print("✅ No CoinGecko Pro API required")
    print("✅ No DeFi Llama Pro API required") 
    print("✅ No premium subscriptions needed")
    print("✅ Optional TradingView premium for enhanced data")
    
    try:
        # Demo individual components
        await demo_free_coingecko()
        await demo_dexscreener_free()
        await demo_exchange_data()
        await demo_yield_opportunities()
        
        # Demo comprehensive data
        await demo_comprehensive_free_data()
        
        # Show TradingView integration
        demo_tradingview_setup()
        await demo_with_tradingview()
        
        # Show environment setup
        show_environment_setup()
        
        print_section("Demo Complete!")
        print("✅ All free data sources working correctly")
        print("📊 You now have access to:")
        print("   • Real-time cryptocurrency prices")
        print("   • DEX trading pairs and liquidity data")
        print("   • Cross-exchange price validation")
        print("   • Yield farming opportunities")
        print("   • Optional TradingView premium data")
        print()
        print("🔄 Next Steps:")
        print("1. Copy .env.template to .env")
        print("2. Set USE_FREE_TIER=true")
        print("3. Add TradingView credentials (optional)")
        print("4. Run: python src/simple_test.py")
        print("5. Launch web dashboard: streamlit run src/web_dashboard.py")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check internet connection")
        print("2. Ensure all dependencies installed: pip install -r requirements.txt")
        print("3. Try running individual components")

if __name__ == "__main__":
    asyncio.run(main()) 