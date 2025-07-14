"""
Test Real Data Integrations
Comprehensive testing of all real data sources and validation of data quality
"""

import asyncio
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"🔍 {title}")
    print("="*80)

def print_section(title: str):
    """Print formatted section"""
    print(f"\n📊 {title}")
    print("-" * 60)

async def test_dexscreener_integration():
    """Test DexScreener API integration"""
    print_section("DexScreener API Integration")
    
    try:
        from real_data_integrations import DexScreenerAPI
        
        async with DexScreenerAPI() as dex_api:
            print("✅ DexScreener API connected")
            
            # Test 1: Search for popular tokens
            print("\n🔍 Testing token search...")
            sui_pairs = await dex_api.search_pairs("SUI")
            print(f"   Found {len(sui_pairs)} SUI pairs")
            
            if sui_pairs:
                top_pair = sui_pairs[0]
                print(f"   Top pair: {top_pair.base_token.get('symbol', 'N/A')}/{top_pair.quote_token.get('symbol', 'N/A')}")
                print(f"   Price: ${top_pair.price_usd if top_pair.price_usd else 'N/A'}")
                print(f"   24h Volume: ${top_pair.volume_24h:,.0f}")
                print(f"   Liquidity: ${top_pair.liquidity_usd:,.0f}")
            
            # Test 2: Get new pairs
            print("\n🆕 Testing new pairs...")
            new_pairs = await dex_api.get_new_pairs()
            print(f"   Found {len(new_pairs)} new pairs")
            
            return {'status': 'success', 'pairs_found': len(sui_pairs), 'new_pairs': len(new_pairs)}
            
    except Exception as e:
        print(f"❌ DexScreener test failed: {e}")
        return {'status': 'failed', 'error': str(e)}

async def test_coingecko_integration():
    """Test CoinGecko Pro API integration"""
    print_section("CoinGecko Pro API Integration")
    
    try:
        from real_data_integrations import CoinGeckoProAPI
        
        async with CoinGeckoProAPI() as cg_api:
            print("✅ CoinGecko API connected")
            
            # Test 1: Get market data
            print("\n💰 Testing market data...")
            symbols = ['BTC', 'ETH', 'SOL', 'SUI', 'SEI']
            market_data = await cg_api.get_market_data(symbols)
            print(f"   Retrieved data for {len(market_data)} symbols")
            
            for data in market_data[:3]:  # Show first 3
                print(f"   {data.symbol}: ${data.price:,.2f} ({data.price_change_24h:+.2f}%)")
                print(f"      Volume: ${data.volume_24h:,.0f}")
                print(f"      Market Cap: ${data.market_cap:,.0f}" if data.market_cap else "      Market Cap: N/A")
            
            # Test 2: Get trending coins
            print("\n📈 Testing trending coins...")
            trending = await cg_api.get_trending_coins()
            print(f"   Found {len(trending)} trending coins")
            
            # Test 3: Get gainers/losers
            print("\n🎯 Testing top gainers/losers...")
            gainers_losers = await cg_api.get_top_gainers_losers()
            print(f"   Gainers: {len(gainers_losers.get('gainers', []))}")
            print(f"   Losers: {len(gainers_losers.get('losers', []))}")
            
            return {'status': 'success', 'market_data_count': len(market_data), 'trending_count': len(trending)}
            
    except Exception as e:
        print(f"❌ CoinGecko test failed: {e}")
        return {'status': 'failed', 'error': str(e)}

async def test_defillama_integration():
    """Test DeFi Llama API integration"""
    print_section("DeFi Llama API Integration")
    
    try:
        from real_data_integrations import DeFiLlamaAPI
        
        async with DeFiLlamaAPI() as defi_api:
            print("✅ DeFi Llama API connected")
            
            # Test 1: Get protocols
            print("\n🏦 Testing protocols data...")
            protocols = await defi_api.get_protocols()
            print(f"   Retrieved {len(protocols)} protocols")
            
            if protocols:
                top_protocols = sorted(protocols, key=lambda x: x.tvl, reverse=True)[:5]
                print("   Top 5 protocols by TVL:")
                for i, protocol in enumerate(top_protocols, 1):
                    print(f"   {i}. {protocol.protocol}: ${protocol.tvl:,.0f}")
            
            # Test 2: Get yield opportunities
            print("\n🌾 Testing yield opportunities...")
            yields = await defi_api.get_high_yield_opportunities(min_apy=15, min_tvl=1000000)
            print(f"   Found {len(yields)} high-yield opportunities")
            
            if yields:
                print("   Top 3 yield opportunities:")
                for i, pool in enumerate(yields[:3], 1):
                    print(f"   {i}. {pool.symbol} on {pool.project}: {pool.apy:.1f}% APY")
                    print(f"      TVL: ${pool.tvl_usd:,.0f}, Chain: {pool.chain}")
            
            # Test 3: Get chains data
            print("\n⛓️  Testing chains data...")
            chains = await defi_api.get_chains()
            print(f"   Retrieved data for {len(chains)} chains")
            
            return {'status': 'success', 'protocols_count': len(protocols), 'yields_count': len(yields)}
            
    except Exception as e:
        print(f"❌ DeFi Llama test failed: {e}")
        return {'status': 'failed', 'error': str(e)}

async def test_sui_integration():
    """Test Sui Network API integration"""
    print_section("Sui Network API Integration")
    
    try:
        from real_data_integrations import SuiNetworkAPI
        
        async with SuiNetworkAPI() as sui_api:
            print("✅ Sui Network API connected")
            
            # Test 1: Get network info
            print("\n🌐 Testing network information...")
            network_info = await sui_api.get_network_info()
            print(f"   Total Transaction Blocks: {network_info.get('total_transaction_blocks', 'N/A')}")
            print(f"   Latest Checkpoint: {network_info.get('latest_checkpoint', 'N/A')}")
            
            # Test 2: Get gas price
            print("\n⛽ Testing gas price...")
            gas_price = await sui_api.get_gas_price()
            print(f"   Current Gas Price: {gas_price} MIST")
            
            # Test 3: Get coin metadata
            print("\n🪙 Testing coin metadata...")
            sui_metadata = await sui_api.get_coin_metadata("0x2::sui::SUI")
            print(f"   SUI metadata available: {'✅' if sui_metadata else '❌'}")
            
            return {'status': 'success', 'network_info': bool(network_info), 'gas_price': gas_price}
            
    except Exception as e:
        print(f"❌ Sui Network test failed: {e}")
        return {'status': 'failed', 'error': str(e)}

async def test_noodles_integration():
    """Test Noodles Finance API integration"""
    print_section("Noodles Finance API Integration")
    
    try:
        from real_data_integrations import NoodlesFinanceAPI
        
        async with NoodlesFinanceAPI() as noodles_api:
            print("✅ Noodles Finance API connected")
            
            # Test 1: Get pools
            print("\n🏊 Testing liquidity pools...")
            pools = await noodles_api.get_pools()
            print(f"   Found {len(pools)} liquidity pools")
            
            for pool in pools[:3]:  # Show first 3
                print(f"   • {pool.get('name', 'N/A')}: {pool.get('apy', 0):.1f}% APY")
                print(f"     TVL: ${pool.get('tvl', 0):,.0f}")
            
            # Test 2: Get farms
            print("\n🚜 Testing yield farms...")
            farms = await noodles_api.get_farms()
            print(f"   Found {len(farms)} yield farms")
            
            for farm in farms[:2]:  # Show first 2
                print(f"   • {farm.get('name', 'N/A')}: {farm.get('apy', 0):.1f}% APY")
                print(f"     TVL: ${farm.get('tvl', 0):,.0f}")
            
            # Test 3: Get token prices
            print("\n💎 Testing token prices...")
            prices = await noodles_api.get_token_prices()
            print(f"   Retrieved prices for {len(prices)} tokens")
            
            for token, price in list(prices.items())[:5]:  # Show first 5
                print(f"   • {token}: ${price:.4f}")
            
            return {'status': 'success', 'pools_count': len(pools), 'farms_count': len(farms)}
            
    except Exception as e:
        print(f"❌ Noodles Finance test failed: {e}")
        return {'status': 'failed', 'error': str(e)}

async def test_additional_sources():
    """Test additional data sources"""
    print_section("Additional Data Sources Integration")
    
    try:
        from real_data_integrations import AdditionalDataSources
        
        async with AdditionalDataSources() as additional_api:
            print("✅ Additional sources API connected")
            
            # Test 1: Jupiter prices
            print("\n🔀 Testing Jupiter aggregator...")
            jupiter_prices = await additional_api.get_jupiter_prices()
            print(f"   Retrieved {len(jupiter_prices)} Jupiter prices")
            
            # Test 2: Birdeye trending
            print("\n🦅 Testing Birdeye trending...")
            birdeye_trending = await additional_api.get_birdeye_trending()
            print(f"   Found {len(birdeye_trending)} trending tokens on Birdeye")
            
            # Test 3: GeckoTerminal pools
            print("\n🦎 Testing GeckoTerminal pools...")
            gecko_pools = await additional_api.get_gecko_terminal_pools()
            print(f"   Found {len(gecko_pools)} pools on GeckoTerminal")
            
            return {'status': 'success', 'jupiter_prices': len(jupiter_prices), 'birdeye_trending': len(birdeye_trending)}
            
    except Exception as e:
        print(f"❌ Additional sources test failed: {e}")
        return {'status': 'failed', 'error': str(e)}

async def test_data_aggregation_engine():
    """Test the comprehensive data aggregation engine"""
    print_section("Data Aggregation Engine Integration")
    
    try:
        from real_data_integrations import DataAggregationEngine
        
        async with DataAggregationEngine() as engine:
            print("✅ Data Aggregation Engine initialized")
            
            # Test comprehensive data fetch
            print("\n🔄 Testing comprehensive data aggregation...")
            symbols = ['BTC', 'ETH', 'SOL', 'SUI', 'SEI']
            start_time = time.time()
            
            comprehensive_data = await engine.get_comprehensive_market_data(symbols)
            fetch_time = time.time() - start_time
            
            print(f"   ⏱️  Data aggregated in {fetch_time:.2f} seconds")
            print(f"   📊 Market data points: {len(comprehensive_data.get('market_data', []))}")
            print(f"   🔄 DEX pairs: {len(comprehensive_data.get('dex_pairs', []))}")
            print(f"   🏦 DeFi protocols: {len(comprehensive_data.get('defi_protocols', []))}")
            print(f"   🌾 DeFi yields: {len(comprehensive_data.get('defi_yields', []))}")
            print(f"   🌐 Sui network data: {'✅' if comprehensive_data.get('sui_network') else '❌'}")
            print(f"   🍜 Noodles Finance data: {'✅' if comprehensive_data.get('noodles_finance') else '❌'}")
            
            # Show data quality metrics
            metadata = comprehensive_data.get('metadata', {})
            print(f"\n📈 Data Quality Metrics:")
            print(f"   Sources Active: {metadata.get('sources_active', 0)}")
            print(f"   Data Quality Score: {metadata.get('data_quality_score', 0):.2f}")
            print(f"   Total Symbols Requested: {len(metadata.get('symbols_requested', []))}")
            
            # Show sample data
            market_data = comprehensive_data.get('market_data', [])
            if market_data:
                print(f"\n💰 Sample Market Data:")
                for data in market_data[:3]:
                    if hasattr(data, 'symbol'):
                        print(f"   {data.symbol}: ${data.price:,.4f} ({data.price_change_24h:+.2f}%)")
                        print(f"      Volume: ${data.volume_24h:,.0f}")
                        print(f"      Source: {data.source}")
            
            # Show top yield opportunities
            defi_yields = comprehensive_data.get('defi_yields', [])
            if defi_yields:
                print(f"\n🌾 Top Yield Opportunities:")
                for i, pool in enumerate(defi_yields[:3], 1):
                    if hasattr(pool, 'symbol'):
                        print(f"   {i}. {pool.symbol} on {pool.project}: {pool.apy:.1f}% APY")
                        print(f"      TVL: ${pool.tvl_usd:,.0f}, Chain: {pool.chain}")
            
            return {
                'status': 'success',
                'fetch_time': fetch_time,
                'data_quality_score': metadata.get('data_quality_score', 0),
                'sources_active': metadata.get('sources_active', 0),
                'total_data_points': len(market_data) + len(comprehensive_data.get('dex_pairs', []))
            }
            
    except Exception as e:
        print(f"❌ Data Aggregation Engine test failed: {e}")
        return {'status': 'failed', 'error': str(e)}

async def test_integration_with_existing_system():
    """Test integration with existing system components"""
    print_section("Integration with Existing System")
    
    try:
        # Test with enhanced data stream integrations
        from data_stream_integrations import DataStreamIntegrations
        
        async with DataStreamIntegrations() as data_streams:
            print("✅ Enhanced Data Stream Integrations connected")
            
            # Test comprehensive data fetch
            symbols = ['BTC', 'ETH', 'SOL', 'SUI']
            start_time = time.time()
            
            data = await data_streams.get_comprehensive_market_data(symbols)
            fetch_time = time.time() - start_time
            
            print(f"   ⏱️  Integration test completed in {fetch_time:.2f} seconds")
            print(f"   📊 CoinGecko data: {len(data.get('coingecko_data', []))}")
            print(f"   🔄 DEX data: {len(data.get('dex_data', []))}")
            print(f"   🏦 DeFi protocols: {len(data.get('defi_protocols', []))}")
            print(f"   🌾 DeFi yields: {len(data.get('defi_yields', []))}")
            print(f"   🎯 Total opportunities: {data.get('total_opportunities', 0)}")
            print(f"   📡 Real data used: {'✅' if data.get('real_data') else '❌'}")
            
            return {
                'status': 'success',
                'real_data_used': data.get('real_data', False),
                'total_opportunities': data.get('total_opportunities', 0),
                'fetch_time': fetch_time
            }
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return {'status': 'failed', 'error': str(e)}

def check_environment_setup():
    """Check if API keys and environment are properly configured"""
    print_section("Environment Configuration Check")
    
    required_vars = ['COINGECKO_API_KEY', 'SUI_RPC_URL']
    optional_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'NEWS_API_KEY']
    
    config_status = {}
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Configured")
            config_status[var] = True
        else:
            print(f"❌ {var}: Not configured")
            config_status[var] = False
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: Configured (optional)")
            config_status[var] = True
        else:
            print(f"⚠️  {var}: Not configured (optional)")
            config_status[var] = False
    
    return config_status

async def run_comprehensive_tests():
    """Run all real data integration tests"""
    print_header("Real Data Integrations - Comprehensive Test Suite")
    
    # Check environment
    env_status = check_environment_setup()
    
    # Initialize test results
    test_results = {}
    
    # Run all tests
    tests = [
        ("DexScreener API", test_dexscreener_integration),
        ("CoinGecko Pro API", test_coingecko_integration),
        ("DeFi Llama API", test_defillama_integration),
        ("Sui Network API", test_sui_integration),
        ("Noodles Finance API", test_noodles_integration),
        ("Additional Sources", test_additional_sources),
        ("Data Aggregation Engine", test_data_aggregation_engine),
        ("System Integration", test_integration_with_existing_system)
    ]
    
    for test_name, test_func in tests:
        try:
            print(f"\n🔍 Running {test_name} test...")
            result = await test_func()
            test_results[test_name] = result
            
            if result.get('status') == 'success':
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
                
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            test_results[test_name] = {'status': 'crashed', 'error': str(e)}
    
    # Generate summary report
    print_section("Test Results Summary")
    
    passed = sum(1 for result in test_results.values() if result.get('status') == 'success')
    total = len(test_results)
    
    print(f"📊 Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    for test_name, result in test_results.items():
        status = result.get('status', 'unknown')
        if status == 'success':
            print(f"✅ {test_name}")
        elif status == 'failed':
            print(f"❌ {test_name}: {result.get('error', 'Unknown error')}")
        else:
            print(f"💥 {test_name}: {result.get('error', 'Crashed')}")
    
    # Save results to file
    try:
        results_file = {
            'timestamp': datetime.now().isoformat(),
            'environment_status': env_status,
            'test_results': test_results,
            'summary': {
                'total_tests': total,
                'passed_tests': passed,
                'success_rate': passed/total*100
            }
        }
        
        with open('real_data_test_results.json', 'w') as f:
            json.dump(results_file, f, indent=2, default=str)
        
        print(f"\n💾 Test results saved to: real_data_test_results.json")
        
    except Exception as e:
        print(f"⚠️  Failed to save test results: {e}")
    
    # Final recommendations
    print_section("Recommendations")
    
    if passed == total:
        print("🎉 All tests passed! Your real data integrations are working perfectly.")
        print("   ✅ Ready for production deployment")
        print("   ✅ All data sources are accessible")
        print("   ✅ Data aggregation is functioning correctly")
    elif passed >= total * 0.75:
        print("✅ Most tests passed! Minor issues to address:")
        failed_tests = [name for name, result in test_results.items() if result.get('status') != 'success']
        for test in failed_tests:
            print(f"   • Fix {test} integration")
    else:
        print("⚠️  Multiple integration issues detected:")
        print("   • Check API keys and environment configuration")
        print("   • Verify network connectivity")
        print("   • Review rate limiting settings")
        print("   • Consider using fallback mock data for development")
    
    return test_results

if __name__ == "__main__":
    print("🚀 Starting Real Data Integrations Test Suite...")
    print("This will test all data sources and validate functionality.\n")
    
    # Run the comprehensive test suite
    asyncio.run(run_comprehensive_tests()) 