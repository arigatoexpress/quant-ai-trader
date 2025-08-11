"""
Multi-Chain Portfolio Analyzer - Analyze wallets across multiple blockchains
"""

import asyncio
import json
import time
import requests
import base64
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import yaml
from openai import OpenAI

from data_fetcher import DataFetcher


@dataclass
class WalletBalance:
    """Represents a token balance in a wallet"""
    token_symbol: str
    token_address: str
    balance: float
    usd_value: float
    chain: str
    wallet_address: str
    timestamp: datetime


@dataclass
class ChainAnalysis:
    """Analysis results for a specific blockchain"""
    chain: str
    total_value: float
    token_count: int
    top_holdings: List[WalletBalance]
    risk_level: str
    diversification_score: float


@dataclass
class PortfolioRecommendation:
    """Trading recommendation based on portfolio analysis"""
    action: str  # 'BUY', 'SELL', 'REBALANCE', 'HOLD'
    token: str
    current_allocation: float
    target_allocation: float
    reasoning: str
    confidence: float
    priority: str  # 'HIGH', 'MEDIUM', 'LOW'
    estimated_impact: float


class MultiChainPortfolioAnalyzer:
    """Comprehensive multi-chain portfolio analyzer"""
    
    def __init__(self, config_path=None):
        # Load configuration
        if config_path is None:
            config_path = 'config/config.yaml'
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        self.data_fetcher = DataFetcher()
        
        # Initialize GROK client for AI analysis
        self.grok_client = OpenAI(
            api_key=config.get('grok_api_key'),
            base_url="https://api.x.ai/v1",
        )
        
        # RPC endpoints
        self.sui_rpc = config.get('sui_rpc', 'https://fullnode.mainnet.sui.io')
        self.solana_rpc = 'https://api.mainnet-beta.solana.com'
        self.ethereum_rpc = 'https://mainnet.infura.io/v3/your-infura-key'  # Replace with actual key
        self.base_rpc = 'https://mainnet.base.org'
        self.sei_rpc = 'https://rpc.sei.io'
        
        # User's wallet addresses (configure in .env file for security)
        self.wallet_addresses = {
            'sui': self._get_wallet_addresses("SUI_WALLET"),
            'solana': self._get_wallet_addresses("SOLANA_WALLET"),
            'ethereum': self._get_wallet_addresses("ETHEREUM_WALLET"),
            'base': self._get_wallet_addresses("BASE_WALLET"),
            'sei': self._get_wallet_addresses("SEI_WALLET")
        }
        
        # Portfolio state tracking
        self.portfolio_history = []
        self.last_analysis = None
        
        print("🔗 Multi-Chain Portfolio Analyzer initialized")
        print(f"   - Monitoring {len(self.wallet_addresses.get('sui', []))} SUI wallets")
        print(f"   - Monitoring {len(self.wallet_addresses.get('solana', []))} Solana wallets")
        print(f"   - Monitoring {len(self.wallet_addresses.get('ethereum', []))} Ethereum wallets")
        print(f"   - Monitoring {len(self.wallet_addresses.get('base', []))} Base wallets")

    def _get_wallet_addresses(self, prefix: str) -> List[str]:
        """Get wallet addresses from environment variables"""
        import os
        wallets = []
        i = 1
        while True:
            wallet = os.getenv(f"{prefix}_{i}")
            if wallet:
                wallets.append(wallet)
                i += 1
            else:
                break
        
        # If no environment wallets found, return empty list (users need to configure)
        if not wallets:
            print(f"⚠️  No {prefix} wallet addresses configured. Set {prefix}_1, {prefix}_2, etc. in .env file")
        
        return wallets
        print(f"   - Monitoring {len(self.wallet_addresses['sei'])} Sei wallets")
    
    def rpc_request(self, url: str, method: str, params: List = None) -> Dict:
        """Make a generic RPC request"""
        if params is None:
            params = []
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"RPC request failed: {response.status_code}")
        except Exception as e:
            print(f"❌ RPC Error for {method}: {e}")
            return {"error": str(e)}
    
    def get_token_price(self, token_symbol: str) -> float:
        """Get current token price from data fetcher"""
        try:
            price, _, _ = self.data_fetcher.fetch_price_and_market_cap(token_symbol)
            return price if price is not None else 0.0
        except Exception as e:
            print(f"❌ Price fetch error for {token_symbol}: {e}")
            return 0.0
    
    def analyze_sui_wallets(self) -> List[WalletBalance]:
        """Analyze all SUI wallets"""
        sui_balances = []
        
        for wallet_address in self.wallet_addresses['sui']:
            try:
                # Get SUI balance
                result = self.rpc_request(self.sui_rpc, "sui_getBalance", [wallet_address])
                
                if "result" in result:
                    balance_data = result["result"]
                    sui_balance = int(balance_data.get("totalBalance", "0")) / 1_000_000_000  # Convert from MIST to SUI
                    
                    if sui_balance > 0:
                        sui_price = self.get_token_price("SUI")
                        sui_balances.append(WalletBalance(
                            token_symbol="SUI",
                            token_address="0x2::sui::SUI",
                            balance=sui_balance,
                            usd_value=sui_balance * sui_price,
                            chain="SUI",
                            wallet_address=wallet_address,
                            timestamp=datetime.now()
                        ))
                
                # Get all coin balances (tokens)
                coins_result = self.rpc_request(self.sui_rpc, "sui_getAllBalances", [wallet_address])
                
                if "result" in coins_result:
                    for coin_data in coins_result["result"]:
                        coin_type = coin_data.get("coinType", "")
                        balance = int(coin_data.get("totalBalance", "0"))
                        
                        if balance > 0 and coin_type != "0x2::sui::SUI":
                            # Try to identify token symbol from coin type
                            token_symbol = self.identify_token_symbol(coin_type)
                            token_price = self.get_token_price(token_symbol)
                            
                            # Convert balance based on token decimals (assuming 6 decimals for most tokens)
                            decimal_balance = balance / 1_000_000
                            
                            sui_balances.append(WalletBalance(
                                token_symbol=token_symbol,
                                token_address=coin_type,
                                balance=decimal_balance,
                                usd_value=decimal_balance * token_price,
                                chain="SUI",
                                wallet_address=wallet_address,
                                timestamp=datetime.now()
                            ))
                
            except Exception as e:
                print(f"❌ Error analyzing SUI wallet {wallet_address}: {e}")
        
        return sui_balances
    
    def analyze_solana_wallets(self) -> List[WalletBalance]:
        """Analyze all Solana wallets"""
        solana_balances = []
        
        for wallet_address in self.wallet_addresses['solana']:
            try:
                # Get SOL balance
                result = self.rpc_request(self.solana_rpc, "getBalance", [wallet_address])
                
                if "result" in result:
                    lamports = result["result"]["value"]
                    sol_balance = lamports / 1_000_000_000  # Convert lamports to SOL
                    
                    if sol_balance > 0:
                        sol_price = self.get_token_price("SOL")
                        solana_balances.append(WalletBalance(
                            token_symbol="SOL",
                            token_address="11111111111111111111111111111112",
                            balance=sol_balance,
                            usd_value=sol_balance * sol_price,
                            chain="SOLANA",
                            wallet_address=wallet_address,
                            timestamp=datetime.now()
                        ))
                
                # Get token accounts
                token_accounts = self.rpc_request(self.solana_rpc, "getTokenAccountsByOwner", [
                    wallet_address,
                    {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"},
                    {"encoding": "jsonParsed"}
                ])
                
                if "result" in token_accounts:
                    for account in token_accounts["result"]["value"]:
                        token_data = account["account"]["data"]["parsed"]["info"]
                        token_amount = float(token_data.get("tokenAmount", {}).get("uiAmount", 0))
                        mint = token_data.get("mint", "")
                        
                        if token_amount > 0:
                            # Try to identify token symbol from mint address
                            token_symbol = self.identify_solana_token(mint)
                            token_price = self.get_token_price(token_symbol)
                            
                            solana_balances.append(WalletBalance(
                                token_symbol=token_symbol,
                                token_address=mint,
                                balance=token_amount,
                                usd_value=token_amount * token_price,
                                chain="SOLANA",
                                wallet_address=wallet_address,
                                timestamp=datetime.now()
                            ))
                
            except Exception as e:
                print(f"❌ Error analyzing Solana wallet {wallet_address}: {e}")
        
        return solana_balances
    
    def analyze_ethereum_wallets(self) -> List[WalletBalance]:
        """Analyze Ethereum wallets"""
        eth_balances = []
        
        for wallet_address in self.wallet_addresses['ethereum']:
            try:
                # Get ETH balance
                result = self.rpc_request(self.ethereum_rpc, "eth_getBalance", [wallet_address, "latest"])
                
                if "result" in result:
                    wei_balance = int(result["result"], 16)
                    eth_balance = wei_balance / 1_000_000_000_000_000_000  # Convert wei to ETH
                    
                    if eth_balance > 0:
                        eth_price = self.get_token_price("ETH")
                        eth_balances.append(WalletBalance(
                            token_symbol="ETH",
                            token_address="0x0000000000000000000000000000000000000000",
                            balance=eth_balance,
                            usd_value=eth_balance * eth_price,
                            chain="ETHEREUM",
                            wallet_address=wallet_address,
                            timestamp=datetime.now()
                        ))
                
                # Note: For ERC-20 tokens, we would need to call specific contract methods
                # This would require knowing the contract addresses and implementing ERC-20 calls
                
            except Exception as e:
                print(f"❌ Error analyzing Ethereum wallet {wallet_address}: {e}")
        
        return eth_balances
    
    def analyze_base_wallets(self) -> List[WalletBalance]:
        """Analyze Base wallets"""
        base_balances = []
        
        for wallet_address in self.wallet_addresses['base']:
            try:
                # Get ETH balance on Base
                result = self.rpc_request(self.base_rpc, "eth_getBalance", [wallet_address, "latest"])
                
                if "result" in result:
                    wei_balance = int(result["result"], 16)
                    eth_balance = wei_balance / 1_000_000_000_000_000_000  # Convert wei to ETH
                    
                    if eth_balance > 0:
                        eth_price = self.get_token_price("ETH")
                        base_balances.append(WalletBalance(
                            token_symbol="ETH",
                            token_address="0x0000000000000000000000000000000000000000",
                            balance=eth_balance,
                            usd_value=eth_balance * eth_price,
                            chain="BASE",
                            wallet_address=wallet_address,
                            timestamp=datetime.now()
                        ))
                
            except Exception as e:
                print(f"❌ Error analyzing Base wallet {wallet_address}: {e}")
        
        return base_balances
    
    def analyze_sei_wallets(self) -> List[WalletBalance]:
        """Analyze Sei wallets"""
        sei_balances = []
        
        for wallet_address in self.wallet_addresses['sei']:
            try:
                # Get SEI balance
                result = self.rpc_request(self.sei_rpc, "sei_getBalance", [wallet_address])
                
                if "result" in result:
                    sei_balance = float(result["result"].get("balance", "0"))
                    
                    if sei_balance > 0:
                        sei_price = self.get_token_price("SEI")
                        sei_balances.append(WalletBalance(
                            token_symbol="SEI",
                            token_address="usei",
                            balance=sei_balance,
                            usd_value=sei_balance * sei_price,
                            chain="SEI",
                            wallet_address=wallet_address,
                            timestamp=datetime.now()
                        ))
                
            except Exception as e:
                print(f"❌ Error analyzing Sei wallet {wallet_address}: {e}")
        
        return sei_balances
    
    def identify_token_symbol(self, coin_type: str) -> str:
        """Identify token symbol from SUI coin type"""
        # This is a simplified version - in production, you'd have a comprehensive mapping
        if "usdc" in coin_type.lower():
            return "USDC"
        elif "usdt" in coin_type.lower():
            return "USDT"
        elif "weth" in coin_type.lower():
            return "WETH"
        else:
            return coin_type.split("::")[-1].upper()[:10]  # Extract last part as symbol
    
    def identify_solana_token(self, mint_address: str) -> str:
        """Identify token symbol from Solana mint address"""
        # Common Solana token mint addresses
        token_map = {
            "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v": "USDC",
            "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB": "USDT",
            "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt": "SRM",
            "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E": "BTC",
            "2FPyTwcZLUg1MDrwsyoP4D6s1tM7hAkHYRjkNb5w6Pxk": "ETH",
        }
        
        return token_map.get(mint_address, mint_address[:8].upper())
    
    def calculate_portfolio_metrics(self, all_balances: List[WalletBalance]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        if not all_balances:
            return {"total_value": 0, "error": "No balances found"}
        
        # Calculate total portfolio value
        total_value = sum(balance.usd_value for balance in all_balances)
        
        # Group by chain
        chain_analysis = {}
        for balance in all_balances:
            chain = balance.chain
            if chain not in chain_analysis:
                chain_analysis[chain] = {
                    "total_value": 0,
                    "balances": [],
                    "token_count": 0
                }
            
            chain_analysis[chain]["total_value"] += balance.usd_value
            chain_analysis[chain]["balances"].append(balance)
            chain_analysis[chain]["token_count"] += 1
        
        # Calculate allocations
        allocations = {}
        for chain, data in chain_analysis.items():
            allocations[chain] = (data["total_value"] / total_value) * 100 if total_value > 0 else 0
        
        # Group by token
        token_analysis = {}
        for balance in all_balances:
            token = balance.token_symbol
            if token not in token_analysis:
                token_analysis[token] = {
                    "total_value": 0,
                    "total_balance": 0,
                    "chains": []
                }
            
            token_analysis[token]["total_value"] += balance.usd_value
            token_analysis[token]["total_balance"] += balance.balance
            token_analysis[token]["chains"].append(balance.chain)
        
        # Calculate token allocations
        token_allocations = {}
        for token, data in token_analysis.items():
            token_allocations[token] = (data["total_value"] / total_value) * 100 if total_value > 0 else 0
        
        # Calculate diversification metrics
        diversification_score = self.calculate_diversification_score(token_allocations)
        risk_level = self.assess_risk_level(allocations, token_allocations)
        
        return {
            "total_value": total_value,
            "chain_allocations": allocations,
            "token_allocations": token_allocations,
            "chain_analysis": chain_analysis,
            "token_analysis": token_analysis,
            "diversification_score": diversification_score,
            "risk_level": risk_level,
            "total_tokens": len(token_analysis),
            "total_chains": len(chain_analysis),
            "analysis_timestamp": datetime.now()
        }
    
    def calculate_diversification_score(self, token_allocations: Dict[str, float]) -> float:
        """Calculate portfolio diversification score (0-100)"""
        if not token_allocations:
            return 0
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = sum((allocation / 100) ** 2 for allocation in token_allocations.values())
        
        # Convert to diversification score (lower HHI = higher diversification)
        diversification_score = (1 - hhi) * 100
        
        return min(100, max(0, diversification_score))
    
    def assess_risk_level(self, chain_allocations: Dict[str, float], token_allocations: Dict[str, float]) -> str:
        """Assess overall portfolio risk level"""
        # Check for concentration risk
        max_chain_allocation = max(chain_allocations.values()) if chain_allocations else 0
        max_token_allocation = max(token_allocations.values()) if token_allocations else 0
        
        if max_chain_allocation > 80 or max_token_allocation > 60:
            return "HIGH"
        elif max_chain_allocation > 60 or max_token_allocation > 40:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_rebalancing_recommendations(self, portfolio_metrics: Dict[str, Any]) -> List[PortfolioRecommendation]:
        """Generate AI-powered rebalancing recommendations"""
        try:
            # Prepare analysis prompt
            prompt = f"""
            Analyze this multi-chain crypto portfolio and provide rebalancing recommendations:
            
            Portfolio Summary:
            - Total Value: ${portfolio_metrics['total_value']:,.2f}
            - Risk Level: {portfolio_metrics['risk_level']}
            - Diversification Score: {portfolio_metrics['diversification_score']:.1f}/100
            - Total Tokens: {portfolio_metrics['total_tokens']}
            - Total Chains: {portfolio_metrics['total_chains']}
            
            Chain Allocations:
            {json.dumps(portfolio_metrics['chain_allocations'], indent=2)}
            
            Token Allocations:
            {json.dumps(portfolio_metrics['token_allocations'], indent=2)}
            
            Current Market Conditions:
            {self.get_current_market_context()}
            
            Provide 3-5 specific rebalancing recommendations focusing on:
            1. Risk reduction through diversification
            2. Optimal chain allocation based on current trends
            3. Token-specific opportunities
            4. Position sizing adjustments
            
            Format each recommendation as:
            Action: [BUY/SELL/REBALANCE]
            Token: [Token Symbol]
            Current Allocation: [%]
            Target Allocation: [%]
            Reasoning: [Detailed explanation]
            Confidence: [0-1]
            Priority: [HIGH/MEDIUM/LOW]
            """
            
            completion = self.grok_client.chat.completions.create(
                model="grok-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3
            )
            
            ai_response = completion.choices[0].message.content
            
            # Parse AI response into structured recommendations
            recommendations = self.parse_ai_recommendations(ai_response, portfolio_metrics)
            
            return recommendations
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return []
    
    def parse_ai_recommendations(self, ai_response: str, portfolio_metrics: Dict[str, Any]) -> List[PortfolioRecommendation]:
        """Parse AI response into structured recommendations"""
        recommendations = []
        
        # This is a simplified parser - in production, you'd use more sophisticated NLP
        lines = ai_response.split('\n')
        current_rec = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith('Action:'):
                current_rec['action'] = line.split(':', 1)[1].strip()
            elif line.startswith('Token:'):
                current_rec['token'] = line.split(':', 1)[1].strip()
            elif line.startswith('Current Allocation:'):
                try:
                    current_rec['current_allocation'] = float(line.split(':', 1)[1].strip().replace('%', ''))
                except:
                    current_rec['current_allocation'] = 0
            elif line.startswith('Target Allocation:'):
                try:
                    current_rec['target_allocation'] = float(line.split(':', 1)[1].strip().replace('%', ''))
                except:
                    current_rec['target_allocation'] = 0
            elif line.startswith('Reasoning:'):
                current_rec['reasoning'] = line.split(':', 1)[1].strip()
            elif line.startswith('Confidence:'):
                try:
                    current_rec['confidence'] = float(line.split(':', 1)[1].strip())
                except:
                    current_rec['confidence'] = 0.5
            elif line.startswith('Priority:'):
                current_rec['priority'] = line.split(':', 1)[1].strip()
                
                # Complete recommendation
                if all(key in current_rec for key in ['action', 'token', 'reasoning']):
                    recommendations.append(PortfolioRecommendation(
                        action=current_rec['action'],
                        token=current_rec['token'],
                        current_allocation=current_rec.get('current_allocation', 0),
                        target_allocation=current_rec.get('target_allocation', 0),
                        reasoning=current_rec['reasoning'],
                        confidence=current_rec.get('confidence', 0.5),
                        priority=current_rec.get('priority', 'MEDIUM'),
                        estimated_impact=abs(current_rec.get('target_allocation', 0) - current_rec.get('current_allocation', 0))
                    ))
                
                current_rec = {}
        
        return recommendations
    
    def get_current_market_context(self) -> str:
        """Get current market context for analysis"""
        try:
            market_data = self.data_fetcher.gather_data() if hasattr(self.data_fetcher, 'gather_data') else {}
            
            context = f"""
            Current Market Prices:
            - BTC: ${self.get_token_price('BTC'):,.2f}
            - ETH: ${self.get_token_price('ETH'):,.2f}
            - SOL: ${self.get_token_price('SOL'):,.2f}
            - SUI: ${self.get_token_price('SUI'):,.2f}
            - SEI: ${self.get_token_price('SEI'):,.2f}
            
            Market Sentiment: Analyzing current trends...
            """
            
            return context
            
        except Exception as e:
            return f"Market context unavailable: {e}"
    
    def analyze_full_portfolio(self) -> Dict[str, Any]:
        """Perform comprehensive portfolio analysis"""
        print("\n🔍 Starting Multi-Chain Portfolio Analysis...")
        print("=" * 60)
        
        # Collect all balances
        all_balances = []
        
        print("📊 Analyzing SUI wallets...")
        sui_balances = self.analyze_sui_wallets()
        all_balances.extend(sui_balances)
        
        print("📊 Analyzing Solana wallets...")
        solana_balances = self.analyze_solana_wallets()
        all_balances.extend(solana_balances)
        
        print("📊 Analyzing Ethereum wallets...")
        ethereum_balances = self.analyze_ethereum_wallets()
        all_balances.extend(ethereum_balances)
        
        print("📊 Analyzing Base wallets...")
        base_balances = self.analyze_base_wallets()
        all_balances.extend(base_balances)
        
        print("📊 Analyzing Sei wallets...")
        sei_balances = self.analyze_sei_wallets()
        all_balances.extend(sei_balances)
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(all_balances)
        
        # Generate recommendations
        recommendations = self.generate_rebalancing_recommendations(portfolio_metrics)
        
        # Create comprehensive analysis
        analysis = {
            "balances": all_balances,
            "metrics": portfolio_metrics,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now()
        }
        
        self.last_analysis = analysis
        self.portfolio_history.append(analysis)
        
        return analysis
    
    def print_portfolio_report(self, analysis: Dict[str, Any] = None):
        """Print comprehensive portfolio report"""
        if analysis is None:
            analysis = self.last_analysis
        
        if not analysis:
            print("❌ No portfolio analysis available")
            return
        
        metrics = analysis["metrics"]
        recommendations = analysis["recommendations"]
        
        print(f"\n💰 PORTFOLIO SUMMARY")
        print("=" * 60)
        print(f"Total Portfolio Value: ${metrics['total_value']:,.2f}")
        print(f"Risk Level: {metrics['risk_level']}")
        print(f"Diversification Score: {metrics['diversification_score']:.1f}/100")
        print(f"Total Tokens: {metrics['total_tokens']}")
        print(f"Total Chains: {metrics['total_chains']}")
        
        print(f"\n🔗 CHAIN ALLOCATION")
        print("-" * 40)
        for chain, allocation in sorted(metrics['chain_allocations'].items(), key=lambda x: x[1], reverse=True):
            print(f"{chain}: {allocation:.1f}% (${metrics['chain_analysis'][chain]['total_value']:,.2f})")
        
        print(f"\n🪙 TOKEN ALLOCATION")
        print("-" * 40)
        for token, allocation in sorted(metrics['token_allocations'].items(), key=lambda x: x[1], reverse=True):
            if allocation > 0.1:  # Only show tokens with >0.1% allocation
                print(f"{token}: {allocation:.1f}% (${metrics['token_analysis'][token]['total_value']:,.2f})")
        
        print(f"\n💡 REBALANCING RECOMMENDATIONS")
        print("-" * 40)
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                emoji = priority_emoji.get(rec.priority, "🔵")
                
                print(f"{i}. {emoji} {rec.action} {rec.token}")
                print(f"   Current: {rec.current_allocation:.1f}% → Target: {rec.target_allocation:.1f}%")
                print(f"   Confidence: {rec.confidence:.1%}")
                print(f"   Reasoning: {rec.reasoning}")
                print()
        else:
            print("No specific recommendations at this time.")
        
        print("=" * 60) 