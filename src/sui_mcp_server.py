import yaml
import json
import requests
from mcp.server import FastMCP
from openai import OpenAI

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
sui_rpc = config['sui_rpc']
grok_api_key = config['grok_api_key']
mcp_port = config['mcp_port']

# Initialize GROK client
grok_client = OpenAI(
    api_key=grok_api_key,
    base_url="https://api.x.ai/v1",
)

# Init MCP server
server = FastMCP("sui-trading-server")

def sui_rpc_request(method, params=None):
    """Make a JSON-RPC request to SUI RPC endpoint"""
    if params is None:
        params = []
    
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params
    }
    
    response = requests.post(sui_rpc, json=payload, headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"RPC request failed: {response.status_code}")

@server.tool(name="get_sui_pool_liquidity")
def get_sui_pool_liquidity(pool_object_id):
    """Get liquidity information for a SUI pool using object ID"""
    try:
        result = sui_rpc_request("sui_getObject", [pool_object_id, {"showContent": True}])
        
        if "result" in result and "data" in result["result"]:
            content = result["result"]["data"].get("content", {})
            fields = content.get("fields", {})
            
            # Extract liquidity info from pool fields
            liquidity_info = {
                "pool_id": pool_object_id,
                "type": content.get("type", "unknown"),
                "fields": fields,
                "status": "success"
            }
            
            return liquidity_info
        else:
            return {"status": "error", "message": "Pool data not found"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@server.tool(name="get_sui_balance")
def get_sui_balance(address):
    """Get SUI balance for an address"""
    try:
        result = sui_rpc_request("sui_getBalance", [address])
        
        if "result" in result:
            balance_data = result["result"]
            return {
                "address": address,
                "balance": balance_data.get("totalBalance", "0"),
                "coin_type": balance_data.get("coinType", "0x2::sui::SUI"),
                "status": "success"
            }
        else:
            return {"status": "error", "message": "Balance data not found"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@server.tool(name="analyze_trade_with_grok")
def analyze_trade_with_grok(pool_id, liquidity_data):
    """Analyze trade opportunities using GROK"""
    try:
        prompt = f"Analyze Sui pool {pool_id} with liquidity {liquidity_data}. Recommend trade for SUI."
        
        completion = grok_client.chat.completions.create(
            model="grok-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return {"recommendation": completion.choices[0].message.content, "status": "success"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    print(f"Starting SUI MCP Server on port {mcp_port}")
    print("Available tools:")
    print("- get_sui_pool_liquidity: Get pool liquidity information")
    print("- get_sui_balance: Get SUI balance for an address")  
    print("- analyze_trade_with_grok: Get GROK analysis for trading")
    
    # Start the server
    server.run(port=mcp_port)