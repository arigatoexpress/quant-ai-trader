import requests
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

# Working Sui Mainnet RPC
SUI_RPC = "https://fullnode.mainnet.sui.io"

def sui_rpc(method, params):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = requests.post(SUI_RPC, json=payload)
    result = response.json().get('result')
    if not result:
        raise ValueError(f"RPC error: {response.json()}")
    return result

full_sail_package = "0xe1b7d5fd116fea5a8f8e85c13754248d56626a8d0a614b7d916c2348d8323149"

def get_all_pools(package_id):
    pools = {}
    query = {"MoveModule": {"package": package_id, "module": "pool_script"}}
    cursor = None
    page = 0
    max_pages = 10
    while True:
        page += 1
        if page > max_pages:
            break
        events_data = sui_rpc("suix_queryEvents", [query, cursor, 1000, False])
        events = events_data.get('data', [])
        new_pools = 0
        for e in events:
            parsed = e.get('parsedJson', {})
            pool_id = parsed.get('pool') or parsed.get('pool_id')
            if pool_id and pool_id not in pools:
                base = parsed.get('amount_a', 'Unknown') or parsed.get('amount_in', 'Unknown')
                quote = parsed.get('amount_b', 'Unknown') or parsed.get('amount_out', 'Unknown')
                pools[pool_id] = f"Pool {pool_id[:10]}... ({base}/{quote})"
                new_pools += 1
        cursor = events_data.get('nextCursor')
        if not cursor:
            break
    return pools

def get_historical_volumes(pool_address, package_id, weeks=4):
    volumes = []
    end_time = datetime.datetime.now(datetime.UTC)
    event_type = f"{package_id}::pool_script::SwapEvent"
    for w in range(weeks):
        start_time = end_time - datetime.timedelta(weeks=1)
        query = {"TimeRange": {"startTime": str(int(start_time.timestamp() * 1000)), "endTime": str(int(end_time.timestamp() * 1000))}}
        events = []
        cursor = None
        while True:
            events_data = sui_rpc("suix_queryEvents", [query, cursor, 1000, False])
            new_events = events_data.get('data', [])
            events.extend(new_events)
            cursor = events_data.get('nextCursor')
            if not cursor:
                break
        week_volume = sum(
            float(parsed.get('amount_in', 0)) + float(parsed.get('amount_out', 0))
            for e in events if e.get('type') == event_type and (parsed := e.get('parsedJson', {})) and parsed.get('pool') == pool_address
        )
        volumes.append((start_time.date(), week_volume))
        end_time = start_time
    return pd.DataFrame(volumes, columns=['week', 'volume'])

def predict_next_week(df):
    if len(df) < 2:
        return 0
    df['week_num'] = np.arange(len(df))
    model = LinearRegression()
    model.fit(df[['week_num']], df['volume'])
    next_week_num = len(df)
    prediction = model.predict([[next_week_num]])[0]
    return max(0, prediction)

def get_volume_predictions():
    pools = get_all_pools(full_sail_package)
    predictions = {}
    for addr, name in pools.items():
        df = get_historical_volumes(addr, full_sail_package)
        pred = predict_next_week(df)
        predictions[name] = pred
    return predictions