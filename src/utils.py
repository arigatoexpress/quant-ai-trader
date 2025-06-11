import pandas as pd

def calculate_momentum(price_series):
    returns = price_series.pct_change().dropna()
    return returns.mean()

def calculate_risk_reward(current_price, predicted_price, risk_tolerance):
    potential_gain = abs(predicted_price - current_price) if predicted_price > current_price else 0
    potential_loss = current_price * risk_tolerance
    return potential_gain / potential_loss if potential_loss > 0 else 0

