import pandas as pd

def calculate_momentum(price_series):
    returns = price_series.pct_change().dropna()
    return returns.mean()

def calculate_risk_reward(current_price, predicted_price, risk_tolerance):
    potential_gain = abs(predicted_price - current_price) if predicted_price > current_price else 0
    potential_loss = current_price * risk_tolerance
    return potential_gain / potential_loss if potential_loss > 0 else 0

def plot_price_chart(price_series, prediction, output_path):
    """Plot historical prices with next-step prediction."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    price_series.plot(label="Price")
    if prediction is not None:
        plt.scatter(price_series.index[-1] + pd.Timedelta(1, unit="d"), prediction, color="red", label="Prediction")
    plt.title("Price Chart")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def sanitize_filename(name):
    """Replace characters that would break file paths."""
    return name.replace("/", "_")

