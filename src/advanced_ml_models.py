"""
Advanced Machine Learning Models for Trading
Implements state-of-the-art ML models for market prediction and signal generation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Optional, Any, Tuple
import joblib
import warnings
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass
import ta
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

@dataclass
class ModelPrediction:
    """Model prediction with confidence metrics"""
    asset: str
    prediction: float
    confidence: float
    model_type: str
    timestamp: datetime
    features_used: List[str]
    probability_distribution: Optional[Dict[str, float]] = None
    supporting_metrics: Optional[Dict[str, Any]] = None

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    mse: float
    mae: float
    r2: float
    sharpe_ratio: float
    hit_rate: float
    profit_factor: float
    max_drawdown: float
    volatility: float
    total_return: float
    annual_return: float
    validation_score: float

class LSTMPricePredictor(nn.Module):
    """LSTM neural network for price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 1):
        super(LSTMPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output
        last_output = attn_output[:, -1, :]
        
        # Apply dropout and linear layer
        output = self.dropout(last_output)
        prediction = self.linear(output)
        
        return prediction

class TransformerPredictor(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1):
        super(TransformerPredictor, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._create_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Apply transformer
        transformer_out = self.transformer(x)
        
        # Use last output for prediction
        last_output = transformer_out[:, -1, :]
        prediction = self.output_projection(self.dropout(last_output))
        
        return prediction

class AdvancedMLModels:
    """Advanced ML models for trading prediction"""
    
    def __init__(self, config_path: str = None):
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                import yaml
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_performances = {}
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🧠 Advanced ML Models initialized")
        print(f"   Device: {self.device}")
        print(f"   Models to train: {len(self.config['models'])}")
        
    def _default_config(self):
        return {
            'models': [
                'lstm', 'transformer', 'xgboost', 'lightgbm', 
                'random_forest', 'gradient_boosting'
            ],
            'sequence_length': 60,
            'prediction_horizon': 1,
            'train_split': 0.8,
            'validation_split': 0.1,
            'features': [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'bollinger_bands', 'stochastic',
                'ema_9', 'ema_21', 'ema_50', 'sma_200',
                'price_momentum', 'volume_momentum',
                'volatility', 'returns'
            ],
            'hyperparameters': {
                'lstm': {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2},
                'transformer': {'d_model': 128, 'nhead': 8, 'num_layers': 6},
                'xgboost': {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.01},
                'lightgbm': {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.01}
            }
        }
    
    def fetch_market_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch market data with technical indicators"""
        try:
            # Fetch price data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Add momentum features
            data = self._add_momentum_features(data)
            
            # Clean data
            data = data.dropna()
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data"""
        # RSI
        data['rsi'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['Close'])
        data['bb_upper'] = bb.bollinger_hband()
        data['bb_lower'] = bb.bollinger_lband()
        data['bb_middle'] = bb.bollinger_mavg()
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'])
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
        
        # Moving Averages
        data['ema_9'] = ta.trend.EMAIndicator(data['Close'], window=9).ema_indicator()
        data['ema_21'] = ta.trend.EMAIndicator(data['Close'], window=21).ema_indicator()
        data['ema_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
        data['sma_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        
        # ATR (Average True Range)
        data['atr'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close']).average_true_range()
        
                # Volume indicators
        try:
            data['volume_sma'] = ta.volume.VolumeSMAIndicator(close=data['Close'], volume=data['Volume']).volume_sma()
        except:
            # Fallback if VolumeSMAIndicator doesn't work
            data['volume_sma'] = data['Volume'].rolling(window=20).mean()
        
        try:
            data['volume_weighted_price'] = ta.volume.VolumWeightedAveragePrice(
                high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume']
            ).volume_weighted_average_price()
        except:
            # Fallback VWAP calculation
            data['volume_weighted_price'] = (data['Close'] * data['Volume']).rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()
        
        return data
    
    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and volatility features"""
        # Price momentum
        data['price_momentum_5'] = data['Close'].pct_change(5)
        data['price_momentum_10'] = data['Close'].pct_change(10)
        data['price_momentum_20'] = data['Close'].pct_change(20)
        
        # Volume momentum
        data['volume_momentum'] = data['Volume'].pct_change(5)
        
        # Volatility
        data['volatility_10'] = data['Close'].rolling(window=10).std()
        data['volatility_20'] = data['Close'].rolling(window=20).std()
        
        # Returns
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Price position relative to high/low
        data['price_position'] = (data['Close'] - data['Low'].rolling(20).min()) / (data['High'].rolling(20).max() - data['Low'].rolling(20).min())
        
        return data
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for model training"""
        # Select relevant features
        feature_columns = [col for col in self.config['features'] if col in data.columns]
        
        # Handle missing feature columns
        missing_cols = [col for col in self.config['features'] if col not in data.columns]
        if missing_cols:
            print(f"⚠️  Missing features: {missing_cols}")
            # Map alternative column names
            feature_columns = []
            for col in self.config['features']:
                if col in data.columns:
                    feature_columns.append(col)
                elif col == 'close' and 'Close' in data.columns:
                    feature_columns.append('Close')
                elif col == 'open' and 'Open' in data.columns:
                    feature_columns.append('Open')
                elif col == 'high' and 'High' in data.columns:
                    feature_columns.append('High')
                elif col == 'low' and 'Low' in data.columns:
                    feature_columns.append('Low')
                elif col == 'volume' and 'Volume' in data.columns:
                    feature_columns.append('Volume')
        
        self.feature_columns = feature_columns
        
        # Create feature matrix
        X = data[feature_columns].values
        
        # Create target (next period close price)
        y = data['Close'].shift(-self.config['prediction_horizon']).values
        
        # Remove NaN values
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/Transformer models"""
        seq_length = self.config['sequence_length']
        
        X_seq = []
        y_seq = []
        
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> nn.Module:
        """Train LSTM model"""
        print("🧠 Training LSTM model...")
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)
        
        # Create model
        model = LSTMPricePredictor(
            input_size=X_train_seq.shape[2],
            **self.config['hyperparameters']['lstm']
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        epochs = 100
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            
            # Mini-batch training
            batch_size = 32
            num_batches = len(X_train_tensor) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_X = X_train_tensor[start_idx:end_idx]
                batch_y = y_train_tensor[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Train Loss {train_loss/num_batches:.6f}, Val Loss {val_loss:.6f}")
        
        # Load best model
        model.load_state_dict(torch.load('best_lstm_model.pth'))
        
        return model
    
    def train_transformer_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: np.ndarray, y_val: np.ndarray) -> nn.Module:
        """Train Transformer model"""
        print("🧠 Training Transformer model...")
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)
        
        # Create model
        model = TransformerPredictor(
            input_size=X_train_seq.shape[2],
            **self.config['hyperparameters']['transformer']
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop (similar to LSTM)
        epochs = 100
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            batch_size = 32
            num_batches = len(X_train_tensor) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                batch_X = X_train_tensor[start_idx:end_idx]
                batch_y = y_train_tensor[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_transformer_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: Train Loss {train_loss/num_batches:.6f}, Val Loss {val_loss:.6f}")
        
        model.load_state_dict(torch.load('best_transformer_model.pth'))
        return model
    
    def train_xgboost_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        print("🧠 Training XGBoost model...")
        
        model = xgb.XGBRegressor(
            **self.config['hyperparameters']['xgboost'],
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return model
    
    def train_lightgbm_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> lgb.LGBMRegressor:
        """Train LightGBM model"""
        print("🧠 Training LightGBM model...")
        
        model = lgb.LGBMRegressor(
            **self.config['hyperparameters']['lightgbm'],
            random_state=42,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return model
    
    def train_all_models(self, symbol: str) -> Dict[str, Any]:
        """Train all models for a given symbol"""
        print(f"🚀 Training models for {symbol}")
        
        # Fetch data
        data = self.fetch_market_data(symbol)
        if data.empty:
            print(f"❌ No data available for {symbol}")
            return {}
        
        # Prepare features
        X, y = self.prepare_features(data)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[symbol] = scaler
        
        # Split data
        train_size = int(len(X_scaled) * self.config['train_split'])
        val_size = int(len(X_scaled) * self.config['validation_split'])
        
        X_train = X_scaled[:train_size]
        y_train = y[:train_size]
        X_val = X_scaled[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X_scaled[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # Train models
        models = {}
        
        if 'lstm' in self.config['models']:
            models['lstm'] = self.train_lstm_model(X_train, y_train, X_val, y_val)
        
        if 'transformer' in self.config['models']:
            models['transformer'] = self.train_transformer_model(X_train, y_train, X_val, y_val)
        
        if 'xgboost' in self.config['models']:
            models['xgboost'] = self.train_xgboost_model(X_train, y_train, X_val, y_val)
        
        if 'lightgbm' in self.config['models']:
            models['lightgbm'] = self.train_lightgbm_model(X_train, y_train, X_val, y_val)
        
        if 'random_forest' in self.config['models']:
            print("🧠 Training Random Forest model...")
            models['random_forest'] = RandomForestRegressor(
                n_estimators=500, max_depth=10, random_state=42
            ).fit(X_train, y_train)
        
        if 'gradient_boosting' in self.config['models']:
            print("🧠 Training Gradient Boosting model...")
            models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=500, max_depth=6, random_state=42
            ).fit(X_train, y_train)
        
        self.models[symbol] = models
        
        # Evaluate models
        performances = {}
        for model_name, model in models.items():
            performance = self.evaluate_model(model, model_name, X_test, y_test, symbol)
            performances[model_name] = performance
        
        self.model_performances[symbol] = performances
        
        return models
    
    def evaluate_model(self, model: Any, model_name: str, X_test: np.ndarray, 
                      y_test: np.ndarray, symbol: str) -> ModelPerformance:
        """Evaluate model performance"""
        try:
            # Make predictions
            if model_name in ['lstm', 'transformer']:
                X_test_seq, y_test_seq = self.create_sequences(X_test, y_test)
                X_test_tensor = torch.FloatTensor(X_test_seq).to(self.device)
                
                model.eval()
                with torch.no_grad():
                    predictions = model(X_test_tensor).cpu().numpy().flatten()
                
                y_true = y_test_seq
            else:
                predictions = model.predict(X_test)
                y_true = y_test
            
            # Calculate metrics
            mse = mean_squared_error(y_true, predictions)
            mae = mean_absolute_error(y_true, predictions)
            r2 = r2_score(y_true, predictions)
            
            # Financial metrics
            returns = np.diff(y_true) / y_true[:-1]
            pred_returns = np.diff(predictions) / predictions[:-1]
            
            # Align returns
            min_len = min(len(returns), len(pred_returns))
            returns = returns[:min_len]
            pred_returns = pred_returns[:min_len]
            
            # Hit rate (directional accuracy)
            hit_rate = np.mean(np.sign(returns) == np.sign(pred_returns))
            
            # Sharpe ratio
            sharpe_ratio = np.mean(pred_returns) / np.std(pred_returns) if np.std(pred_returns) > 0 else 0
            
            # Volatility
            volatility = np.std(pred_returns)
            
            # Total return
            total_return = np.prod(1 + pred_returns) - 1
            
            # Annual return (assuming daily data)
            annual_return = (1 + total_return) ** (252 / len(pred_returns)) - 1
            
            # Max drawdown
            cumulative_returns = np.cumprod(1 + pred_returns)
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = np.min(drawdown)
            
            # Profit factor
            positive_returns = pred_returns[pred_returns > 0]
            negative_returns = pred_returns[pred_returns < 0]
            
            profit_factor = (np.sum(positive_returns) / abs(np.sum(negative_returns))) if len(negative_returns) > 0 else float('inf')
            
            # Validation score (custom metric)
            validation_score = (hit_rate * 0.3) + (r2 * 0.3) + (sharpe_ratio / 5 * 0.2) + ((1 - abs(max_drawdown)) * 0.2)
            
            performance = ModelPerformance(
                model_name=model_name,
                mse=mse,
                mae=mae,
                r2=r2,
                sharpe_ratio=sharpe_ratio,
                hit_rate=hit_rate,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                volatility=volatility,
                total_return=total_return,
                annual_return=annual_return,
                validation_score=validation_score
            )
            
            print(f"   {model_name}: R² {r2:.3f}, Hit Rate {hit_rate:.3f}, Sharpe {sharpe_ratio:.3f}")
            
            return performance
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            return ModelPerformance(
                model_name=model_name,
                mse=float('inf'), mae=float('inf'), r2=0,
                sharpe_ratio=0, hit_rate=0, profit_factor=0,
                max_drawdown=-1, volatility=0, total_return=0,
                annual_return=0, validation_score=0
            )
    
    def predict(self, symbol: str, current_data: pd.DataFrame) -> List[ModelPrediction]:
        """Make predictions using all trained models"""
        if symbol not in self.models:
            print(f"❌ No models trained for {symbol}")
            return []
        
        predictions = []
        
        # Prepare current features
        current_data = self._add_technical_indicators(current_data)
        current_data = self._add_momentum_features(current_data)
        
        X_current, _ = self.prepare_features(current_data)
        X_current_scaled = self.scalers[symbol].transform(X_current[-1:])
        
        for model_name, model in self.models[symbol].items():
            try:
                if model_name in ['lstm', 'transformer']:
                    # Need sequence for neural models
                    if len(X_current_scaled) >= self.config['sequence_length']:
                        X_seq = X_current_scaled[-self.config['sequence_length']:].reshape(1, -1, len(self.feature_columns))
                        X_tensor = torch.FloatTensor(X_seq).to(self.device)
                        
                        model.eval()
                        with torch.no_grad():
                            prediction = model(X_tensor).cpu().numpy()[0][0]
                    else:
                        continue
                else:
                    prediction = model.predict(X_current_scaled)[0]
                
                # Calculate confidence based on model performance
                performance = self.model_performances[symbol][model_name]
                confidence = performance.validation_score
                
                predictions.append(ModelPrediction(
                    asset=symbol,
                    prediction=prediction,
                    confidence=confidence,
                    model_type=model_name,
                    timestamp=datetime.now(),
                    features_used=self.feature_columns,
                    supporting_metrics={
                        'r2': performance.r2,
                        'hit_rate': performance.hit_rate,
                        'sharpe_ratio': performance.sharpe_ratio
                    }
                ))
                
            except Exception as e:
                print(f"❌ Error in {model_name} prediction: {e}")
                continue
        
        return predictions
    
    def get_ensemble_prediction(self, symbol: str, current_data: pd.DataFrame) -> Optional[ModelPrediction]:
        """Get ensemble prediction from all models"""
        predictions = self.predict(symbol, current_data)
        
        if not predictions:
            return None
        
        # Weighted average based on confidence
        total_weight = sum(p.confidence for p in predictions)
        if total_weight == 0:
            return None
        
        ensemble_prediction = sum(p.prediction * p.confidence for p in predictions) / total_weight
        ensemble_confidence = total_weight / len(predictions)
        
        return ModelPrediction(
            asset=symbol,
            prediction=ensemble_prediction,
            confidence=ensemble_confidence,
            model_type='ensemble',
            timestamp=datetime.now(),
            features_used=self.feature_columns,
            supporting_metrics={
                'individual_predictions': len(predictions),
                'best_model': max(predictions, key=lambda x: x.confidence).model_type,
                'prediction_spread': max(p.prediction for p in predictions) - min(p.prediction for p in predictions)
            }
        )
    
    def save_models(self, symbol: str, save_dir: str = 'models'):
        """Save trained models"""
        if symbol not in self.models:
            print(f"❌ No models to save for {symbol}")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models[symbol].items():
            try:
                if model_name in ['lstm', 'transformer']:
                    torch.save(model.state_dict(), f'{save_dir}/{symbol}_{model_name}.pth')
                else:
                    joblib.dump(model, f'{save_dir}/{symbol}_{model_name}.joblib')
                
                print(f"✅ Saved {model_name} model for {symbol}")
                
            except Exception as e:
                print(f"❌ Error saving {model_name}: {e}")
        
        # Save scaler
        joblib.dump(self.scalers[symbol], f'{save_dir}/{symbol}_scaler.joblib')
        
        # Save performance metrics
        with open(f'{save_dir}/{symbol}_performance.json', 'w') as f:
            performance_dict = {}
            for model_name, perf in self.model_performances[symbol].items():
                performance_dict[model_name] = {
                    'mse': perf.mse,
                    'mae': perf.mae,
                    'r2': perf.r2,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'hit_rate': perf.hit_rate,
                    'profit_factor': perf.profit_factor,
                    'max_drawdown': perf.max_drawdown,
                    'total_return': perf.total_return,
                    'annual_return': perf.annual_return,
                    'validation_score': perf.validation_score
                }
            json.dump(performance_dict, f, indent=2)
    
    def generate_model_report(self, symbol: str) -> str:
        """Generate comprehensive model performance report"""
        if symbol not in self.model_performances:
            return f"No performance data available for {symbol}"
        
        report = f"""
🧠 ADVANCED ML MODELS PERFORMANCE REPORT
========================================
Symbol: {symbol}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 MODEL PERFORMANCE COMPARISON
"""
        
        performances = self.model_performances[symbol]
        
        # Sort by validation score
        sorted_models = sorted(performances.items(), key=lambda x: x[1].validation_score, reverse=True)
        
        for model_name, perf in sorted_models:
            report += f"""
🔹 {model_name.upper()}
   📈 Validation Score: {perf.validation_score:.3f}
   🎯 Hit Rate: {perf.hit_rate:.3f} ({perf.hit_rate*100:.1f}%)
   📊 R² Score: {perf.r2:.3f}
   ⚡ Sharpe Ratio: {perf.sharpe_ratio:.3f}
   💰 Annual Return: {perf.annual_return:.3f} ({perf.annual_return*100:.1f}%)
   📉 Max Drawdown: {perf.max_drawdown:.3f} ({perf.max_drawdown*100:.1f}%)
   🔄 Profit Factor: {perf.profit_factor:.3f}
   📊 Volatility: {perf.volatility:.3f}
"""
        
        # Best model recommendation
        best_model = sorted_models[0][0]
        best_perf = sorted_models[0][1]
        
        report += f"""
🏆 BEST MODEL: {best_model.upper()}
   Overall Score: {best_perf.validation_score:.3f}
   
🎯 RECOMMENDATION:
   Use {best_model} for primary predictions
   Consider ensemble approach for robust signals
   
📊 ENSEMBLE BENEFITS:
   • Reduces model-specific bias
   • Improves prediction stability
   • Better risk-adjusted returns
   
⚠️  RISK CONSIDERATIONS:
   • Past performance doesn't guarantee future results
   • Monitor model drift and retrain regularly
   • Use proper position sizing and risk management
"""
        
        return report

def main():
    """Demo function for testing advanced ML models"""
    print("🧠 ADVANCED ML MODELS DEMO")
    print("=" * 60)
    
    # Initialize models
    ml_models = AdvancedMLModels()
    
    # Test symbols
    symbols = ['BTC-USD', 'ETH-USD', 'AAPL']
    
    for symbol in symbols:
        print(f"\n🚀 Training models for {symbol}...")
        
        # Train models
        models = ml_models.train_all_models(symbol)
        
        if models:
            print(f"✅ Successfully trained {len(models)} models for {symbol}")
            
            # Generate report
            report = ml_models.generate_model_report(symbol)
            print(report)
            
            # Save models
            ml_models.save_models(symbol)
            
            # Test prediction
            try:
                current_data = ml_models.fetch_market_data(symbol, period="1mo")
                if not current_data.empty:
                    ensemble_pred = ml_models.get_ensemble_prediction(symbol, current_data)
                    
                    if ensemble_pred:
                        print(f"\n🎯 ENSEMBLE PREDICTION:")
                        print(f"   Symbol: {ensemble_pred.asset}")
                        print(f"   Prediction: ${ensemble_pred.prediction:.2f}")
                        print(f"   Confidence: {ensemble_pred.confidence:.3f}")
                        print(f"   Model Type: {ensemble_pred.model_type}")
                        print(f"   Features Used: {len(ensemble_pred.features_used)}")
                        
                        current_price = current_data['Close'].iloc[-1]
                        price_change = (ensemble_pred.prediction - current_price) / current_price
                        print(f"   Expected Change: {price_change:.2%}")
                        
            except Exception as e:
                print(f"❌ Error testing prediction: {e}")
        else:
            print(f"❌ Failed to train models for {symbol}")
    
    print("\n" + "=" * 60)
    print("🎯 Advanced ML Models Demo Complete!")

if __name__ == "__main__":
    main() 