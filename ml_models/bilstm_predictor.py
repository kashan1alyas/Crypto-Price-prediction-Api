import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from config import settings
from sklearn.preprocessing import RobustScaler
import logging
import os
from datetime import datetime
import pytz
import torch.serialization
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import torch.nn.functional as F

# Allowlist datetime.datetime and numpy scalar types for safe model loading
torch.serialization.add_safe_globals([
    datetime,
    np._core.multiarray.scalar,  # Add numpy scalar type
    np.dtype,  # Add numpy dtype
    np.ndarray,  # Add numpy array type
    np.bool_,  # Add numpy bool type
    np.float64,  # Add numpy float types
    np.float32,
    np.int64,  # Add numpy int types
    np.int32
])

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Updated feature list
FEATURE_LIST = [
    # Core price data
    'Open', 'High', 'Low', 'Close', 'Volume',

    # Momentum indicators
    'RSI_14', 'Momentum', 'ADX', 'CCI', 'Stoch_%K', 'Stoch_%D', 'MFI',

    # Volatility
    'ATR', 'Volatility',

    # Volume
    'OBV', 'VWAP', 'Volume_Profile',

    # Multi-timeframe momentum
    'Momentum_5', 'Momentum_10', 'Momentum_20',

    # Market regime
    'Market_Regime', 'Volatility_Regime',

    # Time features (cyclical)
    'Hour_Sin', 'Hour_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos',
]

def load_model(symbol, timeframe, input_size=28, hidden_size=128, num_layers=2):
    """Load pre-trained model and scaler"""
    model_path = f"models/{symbol}/model_{timeframe}.pth"
    scaler_path = f"models/{symbol}/scaler_{timeframe}.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    config = checkpoint.get('model_config', checkpoint.get('config', {}))
    input_size = config.get('input_size', input_size)
    hidden_size = config.get('hidden_size', hidden_size)
    num_layers = config.get('num_layers', num_layers)
    dropout = float(config.get('dropout', 0.3))
    
    model = BiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    for key, value in state_dict.items():
        if isinstance(value, np.generic):
            state_dict[key] = value.item()
    
    # Strip torch.compile wrapper prefix if present
    state_dict = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    model.eval()
    return model, scaler

class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size: int = 28, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("Invalid model parameters")
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = float(dropout)
        
        self.input_norm = nn.LayerNorm(input_size)
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=input_size if i == 0 else hidden_size * 2,
                hidden_size=hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True
            ) for i in range(num_layers)
        ])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(p=float(self.dropout_rate)) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * 2) for _ in range(num_layers)
        ])
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.output_activation = nn.Tanh()
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Ensure float32 dtype to prevent mixed dtype errors
        x = x.to(dtype=torch.float32)
        
        x = self.input_norm(x)
        
        for i in range(self.num_layers):
            lstm_out, _ = self.lstm_layers[i](x)
            
            if i > 0 and x.size(-1) == lstm_out.size(-1):
                lstm_out = lstm_out + x
            
            lstm_out = self.layer_norms[i](lstm_out)
            lstm_out = self.dropouts[i](lstm_out)
            
            x = lstm_out
        
        attention_weights = self.attention(x)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), x)
        
        out = context_vector.squeeze(1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.dropouts[0](out)
        out = self.fc2(out)
        out = self.output_activation(out)
        
        return out
    
    @staticmethod
    def load_from_checkpoint(checkpoint_path):
        """Load model from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            
            config = checkpoint.get('model_config', checkpoint.get('config', {}))
            
            input_size = config.get('input_size', 28)
            hidden_size = config.get('hidden_size', 128)
            num_layers = config.get('num_layers', 2)
            dropout = float(config.get('dropout', 0.3))
            feature_list = config.get('feature_list', None)
            
            model = BiLSTMWithAttention(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
            
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            for key, value in state_dict.items():
                if isinstance(value, np.generic):
                    state_dict[key] = value.item()
            
            # Strip torch.compile wrapper prefix if present
            state_dict = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            
            loaded_config = {
                'input_size': int(input_size) if isinstance(input_size, np.generic) else input_size,
                'hidden_size': int(hidden_size) if isinstance(hidden_size, np.generic) else hidden_size,
                'num_layers': int(num_layers) if isinstance(num_layers, np.generic) else num_layers,
                'dropout': float(dropout) if isinstance(dropout, np.generic) else dropout,
                'feature_list': feature_list,
                'best_epoch': config.get('best_epoch'),
                'best_metrics': config.get('best_metrics')
            }
            
            return model, loaded_config
            
        except Exception as e:
            logger.error(f"Error loading checkpoint from {checkpoint_path}: {str(e)}")
            raise

def fetch_and_predict_btc_price(symbol="BTC", timeframe="1h"):
    """Fetch data and predict price for given timeframe.
    
    The model predicts percentage returns. The predicted price is derived as:
        predicted_price = last_price * (1 + predicted_return)
    """
    from controllers.prediction import get_latest_data
    try:
        if timeframe not in settings.TIMEFRAME_MAP:
            raise ValueError(f"Invalid timeframe. Must be one of {list(settings.TIMEFRAME_MAP.keys())}")
        
        lookback = settings.TIMEFRAME_MAP[timeframe]["lookback"]
        
        df = get_latest_data(symbol, timeframe)
        
        model, scaler = load_model(symbol, timeframe)
        
        features = df[FEATURE_LIST].values
        if scaler is not None:
            scaled_features = scaler.transform(features)
        else:
            scaled_features = features
        X_input = scaled_features[-lookback:].reshape(1, lookback, len(FEATURE_LIST))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            predictions = []
            for _ in range(50):
                pred = model(torch.FloatTensor(X_input).to(device))
                if pred.dim() == 3:
                    pred = pred[:, -1, :]
                elif pred.dim() == 2:
                    pass
                pred = pred.cpu().numpy()
                predictions.append(pred)
            
            predictions = np.array(predictions)
            prediction_scaled = np.mean(predictions, axis=0)
            confidence_interval = np.percentile(predictions, [5, 95], axis=0)
        
        if prediction_scaled.ndim == 3:
            prediction_scaled = prediction_scaled.squeeze()
        if prediction_scaled.ndim == 0:
            prediction_scaled = prediction_scaled.reshape(1,)
        elif prediction_scaled.ndim > 1:
            prediction_scaled = prediction_scaled.flatten()[:1]
        
        # Inverse transform to get raw predicted return
        if scaler is not None:
            n_features = len(FEATURE_LIST)
            predicted_return = scaler.inverse_transform(
                np.concatenate([prediction_scaled.reshape(1, 1), np.zeros((1, n_features - 1))], axis=1)
            )[0, 0]
            ci_lower_return = scaler.inverse_transform(
                np.concatenate([confidence_interval[0].reshape(1, 1), np.zeros((1, n_features - 1))], axis=1)
            )[0, 0]
            ci_upper_return = scaler.inverse_transform(
                np.concatenate([confidence_interval[1].reshape(1, 1), np.zeros((1, n_features - 1))], axis=1)
            )[0, 0]
        else:
            predicted_return = float(prediction_scaled[0])
            ci_lower_return = float(confidence_interval[0].flatten()[0])
            ci_upper_return = float(confidence_interval[1].flatten()[0])
        
        last_actual_price = float(df['Close'].iloc[-1])
        time_delta = pd.Timedelta(hours=1 if timeframe != "24h" else 24)
        
        # Derive predicted price from return
        predicted_price = last_actual_price * (1.0 + predicted_return)
        ci_lower_price = last_actual_price * (1.0 + ci_lower_return)
        ci_upper_price = last_actual_price * (1.0 + ci_upper_return)
        
        actuals, pred_returns = prepare_data_for_prediction(symbol, timeframe)
        if len(actuals) > 1 and len(pred_returns) > 1:
            mape = mean_absolute_error(actuals, pred_returns) / np.mean(np.abs(actuals)) * 100
        else:
            mape = 0.0
        
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "predicted_price": float(predicted_price),
            "predicted_return": float(predicted_return),
            "confidence_interval": [float(ci_lower_price), float(ci_upper_price)],
            "last_actual_price": last_actual_price,
            "prediction_time": (df.index[-1] + time_delta).isoformat(),
            "current_time": datetime.now(pytz.UTC).isoformat(),
            "rsi": float(df['RSI_14'].iloc[-1]),
            "adx": float(df['ADX'].iloc[-1]),
            "cci": float(df['CCI'].iloc[-1]),
            "price_source": "yfinance",
            "mape": float(mape)
        }
        logger.info(f"Predicted Price for {symbol} {timeframe}: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise

def prepare_data_for_prediction(symbol, timeframe="24h"):
    """Prepare data for evaluating model predictions.
    
    Returns actual percentage returns and predicted percentage returns.
    """
    from controllers.prediction import get_latest_data
    try:
        lookback = settings.TIMEFRAME_MAP[timeframe]["lookback"]
        latest_data = get_latest_data(symbol, timeframe)
        model, scaler = load_model(symbol, timeframe)
        model.eval()
        
        features = latest_data[FEATURE_LIST].values
        if scaler is not None:
            scaled_features = scaler.transform(features)
        else:
            scaled_features = features
        X = []
        actual_returns = []
        close_prices = latest_data['Close'].values
        for i in range(len(scaled_features) - lookback):
            X.append(scaled_features[i:i+lookback])
            # Actual return from this window
            current_close = close_prices[i + lookback - 1]
            next_close = close_prices[i + lookback] if i + lookback < len(close_prices) else current_close
            actual_returns.append((next_close - current_close) / current_close if current_close != 0 else 0.0)
        X = np.array(X)
        actual_returns = np.array(actual_returns)
        if len(X) == 0:
            raise ValueError("Insufficient data for prediction")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            predictions = model(X_tensor)
            if predictions.dim() == 3:
                predictions = predictions[:, -1, :]
            predictions = predictions.cpu().numpy()
        
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        elif predictions.ndim > 2:
            predictions = predictions.reshape(predictions.shape[0], -1)
        
        # Model outputs raw percentage return (target was NOT scaled during training)
        # Simply extract values and clamp to reasonable range [-0.1, 0.1]
        predictions_denorm = predictions.flatten()
        predictions_denorm = np.clip(predictions_denorm, -0.1, 0.1)
        
        # Compare returns (trim to matching length)
        min_len = min(len(actual_returns), len(predictions_denorm))
        actual_returns = actual_returns[-min_len:]
        predictions_denorm = predictions_denorm[-min_len:]
        
        mae = mean_absolute_error(actual_returns, predictions_denorm)
        rmse = np.sqrt(mean_squared_error(actual_returns, predictions_denorm))
        r2 = r2_score(actual_returns, predictions_denorm)
        direction_correct = np.sum(np.sign(actual_returns[1:] - actual_returns[:-1]) == np.sign(predictions_denorm[1:] - predictions_denorm[:-1])) / max(len(actual_returns) - 1, 1)
        logger.info(f"Model performance (returns) - MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.4f}, Directional Accuracy: {direction_correct:.4f}")
        
        return actual_returns, predictions_denorm
    except Exception as e:
        logger.error(f"Error in prepare_data_for_prediction: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        result = fetch_and_predict_btc_price(symbol="BTC", timeframe="1h")
        print(f"Prediction result: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")