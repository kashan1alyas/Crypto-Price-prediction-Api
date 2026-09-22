import os
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib
import logging
import asyncio
from datetime import datetime
import pytz
from typing import Tuple, Optional, Dict, Any, List
from config import settings
from .data_fetcher import DataFetcher
import ta
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import json
from .prediction_config import TIMEFRAME_CONFIG
from torch.utils.data import TensorDataset, DataLoader
from .prediction import DataAugmentor, clear_model_cache
import json as json_mod
from .model_versioning import (
    backup_champion, save_candidate, promote_candidate,
    discard_candidate, compare_metrics, get_champion_metrics,
    rollback, load_registry,
)

HYPERPARAMS_PATH = Path(settings.MODEL_PATH).parent / "config" / "best_hyperparameters.json"

# Setup logging
logger = logging.getLogger(__name__)

from ml_models.bilstm_predictor import BiLSTMWithAttention


def _strip_orig_mod_prefix(state_dict: dict) -> dict:
    """Remove _orig_mod. prefix from torch.compile wrapper keys."""
    return {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}


def _unwrap_model(model):
    """Return the underlying module if wrapped by torch.compile."""
    return getattr(model, "_orig_mod", model)

class ModelTrainer:
    # Enhanced feature list with more technical indicators
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
    
    def __init__(self, symbol: str, timeframe: str, force_promote: bool = False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_dir = Path(settings.MODEL_PATH) / symbol
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.model_dir / f'model_{timeframe}.pth'
        self.scaler_path = self.model_dir / f'scaler_{timeframe}.joblib'
        self.metadata_path = self.model_dir / f'metadata_{timeframe}.json'
        
        self.force_promote = force_promote
        
        # Get config for this timeframe
        self.config = dict(TIMEFRAME_CONFIG[timeframe])
        
        # Load tuned hyperparameters if available
        self._load_best_hyperparams()
        
        # Training parameters
        self.batch_size = self.config["batch_size"]
        self.validation_split = settings.VALIDATION_SPLIT
        self.patience = 40  # Increased patience for slower convergence
        self.max_epochs = 100
        
        # Dynamic min_data_points based on timeframe from TIMEFRAME_MAP
        tf_config = settings.TIMEFRAME_MAP.get(timeframe, {})
        self.min_data_points = tf_config.get('min_samples', 400)  # Default 400 if not found
        
        # Initialize data fetcher
        self.data_fetcher = DataFetcher()

    def _load_best_hyperparams(self) -> None:
        """Load tuned hyperparameters from config/best_hyperparameters.json if available."""
        if not HYPERPARAMS_PATH.exists():
            return
        try:
            with open(HYPERPARAMS_PATH, "r") as f:
                all_params = json_mod.load(f)
            key = f"{self.symbol}_{self.timeframe}"
            entry = all_params.get(key)
            if entry is None:
                return
            best = entry.get("best_params", {})
            if "lr" in best:
                self.config["learning_rate"] = best["lr"]
            if "hidden_size" in best:
                self.config["hidden_size"] = best["hidden_size"]
            if "num_layers" in best:
                self.config["num_layers"] = best["num_layers"]
            if "batch_size" in best:
                self.config["batch_size"] = best["batch_size"]
            if "dropout" in best:
                self.config["dropout"] = best["dropout"]
            logger.info(f"Loaded tuned hyperparameters for {self.symbol} {self.timeframe}: {best}")
        except Exception as e:
            logger.warning(f"Failed to load tuned hyperparameters: {e}")
        
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using the ta library"""
        try:
            df_ta = df.copy()
            eps = 1e-8

            # RSI
            df_ta['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)

            # Momentum (safe pct_change)
            shifted = df['Close'].shift(10)
            df_ta['Momentum'] = (df['Close'] - shifted) / shifted.replace(0, eps)

            # ADX
            df_ta['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

            # CCI (manual with epsilon)
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(20).mean()
            mean_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
            df_ta['CCI'] = (tp - sma_tp) / (0.015 * mean_dev + eps)

            # Stochastic (manual with epsilon)
            low_min = df['Low'].rolling(14).min()
            high_max = df['High'].rolling(14).max()
            df_ta['Stoch_%K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + eps)
            df_ta['Stoch_%D'] = df_ta['Stoch_%K'].rolling(3).mean()

            # MFI (manual with epsilon)
            tp_mfi = (df['High'] + df['Low'] + df['Close']) / 3
            mf = tp_mfi * df['Volume']
            pos_mf = pd.Series(0.0, index=df.index)
            neg_mf = pd.Series(0.0, index=df.index)
            tp_diff = tp_mfi.diff()
            pos_mf[tp_diff > 0] = mf[tp_diff > 0]
            neg_mf[tp_diff < 0] = mf[tp_diff < 0]
            pos_sum = pos_mf.rolling(14).sum()
            neg_sum = neg_mf.rolling(14).sum()
            df_ta['MFI'] = 100 - (100 / (1 + pos_sum / (neg_sum + eps)))

            # ATR
            df_ta['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])

            # Volatility (relative) with epsilon
            vol = df['Close'].rolling(window=20).std()
            vol_mean = df['Close'].rolling(window=20).mean()
            df_ta['Volatility'] = vol / (vol_mean.abs() + eps)

            # OBV
            df_ta['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

            # VWAP with epsilon
            vwap_num = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum()
            vwap_den = df['Volume'].cumsum()
            df_ta['VWAP'] = vwap_num / (vwap_den + eps)

            # Volume profile with epsilon
            vol_ma = df['Volume'].rolling(window=20).mean()
            df_ta['Volume_Profile'] = df['Volume'] / (vol_ma + eps)

            # Multi-timeframe momentum (safe pct_change)
            for period, col in [(5, 'Momentum_5'), (10, 'Momentum_10'), (20, 'Momentum_20')]:
                shifted = df['Close'].shift(period)
                df_ta[col] = (df['Close'] - shifted) / shifted.replace(0, eps)

            # Market regime detection (SMA crossover)
            sma_short = df['Close'].rolling(10).mean()
            sma_long = df['Close'].rolling(30).mean()
            df_ta['Market_Regime'] = np.where(sma_short > sma_long, 1.0,
                                     np.where(sma_short < sma_long, -1.0, 0.0))

            # Volatility regime (safe pct_change)
            ret_vol = (df['Close'] - df['Close'].shift(1)) / (df['Close'].shift(1).replace(0, eps))
            ret_vol = ret_vol.rolling(20).std()
            vol_median = ret_vol.rolling(50).median()
            df_ta['Volatility_Regime'] = np.where(ret_vol > vol_median, 1.0, -1.0)

            # Time features (cyclical encoding)
            if hasattr(df.index, 'hour'):
                df_ta['Hour_Sin'] = np.sin(2 * np.pi * df.index.hour / 24)
                df_ta['Hour_Cos'] = np.cos(2 * np.pi * df.index.hour / 24)
                df_ta['DayOfWeek_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
                df_ta['DayOfWeek_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            else:
                df_ta['Hour_Sin'] = 0.0
                df_ta['Hour_Cos'] = 1.0
                df_ta['DayOfWeek_Sin'] = 0.0
                df_ta['DayOfWeek_Cos'] = 1.0

            # Replace inf/NaN then forward/backward fill
            df_ta = df_ta.replace([np.inf, -np.inf], np.nan)
            df_ta = df_ta.ffill().bfill().fillna(0.0)

            return df_ta[self.FEATURE_LIST]
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
    
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, RobustScaler]:
        """Prepare data for training with enhanced validation.
        
        Target is the percentage return from the last known close price:
            y = (Close[t+lookback] - Close[t+lookback-1]) / Close[t+lookback-1]
        """
        try:
            # Fetch data with fallback mechanism
            data = asyncio.run(self.data_fetcher.get_merged_data(self.symbol, self.timeframe))
            if data is None or len(data) < self.min_data_points:
                raise ValueError(f"Insufficient data points for {self.symbol}: got {len(data) if data is not None else 0}, require {self.min_data_points}")
            
            # Calculate technical indicators
            data = self._add_technical_indicators(data)
            
            # Apply data augmentation
            augmentor = DataAugmentor(self.timeframe)
            data = augmentor.process(data)

            # Re-filter to expected features (augmentor adds extra columns)
            data = data[self.FEATURE_LIST]
            
            # --- Robust data validation ---
            if data.empty:
                raise ValueError("Dataframe is empty after feature filtering")
            
            if len(data) < 50:
                raise ValueError(f"Insufficient valid data points: {len(data)} (need at least 50)")
            
            # Drop rows that are all-NaN after indicators settle
            data = data.dropna(how='all')
            if len(data) < 50:
                raise ValueError(f"Insufficient data after dropping all-NaN rows: {len(data)}")
            
            # Replace inf/-inf with NaN, then forward-fill
            data = data.replace([np.inf, -np.inf], np.nan)
            data = data.ffill().bfill()
            
            if data.isnull().any().any():
                raise ValueError("Data contains NaN values after preprocessing")
            
            # Check for zero-variance columns (flat values cause division-by-zero in scalers)
            zero_var_cols = [c for c in data.columns if data[c].std() == 0]
            if zero_var_cols:
                raise ValueError(f"Zero-variance columns detected: {zero_var_cols}")
            
            # Check for constant columns that would break RobustScaler
            for col in data.columns:
                if data[col].nunique() <= 1:
                    raise ValueError(f"Column '{col}' has only {data[col].nunique()} unique value(s) — cannot scale")
            
            if len(data) < self.min_data_points:
                raise ValueError(f"Insufficient data points after preprocessing: {len(data)}")
            
            # Calculate percentage returns for target
            close_prices = data['Close'].values
            close_col_idx = data.columns.get_loc('Close')
            
            # Validate close prices for division safety
            if np.any(close_prices == 0):
                raise ValueError("Close price contains zero values — percentage return would cause division by zero")
            if np.any(close_prices < 0):
                raise ValueError("Close price contains negative values — percentage return undefined")
            
            # Scale features
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences with validation
            X, y = [], []
            lookback = self.config["lookback"]
            
            for i in range(len(scaled_data) - lookback - 1):
                X.append(scaled_data[i:i+lookback])
                # Percentage return: (next_close - current_close) / current_close
                current_close = close_prices[i + lookback - 1]
                next_close = close_prices[i + lookback]
                pct_return = (next_close - current_close) / current_close if current_close != 0 else 0.0
                y.append(pct_return)
            
            X = np.array(X)
            y = np.array(y)
            
            if len(X) < 100:  # Minimum sequences required
                raise ValueError(f"Insufficient sequences generated: {len(X)}")
            
            # Split data with shuffling for better training
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            split = int(len(indices) * (1 - self.validation_split))
            
            train_idx = indices[:split]
            val_idx = indices[split:]
            
            X_train = torch.FloatTensor(X[train_idx])
            y_train = torch.FloatTensor(y[train_idx])
            X_val = torch.FloatTensor(X[val_idx])
            y_val = torch.FloatTensor(y[val_idx])
            
            return X_train, y_train, X_val, y_val, scaler
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
        
    def train(self) -> Dict[str, Any]:
        """Train the model with champion-challenger versioning.

        Flow:
        1. Backup the current active champion checkpoint
        2. Train a candidate model (saved to _candidate.pth)
        3. Compare candidate vs champion metrics
        4. Promote candidate if better (or if force_promote=True), else discard
        5. Return training results with promotion status
        """
        try:
            # --- Step 1: Backup current champion before training ---
            had_champion = backup_champion(
                str(settings.MODEL_PATH), self.symbol, self.timeframe
            )
            if had_champion:
                logger.info(f"Backed up existing champion for {self.symbol} {self.timeframe}")
            else:
                logger.info(f"No existing champion to backup for {self.symbol} {self.timeframe} (first training run)")

            # Prepare data
            X_train, y_train, X_val, y_val, scaler = self.prepare_data()
            
            # Initialize model with size based on features
            input_size = len(self.FEATURE_LIST)
            model = BiLSTMWithAttention(
                input_size=input_size,
                hidden_size=self.config["hidden_size"],
                num_layers=self.config["num_layers"],
                dropout=self.config["dropout"]
            )
            
            # Warm-start: load existing checkpoint if available
            fine_tuning = False
            if self.model_path.exists():
                try:
                    ckpt = torch.load(self.model_path, map_location='cpu', weights_only=False)
                    state = ckpt.get('model_state_dict', ckpt)
                    if isinstance(state, dict) and 'model_state_dict' in state:
                        state = state['model_state_dict']
                    model.load_state_dict(_strip_orig_mod_prefix(state))
                    fine_tuning = True
                    logger.info(f"Loaded existing checkpoint from {self.model_path} — fine-tuning")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint ({e}) — training from scratch")
            else:
                logger.info(f"No checkpoint found — training from scratch for {self.symbol} {self.timeframe}")
            
            # Move to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)

            # torch.compile accelerates training when PyTorch >= 2.0
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model)
                except Exception:
                    pass
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            
            # Loss function and optimizer with advanced training configuration
            criterion = nn.HuberLoss(delta=1.0)
            
            # Use reduced LR for fine-tuning to prevent catastrophic forgetting
            base_lr = self.config["learning_rate"]
            if fine_tuning:
                train_lr = base_lr * 0.1  # 10x smaller for fine-tuning
                logger.info(f"Fine-tuning learning rate: {train_lr:.2e} (base {base_lr:.2e} × 0.1)")
            else:
                train_lr = base_lr
                logger.info(f"Training from scratch learning rate: {train_lr:.2e}")
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=train_lr,
                weight_decay=0.01,  # L2 regularization
                betas=(0.9, 0.999),  # Default Adam betas
                eps=1e-8  # Default Adam epsilon
            )
            
            # Calculate steps per epoch and total steps
            steps_per_epoch = -(-len(X_train) // self.batch_size)  # ceil division
            total_steps = steps_per_epoch * self.max_epochs
            
            # OneCycleLR scheduler for better convergence
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=train_lr * 10,
                total_steps=total_steps,
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=1000.0,
                anneal_strategy='cos'
            )
            
            # Validate we have enough data for training
            n_train_batches = -(-len(X_train) // self.batch_size)  # ceil
            n_val_batches = -(-len(X_val) // self.batch_size)  # ceil
            if n_train_batches == 0 or n_val_batches == 0:
                raise ValueError(
                    f"Insufficient sequences for {self.symbol} ({self.timeframe}): "
                    f"train_batches={n_train_batches}, val_batches={n_val_batches}"
                )
            
            # Training loop with enhanced monitoring
            best_val_loss = float('inf')
            best_r2_score = float('-inf')
            patience_counter = 0
            training_history = []
            best_model_state = None
            
            # Use a temporary path during training so the active champion stays untouched
            temp_model_path = self.model_dir / f'model_{self.timeframe}_temp.pth'
            
            for epoch in range(self.max_epochs):
                model.train()
                train_loss = 0
                train_predictions = []
                train_actuals = []
                
                # Batch training with progress tracking
                for i in range(0, len(X_train), self.batch_size):
                    batch_X = X_train[i:i+self.batch_size]
                    batch_y = y_train[i:i+self.batch_size]
                    
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output.squeeze(), batch_y)
                    
                    # Backward pass with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler.last_epoch < total_steps:
                        scheduler.step()
                    
                    train_loss += loss.item()
                    train_predictions.extend(output.squeeze().detach().cpu().numpy())
                    train_actuals.extend(batch_y.cpu().numpy())
                
                # Validation
                model.eval()
                val_loss = 0
                val_predictions = []
                val_actuals = []
                
                with torch.no_grad():
                    for i in range(0, len(X_val), self.batch_size):
                        batch_X = X_val[i:i+self.batch_size]
                        batch_y = y_val[i:i+self.batch_size]
                        
                        output = model(batch_X)
                        val_loss += criterion(output.squeeze(), batch_y).item()
                        
                        val_predictions.extend(output.squeeze().cpu().numpy())
                        val_actuals.extend(batch_y.cpu().numpy())
                
                # Calculate metrics
                train_loss /= max(-(-len(X_train) // self.batch_size), 1)
                val_loss /= max(-(-len(X_val) // self.batch_size), 1)
                
                # Convert lists to numpy arrays for metric calculation
                train_predictions = np.array(train_predictions)
                train_actuals = np.array(train_actuals)
                val_predictions = np.array(val_predictions)
                val_actuals = np.array(val_actuals)
                
                # Calculate R2 scores
                train_r2 = r2_score(train_actuals, train_predictions)
                val_r2 = r2_score(val_actuals, val_predictions)
                
                # Calculate MAPE (safe: clamp denominators)
                safe_train = np.maximum(np.abs(train_actuals), 1e-8)
                train_mape = np.mean(np.abs((train_actuals - train_predictions) / safe_train)) * 100
                safe_val = np.maximum(np.abs(val_actuals), 1e-8)
                val_mape = np.mean(np.abs((val_actuals - val_predictions) / safe_val)) * 100
                
                # Calculate RMSE
                train_rmse = np.sqrt(mean_squared_error(train_actuals, train_predictions))
                val_rmse = np.sqrt(mean_squared_error(val_actuals, val_predictions))
                
                # Get current learning rate
                current_lr = scheduler.get_last_lr()[0]
                
                logger.info(
                    f"Epoch {epoch + 1}/{self.max_epochs}\n"
                    f"Train - Loss: {train_loss:.6f}, R2: {train_r2:.4f}, MAPE: {train_mape:.2f}%, RMSE: {train_rmse:.4f}\n"
                    f"Val   - Loss: {val_loss:.6f}, R2: {val_r2:.4f}, MAPE: {val_mape:.2f}%, RMSE: {val_rmse:.4f}\n"
                    f"Learning Rate: {current_lr:.8f}"
                )
                
                # Save training history
                training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'train_r2': float(train_r2),
                    'val_r2': float(val_r2),
                    'train_mape': float(train_mape),
                    'val_mape': float(val_mape),
                    'train_rmse': float(train_rmse),
                    'val_rmse': float(val_rmse),
                    'learning_rate': float(current_lr)
                })
                
                # Early stopping check - prioritize R2 improvement
                improved = False
                
                # Check if R2 score improved
                if val_r2 > best_r2_score:
                    best_r2_score = val_r2
                    improved = True
                    best_model_state = _unwrap_model(model).state_dict().copy()
                    
                    # Save best model to temp path (not the active champion)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': _unwrap_model(model).state_dict(),
                        'model_config': {
                            'input_size': input_size,
                            'hidden_size': self.config["hidden_size"],
                            'num_layers': self.config["num_layers"],
                            'dropout': self.config["dropout"],
                            'lookback': self.config["lookback"],
                            'feature_list': self.FEATURE_LIST,
                            'feature_names': self.FEATURE_LIST,
                            'prediction_mode': 'return'
                        },
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'val_r2': val_r2,
                        'val_mape': val_mape,
                        'train_loss': train_loss,
                        'train_r2': train_r2,
                        'train_mape': train_mape,
                        'rmse': val_rmse,
                        'learning_rate': current_lr
                    }, temp_model_path)
                    
                    # Save scaler to temp path
                    temp_scaler_path = self.model_dir / f'scaler_{self.timeframe}_temp.joblib'
                    joblib.dump(scaler, temp_scaler_path)
                # If R2 didn't improve but loss did significantly
                elif val_loss < best_val_loss * 0.95:  # 5% improvement threshold
                    best_val_loss = val_loss
                    improved = True
                
                if not improved:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                else:
                    patience_counter = 0
                
                # Save training curves after each epoch
                curves_path = self.model_dir / f'training_curves_{self.timeframe}.json'
                with open(curves_path, 'w') as f:
                    json.dump(training_history, f, indent=2)
            
            # --- Step 2: Candidate training complete ---
            # Load best model state from temp checkpoint
            if best_model_state is not None:
                model.load_state_dict(_strip_orig_mod_prefix(best_model_state))
            
            # Build candidate checkpoint data
            candidate_checkpoint = {
                'model_state_dict': _unwrap_model(model).state_dict(),
                'model_config': {
                    'input_size': input_size,
                    'hidden_size': self.config["hidden_size"],
                    'num_layers': self.config["num_layers"],
                    'dropout': self.config["dropout"],
                    'lookback': self.config["lookback"],
                    'feature_list': self.FEATURE_LIST,
                    'feature_names': self.FEATURE_LIST,
                    'prediction_mode': 'return'
                },
                'val_loss': float(best_val_loss),
                'val_r2': float(best_r2_score)
            }

            # Candidate metrics for comparison
            candidate_metrics = {
                'best_val_loss': float(best_val_loss),
                'best_val_r2': float(best_r2_score),
                'final_val_mape': float(val_mape),
                'final_rmse': float(val_rmse),
            }

            # --- Step 3: Champion-Challenger comparison ---
            champion_metrics = get_champion_metrics(
                str(settings.MODEL_PATH), self.symbol, self.timeframe
            )

            should_promote = False
            promotion_reason = ""

            if not had_champion:
                # First training run — always promote
                should_promote = True
                promotion_reason = "First training run — no existing champion"
            elif self.force_promote:
                should_promote = True
                promotion_reason = "Force promote (--override flag)"
            else:
                is_better, reason = compare_metrics(champion_metrics, candidate_metrics)
                should_promote = is_better
                promotion_reason = reason

            if should_promote:
                # --- Step 4a: Promote candidate to active ---
                promote_candidate(
                    str(settings.MODEL_PATH), self.symbol, self.timeframe,
                    champion_metrics=champion_metrics,
                    candidate_metrics=candidate_metrics,
                    feature_list=self.FEATURE_LIST,
                    feature_names=self.FEATURE_LIST,
                    input_size=input_size,
                )
                logger.info(f"PROMOTED candidate for {self.symbol} {self.timeframe}: {promotion_reason}")

                # Copy the temp checkpoint to the active model path
                import shutil
                if temp_model_path.exists():
                    shutil.copy2(temp_model_path, self.model_path)
                    temp_model_path.unlink()
                temp_scaler_path = self.model_dir / f'scaler_{self.timeframe}_temp.joblib'
                if temp_scaler_path.exists():
                    shutil.copy2(temp_scaler_path, self.scaler_path)
                    temp_scaler_path.unlink()

                # Save scalers
                os.makedirs(self.model_dir, exist_ok=True)
                joblib.dump(scaler, self.scaler_path)
                feature_scaler_path = self.model_dir / 'feature_scaler.joblib'
                joblib.dump(scaler, feature_scaler_path)

                # Save metadata
                metadata = {
                    'symbol': self.symbol,
                    'timeframe': self.timeframe,
                    'training_date': datetime.now(pytz.UTC).isoformat(),
                    'epochs_trained': epoch + 1,
                    'best_val_loss': float(best_val_loss),
                    'best_val_r2': float(best_r2_score),
                    'final_val_mape': float(val_mape),
                    'final_rmse': float(val_rmse),
                    'data_points': len(X_train) + len(X_val),
                    'feature_list': self.FEATURE_LIST,
                    'model_parameters': {
                        'input_size': input_size,
                        'hidden_size': self.config["hidden_size"],
                        'num_layers': self.config["num_layers"],
                        'dropout': self.config["dropout"],
                        'batch_size': self.batch_size,
                        'learning_rate': self.config["learning_rate"],
                        'weight_decay': 0.01
                    },
                    'training_history': training_history
                }
                with open(self.metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Export TorchScript-optimized model for fast inference
                try:
                    self._export_optimized_model(input_size)
                except Exception as e:
                    logger.warning(f"Failed to export optimized model (non-fatal): {e}")

                # Invalidate cached model
                clear_model_cache(self.symbol, self.timeframe)

                logger.info(f"Saved promoted model to {self.model_path}")

                return {
                    'status': 'success',
                    'promoted': True,
                    'promotion_reason': promotion_reason,
                    'champion_metrics': champion_metrics,
                    'candidate_metrics': candidate_metrics,
                    'epochs_trained': epoch + 1,
                    'best_val_loss': float(best_val_loss),
                    'best_val_r2': float(best_r2_score),
                    'final_val_mape': float(val_mape),
                    'final_rmse': float(val_rmse),
                    'model_path': str(self.model_path),
                    'data_points': len(X_train) + len(X_val),
                    'validation_metrics': {
                        'rmse': float(val_rmse),
                        'r2': float(val_r2),
                        'mape': float(val_mape)
                    }
                }
            else:
                # --- Step 4b: Candidate rejected, keep champion ---
                discard_candidate(
                    str(settings.MODEL_PATH), self.symbol, self.timeframe
                )
                # Clean up temp files
                if temp_model_path.exists():
                    temp_model_path.unlink()
                temp_scaler_path = self.model_dir / f'scaler_{self.timeframe}_temp.joblib'
                if temp_scaler_path.exists():
                    temp_scaler_path.unlink()

                logger.info(
                    f"REJECTED candidate for {self.symbol} {self.timeframe}: {promotion_reason}. "
                    f"Champion retained."
                )

                return {
                    'status': 'success',
                    'promoted': False,
                    'promotion_reason': promotion_reason,
                    'champion_metrics': champion_metrics,
                    'candidate_metrics': candidate_metrics,
                    'epochs_trained': epoch + 1,
                    'best_val_loss': float(best_val_loss),
                    'best_val_r2': float(best_r2_score),
                    'final_val_mape': float(val_mape),
                    'final_rmse': float(val_rmse),
                    'data_points': len(X_train) + len(X_val),
                    'validation_metrics': {
                        'rmse': float(val_rmse),
                        'r2': float(val_r2),
                        'mape': float(val_mape)
                    }
                }
            
        except ValueError as e:
            logger.warning(f"Training aborted (data validation): {e}")
            return {
                'status': 'training_failed',
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"Training error: {str(e)}\n{traceback.format_exc()}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def _export_optimized_model(self, input_size: int) -> None:
        """Export a TorchScript-optimized model for fast inference.

        Creates a traced version using torch.jit.trace and saves it as
        models/{symbol}/model_{timeframe}_optimized.pt
        """
        optimized_path = self.model_dir / f'model_{self.timeframe}_optimized.pt'

        # Load the promoted checkpoint
        if not self.model_path.exists():
            logger.warning(f"No promoted checkpoint at {self.model_path} — skipping optimized export")
            return

        ckpt = torch.load(self.model_path, map_location='cpu', weights_only=False)
        config = ckpt.get('model_config', ckpt.get('config', {}))

        raw_model = BiLSTMWithAttention(
            input_size=config.get('input_size', input_size),
            hidden_size=config.get('hidden_size', self.config["hidden_size"]),
            num_layers=config.get('num_layers', self.config["num_layers"]),
            dropout=config.get('dropout', self.config["dropout"]),
        )
        state_dict = ckpt.get('model_state_dict', ckpt)
        clean_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        raw_model.load_state_dict(clean_dict)
        raw_model.eval()

        # Create a dummy input matching the expected shape: (1, lookback, input_size)
        lookback = config.get('lookback', self.config.get("lookback", 72))
        dummy_input = torch.randn(1, lookback, input_size)

        # Trace and optimize for inference
        with torch.no_grad():
            traced = torch.jit.trace(raw_model, dummy_input)
            traced_opt = torch.jit.optimize_for_inference(traced)

        traced_opt.save(str(optimized_path))
        logger.info(f"Exported optimized TorchScript model to {optimized_path}")


def train_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
                model: BiLSTMWithAttention, timeframe: str = "24h", symbol: str = "BTC",
                batch_size: int = 128, learning_rate: float = 0.001) -> BiLSTMWithAttention:
    """Train the model with early stopping and enhanced monitoring"""
    try:
        # Define constants
        EPOCHS = 100  # Maximum number of epochs
        PATIENCE = 20  # Increased patience for early stopping (was 15)
        CLIP_GRAD_NORM = 1.0  # Maximum gradient norm for clipping
        
        logger.info(f"Starting model training for {symbol} {timeframe}")
        logger.info(f"Training data shape: X_train {X_train.shape}, y_train {y_train.shape}")
        logger.info(f"Validation data shape: X_val {X_val.shape}, y_val {y_val.shape}")
        logger.info(f"Model hyperparameters: batch_size={batch_size}, learning_rate={learning_rate}")
        
        # Initialize lists to store metrics for plotting
        train_losses = []
        val_losses = []
        val_mapes = []
        val_rmses = []
        val_r2s = []
        
        # Convert data to tensors and ensure correct shape
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1)  # Flatten to 1D
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).reshape(-1)  # Flatten to 1D
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        # Training loop
        for epoch in range(EPOCHS):
            model.train()
            total_train_loss = 0
            num_batches = 0
            
            # Training phase
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_X)
                # Ensure both tensors are the same shape before computing loss
                outputs = outputs.view(-1)  # Flatten predictions
                batch_y = batch_y.view(-1)  # Flatten targets
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                
                # Optimizer step
                optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_train_loss / max(num_batches, 1)
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                # Ensure both tensors are the same shape before computing loss
                val_outputs = val_outputs.view(-1)  # Flatten predictions
                y_val_tensor_reshaped = y_val_tensor.view(-1)  # Flatten targets
                val_loss = criterion(val_outputs, y_val_tensor_reshaped)
                
                # Calculate additional metrics (safe MAPE)
                val_actuals_np = y_val_tensor_reshaped.numpy()
                val_preds_np = val_outputs.numpy()
                safe_denom = np.maximum(np.abs(val_actuals_np), 1e-8)
                val_mape = np.mean(np.abs((val_actuals_np - val_preds_np) / safe_denom))
                val_rmse = np.sqrt(mean_squared_error(val_actuals_np, val_preds_np))
                val_r2 = r2_score(val_actuals_np, val_preds_np)
            
            # Monitor training data quality
            monitor_training_data_quality(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epoch)
            
            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss.item())
            val_mapes.append(val_mape)
            val_rmses.append(val_rmse)
            val_r2s.append(val_r2)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Log progress
            logger.info(f"Epoch {epoch} - Training Data Stats: {get_data_stats(X_train_tensor, y_train_tensor)}")
            logger.info(f"Epoch {epoch} - Validation Data Stats: {get_data_stats(X_val_tensor, y_val_tensor)}")
            logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}, MAPE = {val_mape:.2f}%, RMSE = {val_rmse:.4f}, R2 = {val_r2:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = _unwrap_model(model).state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(_strip_orig_mod_prefix(best_model_state))
            logger.info("Loaded best model state from training")
        
        # Save training curves
        save_training_curves(train_losses, val_losses, val_mapes, val_rmses, val_r2s, os.path.join(settings.MODEL_PATH, f"{symbol}_{timeframe}"))
        
        return model
        
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise 