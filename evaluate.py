"""Automated model evaluation with historical performance tracking."""

import os
import sys
import json
import asyncio
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import pytz
from sklearn.metrics import mean_absolute_error, mean_squared_error

from controllers.data_fetcher import DataFetcher
from ml_models.bilstm_predictor import BiLSTMWithAttention, FEATURE_LIST
from config import settings

EVAL_HISTORY_PATH = Path(settings.MODEL_PATH) / "evaluation_history.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_model(symbol: str, timeframe: str):
    """Load trained model and config from checkpoint."""
    model_path = Path(settings.MODEL_PATH) / symbol / f"model_{timeframe}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("model_config", checkpoint.get("config", {}))

    input_size = cfg.get("input_size", len(FEATURE_LIST))
    hidden_size = cfg.get("hidden_size", 32)
    num_layers = cfg.get("num_layers", 1)
    dropout = cfg.get("dropout", 0.2)

    model = BiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    # Strip torch.compile wrapper prefix if present
    state = {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def evaluate(symbol: str, timeframe: str, days_back: int = 30) -> Dict:
    """Evaluate a single model and return metrics."""
    model, checkpoint = load_model(symbol, timeframe)
    model_config = checkpoint.get("model_config", {})
    lookback = model_config.get("lookback", 48)

    fetcher = DataFetcher()
    data = asyncio.run(fetcher.get_merged_data(symbol, timeframe))
    if data is None or data.empty:
        raise ValueError(f"No data fetched for {symbol} {timeframe}")

    # Ensure all required features exist
    missing = [f for f in FEATURE_LIST if f not in data.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    data = data[list(FEATURE_LIST)].copy()
    data = data.ffill().bfill()

    close_all = asyncio.run(fetcher.get_merged_data(symbol, timeframe))
    if close_all is not None and "Close" in close_all.columns:
        close_series = close_all["Close"]
    else:
        raise ValueError("Cannot retrieve Close prices")

    # Align and trim
    common_idx = data.index.intersection(close_series.index)
    data = data.loc[common_idx]
    close_series = close_series.loc[common_idx]

    # Holdout: last `days_back` days worth of rows
    if hasattr(data.index, "to_pydatetime"):
        cutoff = data.index.max() - np.timedelta64(days_back, "D")
        holdout_mask = data.index >= cutoff
    else:
        holdout_mask = np.ones(len(data), dtype=bool)

    data_holdout = data.loc[holdout_mask].copy()
    close_holdout = close_series.loc[holdout_mask].values

    if len(data_holdout) < lookback + 10:
        raise ValueError(f"Not enough holdout data: {len(data_holdout)} rows (need {lookback + 10})")

    values = data_holdout.values.astype(np.float32)
    preds = []

    with torch.no_grad():
        for i in range(lookback, len(values)):
            seq = values[i - lookback : i]
            x = torch.FloatTensor(seq).unsqueeze(0)
            pred = model(x).item()
            # Model predicts percentage return; convert to price
            current_close = close_holdout[i - 1] if i - 1 >= 0 else close_holdout[0]
            predicted_price = current_close * (1 + pred)
            preds.append(predicted_price)

    actual = close_holdout[lookback:]
    preds = np.array(preds)

    # Directional accuracy
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(preds))
    directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)

    mae = float(mean_absolute_error(actual, preds))
    rmse = float(np.sqrt(mean_squared_error(actual, preds)))
    safe_actual = np.maximum(np.abs(actual), 1e-8)
    mape = float(np.mean(np.abs((actual - preds) / safe_actual)) * 100)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "holdout_days": days_back,
        "holdout_rows": len(actual),
        "model_config": {
            "input_size": model_config.get("input_size"),
            "hidden_size": model_config.get("hidden_size"),
            "num_layers": model_config.get("num_layers"),
        },
        "metrics": {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "mape": round(mape, 4),
            "directional_accuracy": round(directional_accuracy, 2),
        },
    }


def append_to_history(entry: Dict) -> None:
    """Append evaluation result to the historical ledger."""
    EVAL_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if EVAL_HISTORY_PATH.exists():
        with open(EVAL_HISTORY_PATH, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append(entry)

    with open(EVAL_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Appended result to {EVAL_HISTORY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM crypto models")
    parser.add_argument("--symbol", type=str, default="BTC", help="Crypto symbol (default: BTC)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--days", type=int, default=30, help="Holdout period in days (default: 30)")
    args = parser.parse_args()

    try:
        logger.info(f"Evaluating {args.symbol} {args.timeframe} (holdout={args.days}d)")
        result = evaluate(args.symbol, args.timeframe, args.days)

        m = result["metrics"]
        logger.info("Metrics:")
        logger.info(f"  Directional Accuracy: {m['directional_accuracy']:.2f}%")
        logger.info(f"  MAE:  {m['mae']:.6f}")
        logger.info(f"  RMSE: {m['rmse']:.6f}")
        logger.info(f"  MAPE: {m['mape']:.4f}%")

        append_to_history(result)
        logger.info("Done")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
