"""Optuna hyperparameter tuning for BiLSTM cryptocurrency model."""

import sys
import json
import logging
import argparse
from pathlib import Path
import optuna

from controllers.model_trainer import ModelTrainer
from config import settings

HYPERPARAMS_PATH = Path(settings.MODEL_PATH).parent / "config" / "best_hyperparameters.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tuning.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial, symbol: str, timeframe: str) -> float:
    """Optuna objective: train model with suggested hyperparameters, return val loss."""
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)

    trainer = ModelTrainer(symbol, timeframe)

    # Override config with trial-suggested values
    trainer.config = {
        **trainer.config,
        "learning_rate": lr,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
    }
    trainer.batch_size = batch_size

    try:
        result = trainer.train()
    except Exception as e:
        logger.warning(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned(str(e))

    if result["status"] != "success":
        raise optuna.TrialPruned(result.get("error", "training failed"))

    val_loss = result["best_val_loss"]
    val_r2 = result["best_val_r2"]

    trial.set_user_attr("val_r2", val_r2)
    trial.set_user_attr("epochs", result["epochs_trained"])

    logger.info(
        f"Trial {trial.number}: val_loss={val_loss:.6f}  val_r2={val_r2:.4f}  "
        f"params={{lr={lr:.6f}, hidden={hidden_size}, layers={num_layers}, "
        f"batch={batch_size}, dropout={dropout:.3f}}}"
    )
    return val_loss


def save_best_params(symbol: str, timeframe: str, study: optuna.study.Study) -> None:
    """Persist best trial parameters to config/best_hyperparameters.json."""
    best = study.best_trial
    HYPERPARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing file or start fresh
    if HYPERPARAMS_PATH.exists():
        with open(HYPERPARAMS_PATH, "r") as f:
            all_params = json.load(f)
    else:
        all_params = {}

    key = f"{symbol}_{timeframe}"
    all_params[key] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "best_val_loss": best.value,
        "best_val_r2": best.user_attrs.get("val_r2"),
        "best_trial": best.number,
        "best_params": best.params,
    }

    with open(HYPERPARAMS_PATH, "w") as f:
        json.dump(all_params, f, indent=2)

    logger.info(f"Saved best params to {HYPERPARAMS_PATH}")


def run_tuning(
    symbol: str,
    timeframe: str,
    n_trials: int = 50,
    timeout: int | None = None,
) -> optuna.study.Study:
    """Run Optuna study and return the completed study."""
    study = optuna.create_study(
        study_name=f"{symbol}_{timeframe}_tuning",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        lambda trial: objective(trial, symbol, timeframe),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    # Report best result
    best = study.best_trial
    logger.info("=" * 60)
    logger.info("Tuning complete")
    logger.info(f"  Best trial:  {best.number}")
    logger.info(f"  Val loss:    {best.value:.6f}")
    logger.info(f"  Val R2:      {best.user_attrs.get('val_r2', 'N/A')}")
    logger.info(f"  Epochs:      {best.user_attrs.get('epochs', 'N/A')}")
    logger.info(f"  Parameters:")
    for k, v in best.params.items():
        logger.info(f"    {k}: {v}")
    logger.info("=" * 60)

    save_best_params(symbol, timeframe, study)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune BiLSTM hyperparameters with Optuna")
    parser.add_argument("--symbol", type=str, default="BTC", help="Crypto symbol (default: BTC)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials (default: 50)")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds (optional)")
    args = parser.parse_args()

    run_tuning(args.symbol, args.timeframe, n_trials=args.trials, timeout=args.timeout)
