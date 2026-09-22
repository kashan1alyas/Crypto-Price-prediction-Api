"""Configuration for model training and prediction."""

# Training constants
EPOCHS = 100
PATIENCE = 40  # Increased for slower convergence with lower learning rate
CLIP_GRAD_NORM = 0.5

# Model hyperparameters for different timeframes
TIMEFRAME_CONFIG = {
    "30m": {
        "lookback": 48,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.3,
        "batch_size": 128,
        "learning_rate": 0.00005
    },
    "1h": {
        "lookback": 48,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.2,
        "batch_size": 128,
        "learning_rate": 0.001
    },
    "4h": {
        "lookback": 36,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.2,
        "batch_size": 128,
        "learning_rate": 0.001
    },
    "24h": {
        "lookback": 30,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.2,
        "batch_size": 128,
        "learning_rate": 0.001
    }
} 