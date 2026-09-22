"""
Configuration package for the cryptocurrency prediction system.
"""
from .settings import *

# Feature list with importance weights
FEATURE_LIST = {
    # Price data (Primary features)
    'Open': 1.0,
    'High': 1.0,
    'Low': 1.0,
    'Close': 1.0,
    'Volume': 0.9,

    # Momentum indicators
    'RSI_14': 0.85,
    'Momentum': 0.7,
    'ADX': 0.7,
    'CCI': 0.7,
    'Stoch_%K': 0.7,
    'Stoch_%D': 0.7,
    'MFI': 0.7,

    # Volatility
    'ATR': 0.7,
    'Volatility': 0.7,

    # Volume
    'OBV': 0.8,
    'VWAP': 0.85,
    'Volume_Profile': 0.8,

    # Multi-timeframe momentum
    'Momentum_5': 0.7,
    'Momentum_10': 0.75,
    'Momentum_20': 0.85,

    # Market regime
    'Market_Regime': 0.6,
    'Volatility_Regime': 0.6,

    # Time features (cyclical)
    'Hour_Sin': 0.5,
    'Hour_Cos': 0.5,
    'DayOfWeek_Sin': 0.5,
    'DayOfWeek_Cos': 0.5,
}

# Tortoise ORM Settings
TORTOISE_ORM = {
    "connections": {"default": DATABASE_URL},
    "apps": {
        "models": {
            "models": ["models.user", "aerich.models"],
            "default_connection": "default",
        },
    },
}