# Implementation Plan: Fix Noise Indicators & Data Diversification

## Overview
Remove 10 noise/lagging indicators and add 10 diversification features to improve model generalization and reduce overfitting.

---

## Phase 1: Update Feature Definitions (Source of Truth)

### 1.1 `config.py` (lines 86-126)
**Remove:**
- `SMA_20`, `EMA_20` (lagging)
- `MACD`, `MACD_Signal`, `MACD_Hist` (redundant with MAs)
- `Bollinger_middle`, `Bollinger_Upper`, `Bollinger_Lower` (redundant)
- `Lag1` (nearly identical to Close)
- `Sentiment_Up` (hardcoded 0.5)

**Add:**
- `Market_Regime` (0.6) - bull/bear/sideways encoding
- `Volatility_Regime` (0.6) - high/low volatility state
- `Hour_Sin`, `Hour_Cos` (0.5) - cyclical hour encoding
- `DayOfWeek_Sin`, `DayOfWeek_Cos` (0.5) - cyclical day encoding
- `Volume_Profile` (0.8) - volume relative to average
- `Momentum_5`, `Momentum_10`, `Momentum_20` (0.7-0.85) - multi-timeframe momentum

### 1.2 `config/__init__.py` (lines 7-36)
- Mirror changes from config.py
- Add missing kept features: `Momentum`, `Volatility`, `ADX`, `CCI`, `MFI`

### 1.3 `config/settings.py` (lines 56-59, 237-241)
- Update `TECHNICAL_INDICATORS` list
- Update `REQUIRED_FEATURES` list

---

## Phase 2: Fix Feature Calculations

### 2.1 `controllers/data_fetcher.py` (lines 510-600)
**Remove indicator calculations:**
- SMA_20 (line 517)
- EMA_20 (line 520)
- MACD block (lines 525-529)
- Bollinger block (lines 531-535)
- Lag1 (line 566)
- Sentiment_Up (line 569)

**Fix existing calculations:**
- `Momentum`: Change from `diff(10)` to `pct_change(10)` (line 560)
- `Volatility`: Change from `rolling(20).std()` to `rolling(20).std() / rolling(20).mean()` (line 563)

**Add new features:**
```python
# Market regime detection (using SMA crossover)
sma_short = df['Close'].rolling(10).mean()
sma_long = df['Close'].rolling(30).mean()
df_ta['Market_Regime'] = np.where(sma_short > sma_long, 1.0, 
                         np.where(sma_short < sma_long, -1.0, 0.0))

# Volatility regime
vol = df['Close'].pct_change().rolling(20).std()
vol_median = vol.rolling(50).median()
df_ta['Volatility_Regime'] = np.where(vol > vol_median, 1.0, -1.0)

# Time features (cyclical encoding)
if hasattr(df.index, 'hour'):
    df_ta['Hour_Sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df_ta['Hour_Cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df_ta['DayOfWeek_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df_ta['DayOfWeek_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

# Volume profile
df_ta['Volume_Profile'] = df['Volume'] / df['Volume'].rolling(20).mean()

# Multi-timeframe momentum
df_ta['Momentum_5'] = df['Close'].pct_change(5)
df_ta['Momentum_10'] = df['Close'].pct_change(10)
df_ta['Momentum_20'] = df['Close'].pct_change(20)
```

### 2.2 `controllers/model_trainer.py` (lines 76-131)
- Apply same removals and fixes as data_fetcher.py
- Ensure `_add_technical_indicators()` returns only FEATURE_LIST columns

---

## Phase 3: Update Model Files

### 3.1 `ml_models/bilstm_predictor.py` (lines 35-56)
- Update `FEATURE_LIST` to match new features
- Verify `input_size` default parameter matches new feature count

### 3.2 `controllers/prediction.py`
**Update `get_feature_list_for_model()` (lines 1108-1157):**
- Remove old features from local `feature_list`
- Add new diversification features
- Fix fallback calculations for Momentum and Volatility

**Update manual indicator calculations (lines 1780-1909):**
- Remove SMA_20, EMA_20, MACD, Bollinger calculations
- Fix Momentum to use pct_change
- Fix Volatility to use relative calculation
- Add new diversification feature calculations

**Update probability calculation (lines 819-872):**
- Remove MACD analysis block (lines 820-841)
- Remove Bollinger analysis block (lines 843-872)
- Update `confidence_factors` dict (line 930)

---

## Phase 4: Integrate DataAugmentor in Training

### 4.1 `controllers/model_trainer.py` (lines 137-202)
**Add import:**
```python
from controllers.prediction import DataAugmentor
```

**Update `prepare_data()` method (after line 150):**
```python
# After: data = self._add_technical_indicators(data)
# Add:
augmentor = DataAugmentor(self.timeframe)
data = augmentor.process(data)
```

### 4.2 Fix DataAugmentor internal calculations (prediction.py)
**Line 2210:** Change momentum calculation to percentage:
```python
# From: df[f'momentum_{period}'] = df['Close'] - df['Close'].shift(period)
# To: df[f'momentum_{period}'] = df['Close'].pct_change(period)
```

**Line 2219:** Fix RSI reference:
```python
# From: df['rsi_smooth'] = df['RSI'].rolling(3).mean()
# To: df['rsi_smooth'] = df['RSI_14'].rolling(3).mean()
```

---

## Phase 5: Fix Ancillary Files

### 5.1 `evaluate.py` (line 85)
```python
# From: X = data[FEATURE_LIST].values
# To: X = data[list(FEATURE_LIST.keys())].values
```

---

## Phase 6: Verification

1. **Run lint/typecheck** after each file edit
2. **Verify feature count** matches `input_size` in model architecture
3. **Test data pipeline** - ensure new features are calculated correctly
4. **Retrain models** - all existing .pth files are stale
5. **Verify prediction endpoint** returns valid results

---

## New Feature Count Breakdown

| Category | Features | Count |
|----------|----------|-------|
| Core Price | Open, High, Low, Close, Volume | 5 |
| Momentum | RSI_14, Momentum, ADX, CCI, Stoch_%K, Stoch_%D, MFI | 7 |
| Volatility | ATR, Volatility | 2 |
| Volume | OBV, VWAP, Volume_Profile | 3 |
| Multi-Timeframe Momentum | Momentum_5, Momentum_10, Momentum_20 | 3 |
| Market Regime | Market_Regime, Volatility_Regime | 2 |
| Time (Cyclical) | Hour_Sin, Hour_Cos, DayOfWeek_Sin, DayOfWeek_Cos | 4 |
| **Total** | | **26** |

---

## Files to Modify (in order)

1. `config.py`
2. `config/__init__.py`
3. `config/settings.py`
4. `controllers/data_fetcher.py`
5. `controllers/model_trainer.py`
6. `controllers/prediction.py`
7. `ml_models/bilstm_predictor.py`
8. `evaluate.py`

---

## Post-Implementation

- Retrain all models: `python train.py --symbol BTC --timeframe 1h`
- Verify prediction works: `python -c "from ml_models.bilstm_predictor import fetch_and_predict_btc_price; print(fetch_and_predict_btc_price())"`
- Run API server and test endpoint
