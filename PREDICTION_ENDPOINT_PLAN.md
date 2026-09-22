# Implementation Plan: Prediction Endpoint Model Not Found Handling

## Overview
Update the `/api/predict/{symbol}` endpoint to return a user-friendly JSON response when a model doesn't exist, instead of throwing a 500 error. The response should ask whether they want to trigger training.

---

## Current Behavior
1. User calls `GET /api/predict/ETH?timeframe=1h`
2. `routes/prediction.py` calls `predict_next_price("ETH", "1h")`
3. `controllers/prediction.py` line 965-966 raises `FileNotFoundError`
4. Route handler catches it as generic `Exception` at line 35-38
5. Returns 500: `{"detail": "An unexpected internal server error occurred: No trained model found for ETH 1h"}`

---

## Desired Behavior
1. User calls `GET /api/predict/ETH?timeframe=1h`
2. System checks if model exists at `models/ETH/model_1h.pth`
3. If not found, return 404 with JSON:
```json
{
  "status": "no_model",
  "symbol": "ETH",
  "timeframe": "1h",
  "message": "No trained model found for ETH (1h).",
  "train_endpoint": "/api/train/ETH?timeframe=1h",
  "action_required": true
}
```

---

## Implementation Options

### Option A: Return 404 with Training Prompt (Recommended)
**File: `routes/prediction.py`**

Modify the route handler to catch `FileNotFoundError` specifically and return a structured response.

```python
@router.get("/predict/{symbol}", operation_id="get_prediction_for_symbol")
async def get_prediction(symbol: str, timeframe: Timeframe):
    try:
        if not symbol or len(symbol.strip()) == 0:
            raise ValueError("Symbol cannot be empty")

        prediction_result = await predict_next_price(symbol.upper(), timeframe.value)

        if isinstance(prediction_result, dict) and "error" in prediction_result:
            raise HTTPException(status_code=400, detail=prediction_result["error"])

        return prediction_result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        # Return structured response for missing model
        return JSONResponse(
            status_code=404,
            content={
                "status": "no_model",
                "symbol": symbol.upper(),
                "timeframe": timeframe.value,
                "message": f"No trained model found for {symbol.upper()} ({timeframe.value}).",
                "train_endpoint": f"/api/train/{symbol.upper()}?timeframe={timeframe.value}",
                "action_required": True
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            content={"detail": f"An unexpected internal server error occurred: {str(e)}"},
            status_code=500,
        )
```

### Option B: Add Training Endpoint
**File: `routes/prediction.py`**

Add a new endpoint to trigger training:

```python
@router.post("/train/{symbol}", operation_id="train_model_for_symbol")
async def train_model_endpoint(symbol: str, timeframe: Timeframe):
    """
    Trigger training for a given symbol and timeframe.
    Returns immediately with a training started message.
    """
    try:
        from controllers.model_trainer import ModelTrainer
        
        # Start training in background
        import asyncio
        from concurrent.futures import ProcessPoolExecutor
        
        def train_in_background(sym, tf):
            trainer = ModelTrainer(sym, tf)
            return trainer.train()
        
        # Run training in executor to not block the event loop
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(
                executor,
                train_in_background,
                symbol.upper(),
                timeframe.value
            )
        
        return {
            "status": "training_started",
            "symbol": symbol.upper(),
            "timeframe": timeframe.value,
            "message": f"Training started for {symbol.upper()} ({timeframe.value}). This may take several minutes.",
            "check_status_endpoint": f"/api/model_status/{symbol.upper()}?timeframe={timeframe.value}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")
```

### Option C: Combined Approach (Best UX)
1. Add model check in `predict_next_price` that returns a dict instead of raising exception
2. Add training endpoint
3. Update frontend to show "Train Model" button

---

## Recommended Implementation: Option A + B

### Step 1: Update `routes/prediction.py`
- Add `FileNotFoundError` handler in `get_prediction`
- Add new `POST /api/train/{symbol}` endpoint
- Add new `GET /api/model_status/{symbol}` endpoint

### Step 2: Update `controllers/prediction.py`
- Modify `predict_next_price` to return a dict with `status: "no_model"` instead of raising `FileNotFoundError`
- This allows the route to handle it gracefully

### Step 3: Add Model Status Check
- Add helper function to check if model exists and return metadata

---

## Detailed Changes

### File 1: `routes/prediction.py`

**Current imports:**
```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from enum import Enum
from controllers.prediction import predict_next_price
```

**Add imports:**
```python
import os
from config import settings
```

**Modify `get_prediction` function (lines 14-39):**
- Add `FileNotFoundError` handler before generic `Exception` handler
- Return structured 404 response with training prompt

**Add new endpoints:**
```python
@router.post("/train/{symbol}", operation_id="train_model_for_symbol")
async def train_model_endpoint(symbol: str, timeframe: Timeframe):
    """Trigger model training for a symbol"""

@router.get("/model_status/{symbol}", operation_id="check_model_status")
async def get_model_status(symbol: str, timeframe: Timeframe):
    """Check if model exists and return status"""
```

### File 2: `controllers/prediction.py`

**Modify `predict_next_price` (line 958-966):**
- Change from raising `FileNotFoundError` to returning a dict:
```python
if not os.path.exists(model_path):
    return {
        "status": "no_model",
        "symbol": symbol,
        "timeframe": timeframe,
        "error": f"No trained model found for {symbol} {timeframe}"
    }
```

---

## Response Schema

### Missing Model Response (404):
```json
{
  "status": "no_model",
  "symbol": "ETH",
  "timeframe": "1h",
  "message": "No trained model found for ETH (1h).",
  "train_endpoint": "/api/train/ETH?timeframe=1h",
  "action_required": true
}
```

### Training Started Response (200):
```json
{
  "status": "training_started",
  "symbol": "ETH",
  "timeframe": "1h",
  "message": "Training started for ETH (1h). This may take several minutes.",
  "check_status_endpoint": "/api/model_status/ETH?timeframe=1h"
}
```

### Model Status Response (200):
```json
{
  "status": "ready",
  "symbol": "BTC",
  "timeframe": "1h",
  "model_path": "models/BTC/model_1h.pth",
  "last_trained": "2025-05-27T10:30:00Z"
}
```

---

## Files to Modify

1. **`routes/prediction.py`** - Main changes
   - Add `FileNotFoundError` handler
   - Add `/api/train/{symbol}` endpoint
   - Add `/api/model_status/{symbol}` endpoint

2. **`controllers/prediction.py`** - Minor change
   - Modify `predict_next_price` to return dict instead of raising exception

---

## Testing

1. Test missing model returns 404 with proper JSON
2. Test training endpoint starts training
3. Test model status endpoint returns correct status
4. Test existing model still works normally

---

## Edge Cases

1. **Symbol with spaces**: Validate and strip
2. **Invalid timeframe**: Already handled by Enum
3. **Training already in progress**: Check if model directory exists but is incomplete
4. **Training fails**: Return error status from training endpoint

---

## Implementation Order

1. Modify `controllers/prediction.py` - Change `predict_next_price` to return dict for missing model
2. Modify `routes/prediction.py` - Add FileNotFoundError handler and new endpoints
3. Test all endpoints
