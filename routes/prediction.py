from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from enum import Enum
from controllers.prediction import predict_next_price
import os
from config import settings
import asyncio
from concurrent.futures import ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class Timeframe(str, Enum):
    thirty_min = "30m"
    one_hour = "1h"
    four_hour = "4h"
    twenty_four_hour = "24h"

@router.get("/predict/{symbol}", operation_id="get_prediction_for_symbol")
async def get_prediction(symbol: str, timeframe: Timeframe, auto_train: bool = False):
    """
    Get a prediction for a given symbol and timeframe.
    This endpoint is public and requires no authentication.
    
    Args:
        symbol: Trading symbol (e.g., BTC, ETH)
        timeframe: Prediction timeframe (30m, 1h, 4h, 24h)
        auto_train: If True, automatically train model if missing before predicting.
                   If False (default), return a prompt to train the model.
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            raise ValueError("Symbol cannot be empty")

        symbol = symbol.upper()
        prediction_result = await predict_next_price(symbol, timeframe.value)

        # Check if model was not found
        if isinstance(prediction_result, dict) and prediction_result.get("status") == "no_model":
            if auto_train:
                # Auto-train: train the model synchronously, then predict
                from controllers.model_trainer import ModelTrainer
                import concurrent.futures

                logger.info(f"Auto-training model for {symbol} ({timeframe.value})...")

                def train_sync(sym, tf):
                    trainer = ModelTrainer(sym, tf)
                    return trainer.train()

                # Run training in a thread pool to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    training_result = await loop.run_in_executor(
                        pool, train_sync, symbol, timeframe.value
                    )

                if training_result.get("status") != "success":
                    return JSONResponse(
                        status_code=500,
                        content={
                            "status": "training_failed",
                            "symbol": symbol,
                            "timeframe": timeframe.value,
                            "message": f"Auto-training failed for {symbol} ({timeframe.value}).",
                            "error": training_result.get("error", "Unknown training error"),
                            "train_endpoint": f"/api/train/{symbol}?timeframe={timeframe.value}"
                        }
                    )

                logger.info(f"Auto-training complete for {symbol} ({timeframe.value}). Fetching prediction...")

                # Now fetch the prediction with the newly trained model
                prediction_result = await predict_next_price(symbol, timeframe.value)

                # If still no model after training, something went wrong
                if isinstance(prediction_result, dict) and prediction_result.get("status") == "no_model":
                    return JSONResponse(
                        status_code=500,
                        content={
                            "status": "prediction_failed_after_train",
                            "symbol": symbol,
                            "timeframe": timeframe.value,
                            "message": f"Model trained but prediction failed for {symbol} ({timeframe.value}).",
                            "train_endpoint": f"/api/train/{symbol}?timeframe={timeframe.value}"
                        }
                    )
            else:
                # No auto_train: return the prompt to train
                return JSONResponse(
                    status_code=404,
                    content={
                        "status": "no_model",
                        "symbol": prediction_result["symbol"],
                        "timeframe": prediction_result["timeframe"],
                        "message": f"No trained model found for {prediction_result['symbol']} ({prediction_result['timeframe']}).",
                        "train_endpoint": f"/api/train/{prediction_result['symbol']}?timeframe={prediction_result['timeframe']}",
                        "auto_train_tip": f"Set auto_train=true to automatically train: GET /api/predict/{prediction_result['symbol']}?timeframe={prediction_result['timeframe']}&auto_train=true",
                        "action_required": True
                    }
                )

        if isinstance(prediction_result, dict) and "error" in prediction_result:
            raise HTTPException(status_code=400, detail=prediction_result["error"])

        return prediction_result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            content={"detail": f"An unexpected internal server error occurred: {str(e)}"},
            status_code=500,
        )

@router.post("/train/{symbol}", operation_id="train_model_for_symbol")
async def train_model_endpoint(symbol: str, timeframe: Timeframe):
    """
    Trigger training for a given symbol and timeframe.
    Returns immediately with a training started message.
    Training runs in background.
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            raise ValueError("Symbol cannot be empty")

        from controllers.model_trainer import ModelTrainer

        symbol = symbol.upper()

        # Check if training is already in progress (model directory exists with partial files)
        model_dir = os.path.join(settings.MODEL_PATH, symbol)
        model_path = os.path.join(model_dir, f"model_{timeframe.value}.pth")

        if os.path.exists(model_path):
            return JSONResponse(
                status_code=200,
                content={
                    "status": "model_exists",
                    "symbol": symbol,
                    "timeframe": timeframe.value,
                    "message": f"A trained model already exists for {symbol} ({timeframe.value}).",
                    "prediction_endpoint": f"/api/predict/{symbol}?timeframe={timeframe.value}",
                    "action_required": False
                }
            )

        # Start training in background thread
        import threading

        def train_in_background(sym, tf):
            try:
                logger.info(f"Starting background training for {sym} ({tf})")
                trainer = ModelTrainer(sym, tf)
                result = trainer.train()
                logger.info(f"Training completed for {sym} ({tf}): {result.get('status', 'unknown')}")
            except Exception as e:
                logger.error(f"Background training failed for {sym} ({tf}): {str(e)}")

        training_thread = threading.Thread(
            target=train_in_background,
            args=(symbol, timeframe.value),
            daemon=True
        )
        training_thread.start()

        return {
            "status": "training_started",
            "symbol": symbol,
            "timeframe": timeframe.value,
            "message": f"Training started for {symbol} ({timeframe.value}). This may take several minutes.",
            "check_status_endpoint": f"/api/model_status/{symbol}?timeframe={timeframe.value}",
            "prediction_endpoint": f"/api/predict/{symbol}?timeframe={timeframe.value}"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start training for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/model_status/{symbol}", operation_id="check_model_status")
async def get_model_status(symbol: str, timeframe: Timeframe):
    """
    Check if a model exists for a given symbol and timeframe.
    Returns model status and metadata if available.
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            raise ValueError("Symbol cannot be empty")

        symbol = symbol.upper()
        model_dir = os.path.join(settings.MODEL_PATH, symbol)
        model_path = os.path.join(model_dir, f"model_{timeframe.value}.pth")
        scaler_path = os.path.join(model_dir, f"scaler_{timeframe.value}.joblib")
        metadata_path = os.path.join(model_dir, f"metadata_{timeframe.value}.json")

        model_exists = os.path.exists(model_path)
        scaler_exists = os.path.exists(scaler_path)

        result = {
            "symbol": symbol,
            "timeframe": timeframe.value,
            "model_exists": model_exists,
            "scaler_exists": scaler_exists,
            "model_path": model_path if model_exists else None,
        }

        # Load metadata if available
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            result["metadata"] = metadata
            result["last_trained"] = metadata.get("training_date", None)

        if model_exists:
            result["status"] = "ready"
            result["message"] = f"Model ready for {symbol} ({timeframe.value})."
            result["prediction_endpoint"] = f"/api/predict/{symbol}?timeframe={timeframe.value}"
        else:
            result["status"] = "no_model"
            result["message"] = f"No trained model found for {symbol} ({timeframe.value})."
            result["train_endpoint"] = f"/api/train/{symbol}?timeframe={timeframe.value}"

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check model status: {str(e)}")


@router.post("/models/rollback/{symbol}", operation_id="rollback_model")
async def rollback_model(symbol: str, timeframe: Timeframe):
    """
    Rollback a trained model to the previous champion checkpoint.

    Restores the champion backup checkpoint back to the active position,
    updates the model registry, and clears the prediction cache so live
    inference immediately reverts to the previous weights.

    Args:
        symbol: Trading symbol (e.g., BTC, ETH)
        timeframe: Model timeframe (30m, 1h, 4h, 24h)

    Returns:
        JSON response confirming the rollback version and previous performance metrics.
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            raise ValueError("Symbol cannot be empty")

        symbol = symbol.upper()

        from controllers.model_versioning import rollback as do_rollback

        result = do_rollback(str(settings.MODEL_PATH), symbol, timeframe.value)
        return result

    except FileNotFoundError as e:
        return JSONResponse(
            status_code=404,
            content={
                "status": "rollback_failed",
                "symbol": symbol,
                "timeframe": timeframe.value,
                "error": str(e),
                "message": f"No previous champion checkpoint found for {symbol} ({timeframe.value}). "
                           f"Cannot rollback without a backup.",
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Rollback failed for {symbol} ({timeframe.value}): {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "rollback_failed",
                "symbol": symbol,
                "timeframe": timeframe.value,
                "error": str(e),
                "message": f"Rollback failed for {symbol} ({timeframe.value}).",
            }
        )