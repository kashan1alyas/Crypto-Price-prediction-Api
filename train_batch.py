"""Batch training script for multiple crypto assets."""

import sys
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from controllers.model_trainer import ModelTrainer
from controllers.prediction import clear_model_cache, CRYPTO_SYMBOLS
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_training.log'),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL", "LINK"]


def train_one(symbol: str, timeframe: str, force_promote: bool = False) -> Dict:
    """Train a single symbol/timeframe pair. Returns result dict."""
    start = time.time()
    try:
        trainer = ModelTrainer(symbol, timeframe, force_promote=force_promote)
        result = trainer.train()
        elapsed = time.time() - start
        result["elapsed_seconds"] = round(elapsed, 1)
        return result
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"{symbol} {timeframe} failed after {elapsed:.1f}s: {e}")
        return {"status": "error", "error": str(e), "elapsed_seconds": round(elapsed, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-train BiLSTM crypto models")
    parser.add_argument(
        "--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
        help=f"Comma-separated crypto symbols (default: {','.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument(
        "--override", action="store_true",
        help="Force-promote candidate models even if they don't beat the champion",
    )
    args = parser.parse_args()

    # Split comma-separated symbols into a list
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Crypto asset classification safeguard: warn if any symbol is not in CRYPTO_SYMBOLS
    non_crypto = [s for s in symbols if s not in CRYPTO_SYMBOLS]
    if non_crypto:
        logger.warning(
            f"The following symbols are NOT in CRYPTO_SYMBOLS and may be routed to stock "
            f"data sources instead of CoinGecko: {', '.join(non_crypto)}. "
            f"Known crypto symbols: {', '.join(sorted(CRYPTO_SYMBOLS))}"
        )

    if args.override:
        logger.info("--override flag set: candidates will be force-promoted even if they don't beat the champion")

    results: List[Dict] = []
    total_start = time.time()

    for symbol in symbols:
        logger.info(f"--- Training {symbol} ({args.timeframe}) ---")
        result = train_one(symbol, args.timeframe, force_promote=args.override)
        result["symbol"] = symbol
        result["timeframe"] = args.timeframe
        results.append(result)

        if result["status"] == "success":
            promoted = result.get("promoted", False)
            if promoted:
                logger.info(
                    f"{symbol}: PROMOTED val_loss={result['best_val_loss']:.6f}  "
                    f"epochs={result['epochs_trained']}  "
                    f"time={result['elapsed_seconds']}s  "
                    f"reason={result.get('promotion_reason', 'N/A')}"
                )
            else:
                logger.info(
                    f"{symbol}: REJECTED val_loss={result['best_val_loss']:.6f}  "
                    f"epochs={result['epochs_trained']}  "
                    f"time={result['elapsed_seconds']}s  "
                    f"reason={result.get('promotion_reason', 'N/A')}"
                )
        else:
            logger.warning(f"{symbol}: {result.get('error', 'unknown error')}")

    total_elapsed = time.time() - total_start
    successful = [r for r in results if r.get("status") == "success"]
    promoted = [r for r in successful if r.get("promoted", False)]
    rejected = [r for r in successful if not r.get("promoted", False)]
    failed = [r for r in results if r.get("status") != "success"]

    # Ensure all trained models are evicted from cache
    clear_model_cache()

    logger.info("=" * 60)
    logger.info("Batch training complete")
    logger.info(f"  Total:     {len(results)}")
    logger.info(f"  Success:   {len(successful)} (Promoted: {len(promoted)}, Rejected: {len(rejected)})")
    logger.info(f"  Failed:    {len(failed)}")
    logger.info(f"  Wall time: {total_elapsed:.1f}s")

    if successful:
        logger.info("\nSuccessful models:")
        for r in successful:
            status_tag = "PROMOTED" if r.get("promoted") else "REJECTED"
            logger.info(
                f"  [{status_tag}] {r['symbol']} ({r['timeframe']}): "
                f"val_loss={r['best_val_loss']:.6f}  "
                f"val_r2={r['best_val_r2']:.4f}  "
                f"epochs={r['epochs_trained']}  "
                f"time={r['elapsed_seconds']}s  "
                f"reason={r.get('promotion_reason', 'N/A')}"
            )

    if failed:
        logger.info("\nFailed models:")
        for r in failed:
            logger.info(f"  {r['symbol']} ({r['timeframe']}): {r.get('error')}")

    # Persist batch summary
    summary_path = Path(settings.MODEL_PATH) / "batch_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "timeframe": args.timeframe,
        "symbols": symbols,
        "total_seconds": round(total_elapsed, 1),
        "successful": len(successful),
        "promoted": len(promoted),
        "rejected": len(rejected),
        "failed": len(failed),
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
