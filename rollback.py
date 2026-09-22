"""CLI utility for rolling back a model to the previous champion checkpoint.

Usage:
    python rollback.py --symbol BTC --timeframe 1h
    python rollback.py --symbol ETH --timeframe 4h --list
"""

import argparse
import json
import sys
import logging
from pathlib import Path

from config import settings
from controllers.model_versioning import rollback, load_registry, get_champion_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def list_versions(symbol: str) -> None:
    """Print all recorded versions for a symbol from the registry."""
    registry = load_registry(str(settings.MODEL_PATH), symbol)
    if not registry:
        print(f"No version history found for {symbol}.")
        return

    print(f"\nVersion history for {symbol}:")
    print("=" * 60)
    for tf, entry in sorted(registry.items()):
        versions = entry.get("versions", [])
        last_promoted = entry.get("last_promoted", "N/A")
        last_rollback = entry.get("last_rollback", "N/A")
        print(f"\n  Timeframe: {tf}")
        print(f"  Active version:    {entry.get('active_version', 'N/A')}")
        print(f"  Last promoted:     {last_promoted}")
        print(f"  Last rollback:     {last_rollback}")
        print(f"  Version records:   {len(versions)}")
        if versions:
            print("  History (last 5):")
            for i, v in enumerate(versions[-5:]):
                idx = len(versions) - len(versions[-5:]) + i + 1
                if "promoted_at" in v:
                    print(f"    [{idx}] PROMOTED at {v['promoted_at']}")
                    champ = v.get("champion_metrics", {})
                    cand = v.get("candidate_metrics", {})
                    if champ:
                        print(f"         Champion: R2={champ.get('best_val_r2', 'N/A')}, Loss={champ.get('best_val_loss', 'N/A')}")
                    if cand:
                        print(f"         Candidate: R2={cand.get('best_val_r2', 'N/A')}, Loss={cand.get('best_val_loss', 'N/A')}")
                elif "rolled_back_at" in v:
                    print(f"    [{idx}] ROLLED BACK at {v['rolled_back_at']}")
                    restored = v.get("restored_metrics", {})
                    if restored:
                        print(f"         Restored: R2={restored.get('best_val_r2', 'N/A')}, Loss={restored.get('best_val_loss', 'N/A')}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rollback a trained model to the previous champion checkpoint"
    )
    parser.add_argument("--symbol", type=str, required=True, help="Symbol to rollback (e.g., BTC, ETH)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (default: 1h)")
    parser.add_argument(
        "--list", action="store_true", dest="list_versions",
        help="List version history for the symbol instead of rolling back",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    timeframe = args.timeframe

    if args.list_versions:
        list_versions(symbol)
        return

    # Show current champion metrics before rollback
    print(f"\nChecking current champion for {symbol} ({timeframe})...")
    current = get_champion_metrics(str(settings.MODEL_PATH), symbol, timeframe)
    if current:
        print(f"  Current champion: R2={current.get('best_val_r2', 'N/A')}, Loss={current.get('best_val_loss', 'N/A')}")
    else:
        print("  No current champion metrics found.")

    # Perform rollback
    print(f"\nRolling back {symbol} ({timeframe}) to previous champion...")
    try:
        result = rollback(str(settings.MODEL_PATH), symbol, timeframe)
        print(f"\nRollback successful!")
        print(f"  Status:   {result['status']}")
        print(f"  Message:  {result['message']}")
        restored = result.get("restored_metrics", {})
        if restored:
            print(f"  Restored metrics:")
            print(f"    Val R2:   {restored.get('best_val_r2', 'N/A')}")
            print(f"    Val Loss: {restored.get('best_val_loss', 'N/A')}")
            print(f"    Val MAPE: {restored.get('final_val_mape', 'N/A')}")
            print(f"    Trained:  {restored.get('training_date', 'N/A')}")
            print(f"    Epochs:   {restored.get('epochs_trained', 'N/A')}")
    except FileNotFoundError as e:
        print(f"\nRollback failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nRollback failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
