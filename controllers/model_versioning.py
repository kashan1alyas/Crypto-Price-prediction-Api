"""Model versioning, champion-challenger validation, and rollback system.

Manages checkpoint backup, versioning, promotion, and rollback for trained models.
All checkpoint files follow the naming convention:
    models/{symbol}/model_{timeframe}.pth               # Active champion
    models/{symbol}/model_{timeframe}_champion.pth      # Backup of previous champion
    models/{symbol}/model_{timeframe}_candidate.pth     # Newly trained candidate
"""

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


REGISTRY_FILENAME = "model_registry.json"


def _registry_path(model_path: str, symbol: str) -> Path:
    """Return path to model_registry.json for a symbol."""
    return Path(model_path) / symbol / REGISTRY_FILENAME


def load_registry(model_path: str, symbol: str) -> Dict[str, Any]:
    """Load the registry for a symbol, returning empty dict if missing."""
    path = _registry_path(model_path, symbol)
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Corrupt registry at {path}, starting fresh: {e}")
        return {}


def save_registry(model_path: str, symbol: str, registry: Dict[str, Any]) -> None:
    """Persist the registry for a symbol."""
    path = _registry_path(model_path, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2)
    tmp_path.replace(path)


def _checkpoint_paths(model_path: str, symbol: str, timeframe: str):
    """Return (active, champion_backup, candidate) Path objects."""
    base = Path(model_path) / symbol
    active = base / f"model_{timeframe}.pth"
    champion_backup = base / f"model_{timeframe}_champion.pth"
    candidate = base / f"model_{timeframe}_candidate.pth"
    return active, champion_backup, candidate


def _scaler_paths(model_path: str, symbol: str, timeframe: str):
    """Return (active_scaler, backup_scaler) Path objects."""
    base = Path(model_path) / symbol
    active = base / f"scaler_{timeframe}.joblib"
    backup = base / f"scaler_{timeframe}_champion.joblib"
    return active, backup


def _feature_scaler_paths(model_path: str, symbol: str):
    """Return (active, backup) feature_scaler paths."""
    base = Path(model_path) / symbol
    active = base / "feature_scaler.joblib"
    backup = base / "feature_scaler_champion.joblib"
    return active, backup


def _metadata_paths(model_path: str, symbol: str, timeframe: str):
    """Return (active, backup) metadata paths."""
    base = Path(model_path) / symbol
    active = base / f"metadata_{timeframe}.json"
    backup = base / f"metadata_{timeframe}_champion.json"
    return active, backup


def backup_champion(model_path: str, symbol: str, timeframe: str) -> bool:
    """Copy the active checkpoint + scaler + metadata to champion backup files.

    Returns True if a backup was created (active checkpoint existed).
    """
    active_ckpt, champion_ckpt, _ = _checkpoint_paths(model_path, symbol, timeframe)
    active_scaler, champion_scaler = _scaler_paths(model_path, symbol, timeframe)
    active_feat, champion_feat = _feature_scaler_paths(model_path, symbol)
    active_meta, champion_meta = _metadata_paths(model_path, symbol, timeframe)

    if not active_ckpt.exists():
        logger.info(f"No active checkpoint to backup for {symbol} {timeframe}")
        return False

    shutil.copy2(active_ckpt, champion_ckpt)
    logger.info(f"Backed up champion checkpoint: {active_ckpt} -> {champion_ckpt}")

    if active_scaler.exists():
        shutil.copy2(active_scaler, champion_scaler)
    if active_feat.exists():
        shutil.copy2(active_feat, champion_feat)
    if active_meta.exists():
        shutil.copy2(active_meta, champion_meta)

    return True


def save_candidate(model_path: str, symbol: str, timeframe: str,
                   checkpoint_data: Dict[str, Any],
                   scaler=None, feature_scaler=None,
                   metadata: Dict[str, Any] = None) -> None:
    """Save a newly trained candidate checkpoint alongside its scaler/metadata."""
    base = Path(model_path) / symbol
    base.mkdir(parents=True, exist_ok=True)

    _, _, candidate_ckpt = _checkpoint_paths(model_path, symbol, timeframe)
    torch = __import__("torch")
    torch.save(checkpoint_data, candidate_ckpt)
    logger.info(f"Saved candidate checkpoint: {candidate_ckpt}")

    if scaler is not None:
        import joblib
        candidate_scaler = base / f"scaler_{timeframe}_candidate.joblib"
        joblib.dump(scaler, candidate_scaler)

    if feature_scaler is not None:
        import joblib
        candidate_feat = base / "feature_scaler_candidate.joblib"
        joblib.dump(feature_scaler, candidate_feat)

    if metadata is not None:
        candidate_meta = base / f"metadata_{timeframe}_candidate.json"
        with open(candidate_meta, "w") as f:
            json.dump(metadata, f, indent=2)


def get_active_schema(model_path: str, symbol: str, timeframe: str) -> Optional[list]:
    """Return the ordered feature list for the active version of a model.

    Reads from model_registry.json. Returns None if no schema is recorded.
    """
    registry = load_registry(model_path, symbol)
    entry = registry.get(timeframe, {})
    active_version = entry.get("active_version")
    versions = entry.get("versions", [])
    if not versions or active_version is None:
        return None
    idx = active_version - 1  # active_version is 1-indexed
    if idx < 0 or idx >= len(versions):
        return None
    return versions[idx].get("feature_names")


def get_active_input_size(model_path: str, symbol: str, timeframe: str) -> Optional[int]:
    """Return the input_size for the active version of a model.

    Reads from model_registry.json. Returns None if not recorded.
    """
    registry = load_registry(model_path, symbol)
    entry = registry.get(timeframe, {})
    active_version = entry.get("active_version")
    versions = entry.get("versions", [])
    if not versions or active_version is None:
        return None
    idx = active_version - 1
    if idx < 0 or idx >= len(versions):
        return None
    return versions[idx].get("input_size")


def promote_candidate(model_path: str, symbol: str, timeframe: str,
                      champion_metrics: Dict[str, Any],
                      candidate_metrics: Dict[str, Any],
                      feature_list: list = None,
                      feature_names: list = None,
                      input_size: int = None) -> None:
    """Promote the candidate to active champion.

    Overwrites the active checkpoint with the candidate and records the
    promotion in the model_registry.json.
    """
    import shutil as _shutil

    active_ckpt, _, candidate_ckpt = _checkpoint_paths(model_path, symbol, timeframe)
    active_scaler, champion_scaler = _scaler_paths(model_path, symbol, timeframe)
    active_feat, champion_feat = _feature_scaler_paths(model_path, symbol)
    active_meta, champion_meta = _metadata_paths(model_path, symbol, timeframe)
    base = Path(model_path) / symbol

    # Move candidate -> active
    if candidate_ckpt.exists():
        _shutil.copy2(candidate_ckpt, active_ckpt)
        candidate_ckpt.unlink()
        logger.info(f"Promoted candidate -> active: {active_ckpt}")

    # Move candidate scaler -> active
    candidate_scaler = base / f"scaler_{timeframe}_candidate.joblib"
    if candidate_scaler.exists():
        _shutil.copy2(candidate_scaler, active_scaler)
        candidate_scaler.unlink()

    candidate_feat = base / "feature_scaler_candidate.joblib"
    if candidate_feat.exists():
        _shutil.copy2(candidate_feat, active_feat)
        candidate_feat.unlink()

    candidate_meta = base / f"metadata_{timeframe}_candidate.json"
    if candidate_meta.exists():
        _shutil.copy2(candidate_meta, active_meta)
        candidate_meta.unlink()

    # Clean up stale champion backup (the old champion backup may exist)
    _, old_champion_ckpt, _ = _checkpoint_paths(model_path, symbol, timeframe)
    if old_champion_ckpt.exists():
        old_champion_ckpt.unlink()
        logger.info(f"Removed stale champion backup: {old_champion_ckpt}")

    # Update registry
    registry = load_registry(model_path, symbol)
    entry = registry.get(timeframe, {})
    versions = entry.get("versions", [])

    version_record = {
        "promoted_at": datetime.utcnow().isoformat(),
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "feature_list": feature_list or [],
        "feature_names": feature_names or feature_list or [],
        "input_size": input_size,
    }
    versions.append(version_record)

    # Keep only the last 20 version records
    if len(versions) > 20:
        versions = versions[-20:]

    registry[timeframe] = {
        "active_version": len(versions),
        "last_promoted": datetime.utcnow().isoformat(),
        "champion_metrics": candidate_metrics,
        "versions": versions,
    }
    save_registry(model_path, symbol, registry)
    logger.info(f"Registry updated for {symbol} {timeframe}, version {len(versions)}")


def discard_candidate(model_path: str, symbol: str, timeframe: str) -> None:
    """Remove candidate checkpoint and associated files (champion wins)."""
    base = Path(model_path) / symbol
    _, _, candidate_ckpt = _checkpoint_paths(model_path, symbol, timeframe)
    candidate_scaler = base / f"scaler_{timeframe}_candidate.joblib"
    candidate_feat = base / "feature_scaler_candidate.joblib"
    candidate_meta = base / f"metadata_{timeframe}_candidate.json"

    for f in [candidate_ckpt, candidate_scaler, candidate_feat, candidate_meta]:
        if f.exists():
            f.unlink()
            logger.info(f"Discarded candidate file: {f}")


def rollback(model_path: str, symbol: str, timeframe: str) -> Dict[str, Any]:
    """Rollback to the previous champion checkpoint.

    Restores the champion backup files back to active positions,
    clears the prediction cache, and returns a status dict.

    Returns:
        Dict with status, restored metrics, and version info.

    Raises:
        FileNotFoundError: If no champion backup exists to rollback to.
    """
    active_ckpt, champion_ckpt, _ = _checkpoint_paths(model_path, symbol, timeframe)
    active_scaler, champion_scaler = _scaler_paths(model_path, symbol, timeframe)
    active_feat, champion_feat = _feature_scaler_paths(model_path, symbol)
    active_meta, champion_meta = _metadata_paths(model_path, symbol, timeframe)

    if not champion_ckpt.exists():
        raise FileNotFoundError(
            f"No champion backup found for {symbol} {timeframe}. "
            f"Cannot rollback without a previous checkpoint."
        )

    # Restore champion -> active
    shutil.copy2(champion_ckpt, active_ckpt)
    logger.info(f"Restored champion checkpoint: {champion_ckpt} -> {active_ckpt}")

    if champion_scaler.exists():
        shutil.copy2(champion_scaler, active_scaler)
    if champion_feat.exists():
        shutil.copy2(champion_feat, active_feat)
    if champion_meta.exists():
        shutil.copy2(champion_meta, active_meta)

    # Read restored metadata for response
    restored_metrics = {}
    if active_meta.exists():
        try:
            with open(active_meta, "r") as f:
                meta = json.load(f)
            restored_metrics = {
                "best_val_loss": meta.get("best_val_loss"),
                "best_val_r2": meta.get("best_val_r2"),
                "final_val_mape": meta.get("final_val_mape"),
                "training_date": meta.get("training_date"),
                "epochs_trained": meta.get("epochs_trained"),
            }
        except Exception:
            pass

    # Update registry — find the previous version's schema and set it as active
    registry = load_registry(model_path, symbol)
    entry = registry.get(timeframe, {})
    versions = entry.get("versions", [])

    # Find the most recent promotion record (before this rollback) to get the schema
    restored_feature_names = None
    restored_input_size = None
    for v in reversed(versions):
        if "feature_names" in v and v.get("feature_names"):
            restored_feature_names = v["feature_names"]
            restored_input_size = v.get("input_size")
            break

    rollback_record = {
        "rolled_back_at": datetime.utcnow().isoformat(),
        "restored_metrics": restored_metrics,
        "restored_feature_names": restored_feature_names,
        "restored_input_size": restored_input_size,
    }
    versions.append(rollback_record)

    # Set active_version to point at the restored schema version
    new_active_version = len(versions)  # 1-indexed

    registry[timeframe] = {
        **entry,
        "active_version": new_active_version,
        "last_rollback": datetime.utcnow().isoformat(),
        "versions": versions[-20:],
    }
    save_registry(model_path, symbol, registry)

    # Clear prediction cache
    try:
        from controllers.prediction import clear_model_cache
        clear_model_cache(symbol, timeframe)
        logger.info(f"Cleared prediction cache for {symbol} {timeframe}")
    except ImportError:
        logger.warning("Could not import clear_model_cache — cache not invalidated")

    return {
        "status": "rollback_success",
        "symbol": symbol,
        "timeframe": timeframe,
        "message": f"Rolled back to previous champion for {symbol} ({timeframe}).",
        "restored_metrics": restored_metrics,
        "restored_feature_names": restored_feature_names,
        "restored_input_size": restored_input_size,
        "version_info": entry,
    }


def compare_metrics(champion: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """Compare champion vs candidate metrics.

    The candidate must have a higher val_r2 AND lower val_loss to be promoted.
    If the candidate has only one improvement it is still considered a win
    (any improvement is acceptable).

    Returns:
        (candidate_is_better, reason)
    """
    champ_r2 = champion.get("best_val_r2", 0)
    cand_r2 = candidate.get("best_val_r2", 0)
    champ_loss = champion.get("best_val_loss", float("inf"))
    cand_loss = candidate.get("best_val_loss", float("inf"))

    r2_improved = cand_r2 > champ_r2
    loss_improved = cand_loss < champ_loss

    if r2_improved and loss_improved:
        return True, f"Both metrics improved: R2 {champ_r2:.4f}->{cand_r2:.4f}, Loss {champ_loss:.6f}->{cand_loss:.6f}"
    elif r2_improved:
        return True, f"R2 improved: {champ_r2:.4f}->{cand_r2:.4f} (loss: {champ_loss:.6f}->{cand_loss:.6f})"
    elif loss_improved:
        return True, f"Loss improved: {champ_loss:.6f}->{cand_loss:.6f} (R2: {champ_r2:.4f}->{cand_r2:.4f})"
    else:
        return False, f"No improvement: R2 {champ_r2:.4f}->{cand_r2:.4f}, Loss {champ_loss:.6f}->{cand_loss:.6f}"


def get_champion_metrics(model_path: str, symbol: str, timeframe: str) -> Dict[str, Any]:
    """Read metrics from the active checkpoint's metadata."""
    _, meta_path, _ = _metadata_paths(model_path, symbol, timeframe)
    if not meta_path.exists():
        # Fallback: read from the checkpoint itself
        active_ckpt, _, _ = _checkpoint_paths(model_path, symbol, timeframe)
        if active_ckpt.exists():
            import torch
            ckpt = torch.load(active_ckpt, map_location="cpu", weights_only=False)
            return {
                "best_val_loss": ckpt.get("val_loss"),
                "best_val_r2": ckpt.get("val_r2"),
            }
        return {}
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return {
            "best_val_loss": meta.get("best_val_loss"),
            "best_val_r2": meta.get("best_val_r2"),
            "final_val_mape": meta.get("final_val_mape"),
            "feature_list": meta.get("feature_list", []),
        }
    except Exception:
        return {}
