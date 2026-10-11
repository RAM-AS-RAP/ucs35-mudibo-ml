"""Load checksum-verified sklearn joblib components and portable XGBoost JSON.

Joblib files can execute code during loading, so only checksum-verified files
from this repository are loaded. Archived original pickle bundles are not loaded
because their XGBoost payloads do not deserialize across environments.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from importlib import metadata

MODEL_DIR = Path(__file__).resolve().parents[3] / "results" / "artifacts" / "models"
CHECKSUM_PATH = MODEL_DIR / "MODEL_CHECKSUMS.json"
EXPECTED_VERSIONS = {"scikit-learn": "1.6.1", "xgboost": "3.4.1"}
EXPECTED_FEATURE_COUNT = 14
EXPECTED_FEATURE_COLS = [
    "log_return_lag_1", "log_return_lag_2", "log_return_lag_3",
    "log_ratio_3m", "log_ratio_12m", "month_sin", "month_cos", "year",
    "market_Nairobi", "market_Kisumu", "market_Mombasa", "market_Nakuru",
    "commodity_Maize", "commodity_Beans",
]
MODEL_PATHS = {
    "h1": {
        "xgb": MODEL_DIR / "xgb_h1.json",
        "sklearn": MODEL_DIR / "stack_h1_sklearn.joblib",
    },
    "h2": {
        "xgb": MODEL_DIR / "xgb_h2.json",
        "sklearn": MODEL_DIR / "stack_h2_sklearn.joblib",
    },
}


class ModelLoadError(RuntimeError):
    """A trusted, compatible model bundle could not be loaded."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checksums() -> dict:
    if not CHECKSUM_PATH.is_file():
        raise ModelLoadError(f"Model checksum file is missing: {CHECKSUM_PATH}; run scripts/write_model_checksums.py")
    try:
        checksums = json.loads(CHECKSUM_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exception:
        raise ModelLoadError(f"Could not read model checksum file {CHECKSUM_PATH}: {exception}") from exception
    if not isinstance(checksums, dict):
        raise ModelLoadError(f"Model checksum file must contain a mapping: {CHECKSUM_PATH}")
    return checksums


def _expected_digest(checksums: dict, path: Path) -> str:
    expected_digest = checksums.get(path.name)
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ModelLoadError(f"Checksum entry for {path.name} is missing or invalid in {CHECKSUM_PATH}")
    return expected_digest


def _verify_checksums(paths: dict[str, Path]) -> None:
    checksums = _load_checksums()
    for path in paths.values():
        expected = _expected_digest(checksums, path)
        if _sha256(path).lower() != expected.lower():
            raise ModelLoadError(f"SHA-256 mismatch for {path.name}; refusing to load the model")


def _verify_versions() -> None:
    installed = {}
    for package in EXPECTED_VERSIONS:
        try:
            installed[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed[package] = "not installed"
    if installed["scikit-learn"] != EXPECTED_VERSIONS["scikit-learn"]:
        raise ModelLoadError(
            "Saved sklearn components require scikit-learn 1.6.1; "
            f"installed scikit-learn={installed['scikit-learn']}, xgboost={installed['xgboost']}"
        )
    if installed["xgboost"] != EXPECTED_VERSIONS["xgboost"]:
        print(
            "WARNING: repackaged XGBoost JSON was exported with xgboost 3.4.1; "
            f"installed scikit-learn={installed['scikit-learn']}, xgboost={installed['xgboost']}. "
            "Golden parity tests verify prediction compatibility."
        )


def _load_estimators(paths: dict[str, Path]) -> tuple[dict, object]:
    import joblib
    import xgboost

    try:
        bundle = joblib.load(paths["sklearn"])
    except Exception as exception:
        raise ModelLoadError(
            f"Could not load checksum-verified {paths['sklearn'].name}: "
            f"{type(exception).__name__}: {exception}"
        ) from exception
    try:
        xgb_model = xgboost.XGBRegressor()
        xgb_model.load_model(paths["xgb"])
    except Exception as exception:
        raise ModelLoadError(
            f"Could not load checksum-verified {paths['xgb'].name}: "
            f"{type(exception).__name__}: {exception}"
        ) from exception
    return bundle, xgb_model


def _validate_bundle(bundle, xgb_model, horizon: str) -> dict:
    required = {"rf_model", "ridge_meta", "feature_cols"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        missing = sorted(required.difference(bundle)) if isinstance(bundle, dict) else sorted(required)
        raise ModelLoadError(f"H{horizon[-1]} sklearn artifact is missing required keys: {missing}")
    feature_cols = bundle["feature_cols"]
    if (
        not isinstance(feature_cols, list)
        or len(feature_cols) != EXPECTED_FEATURE_COUNT
        or feature_cols != EXPECTED_FEATURE_COLS
    ):
        raise ModelLoadError("Saved feature_cols do not match the expected 14 names and order")
    if getattr(bundle["ridge_meta"], "n_features_in_", None) != 2:
        raise ModelLoadError("ridge_meta.n_features_in_ must equal 2")
    rf_features = getattr(bundle["rf_model"], "feature_names_in_", None)
    if rf_features is None or list(rf_features) != feature_cols:
        raise ModelLoadError("Random Forest feature_names_in_ do not match feature_cols order")
    booster = xgb_model.get_booster()
    xgb_features = booster.feature_names
    if xgb_features is not None:
        if list(xgb_features) != feature_cols:
            raise ModelLoadError("XGBoost booster feature names do not match feature_cols order")
    elif booster.num_features() != EXPECTED_FEATURE_COUNT:
        raise ModelLoadError(
            f"XGBoost booster has {booster.num_features()} features; expected {EXPECTED_FEATURE_COUNT}"
        )
    return {
        "rf_model": bundle["rf_model"], "xgb_model": xgb_model,
        "ridge_meta": bundle["ridge_meta"], "feature_cols": list(feature_cols), "horizon": horizon,
    }


def load_model(horizon: str) -> dict:
    """Load checksum-verified sklearn joblib components and portable XGBoost JSON."""
    if horizon not in MODEL_PATHS:
        raise ValueError("horizon must be 'h1' or 'h2'")
    paths = MODEL_PATHS[horizon]
    for path in paths.values():
        if not path.is_file():
            raise ModelLoadError(f"Model file is missing: {path}; restore it from the repository")
    _verify_checksums(paths)
    _verify_versions()
    bundle, xgb_model = _load_estimators(paths)
    return _validate_bundle(bundle, xgb_model, horizon)


def load_all_models() -> dict[str, dict]:
    """Load both verified horizons and require identical saved feature order."""
    models = {horizon: load_model(horizon) for horizon in ("h1", "h2")}
    if models["h1"]["feature_cols"] != models["h2"]["feature_cols"]:
        raise ModelLoadError("H1 and H2 model feature columns or order differ")
    return models
