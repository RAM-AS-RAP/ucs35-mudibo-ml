"""Evaluation metrics and model-scoring utilities."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_PREDICTION_COLUMNS = {
    "Persistence": "persistence_pred",
    "Linear Regression": "linear_regression_pred",
    "Random Forest": "random_forest_pred",
    "XGBoost": "xgboost_pred",
    "Stacking ensemble": "stacking_pred",
}


def reconstruct_price(origin_price: pd.Series, predicted_log_change: pd.Series) -> pd.Series:
    """Reconstruct KES/kg prices positionally, preserving the origin-price index."""
    origins = np.asarray(origin_price, dtype=float)
    changes = np.asarray(predicted_log_change, dtype=float)
    if origins.ndim != 1 or origins.shape != changes.shape or not len(origins):
        raise ValueError("Origin prices and changes must be nonempty equal-length vectors")
    if not np.isfinite(origins).all() or not np.isfinite(changes).all() or (origins <= 0).any():
        raise ValueError("Prices must be positive and prices/changes finite")
    with np.errstate(over="ignore", invalid="ignore"):
        prices = origins * np.exp(changes)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Reconstructed prices must be finite and positive")
    return pd.Series(prices, index=origin_price.index, name="reconstructed_price")


def compute_metrics(y_true_log_change, y_pred_log_change, origin_price, true_price) -> dict:
    """Compute log/price errors (MAPE is a percentage)."""
    actual = np.asarray(y_true_log_change, dtype=float)
    predicted = np.asarray(y_pred_log_change, dtype=float)
    origins = pd.Series(np.asarray(origin_price, dtype=float))
    actual_prices = np.asarray(true_price, dtype=float)
    if actual.ndim != 1 or not len(actual) or any(
        values.shape != actual.shape for values in (predicted, origins.to_numpy(), actual_prices)
    ):
        raise ValueError("Metric inputs must be nonempty equal-length vectors")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Log targets and predictions must be finite")
    reconstructed_actual = reconstruct_price(origins, pd.Series(actual)).to_numpy()
    reconstructed_predicted = reconstruct_price(origins, pd.Series(predicted)).to_numpy()
    if not np.isfinite(actual_prices).all() or (actual_prices <= 0).any():
        raise ValueError("True prices must be finite and positive")
    if not np.allclose(actual_prices, reconstructed_actual, rtol=1e-7, atol=1e-8):
        raise ValueError("True prices disagree with origin prices and true log changes")
    log_errors = predicted - actual
    price_errors = reconstructed_predicted - reconstructed_actual
    return {
        "mae_log": float(np.mean(np.abs(log_errors))),
        "rmse_log": float(np.sqrt(np.mean(log_errors ** 2))),
        "mape_price_pct": float(100 * np.mean(np.abs(price_errors) / reconstructed_actual)),
        "mae_price": float(np.mean(np.abs(price_errors))),
        "rmse_price": float(np.sqrt(np.mean(price_errors ** 2))),
    }


def build_results_table(predictions_df, market_col, commodity_col) -> pd.DataFrame:
    """Report each of five models per series and pooled over all test observations."""
    required = {
        market_col, commodity_col, "target", "origin_price", "true_price",
        *MODEL_PREDICTION_COLUMNS.values(),
    }
    missing = required.difference(predictions_df.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if predictions_df.empty or predictions_df[[market_col, commodity_col]].isna().any().any():
        raise ValueError("Prediction table must be nonempty with valid series identities")
    series_groups = list(predictions_df.groupby([market_col, commodity_col], sort=True))
    records = []
    for model_name, prediction_col in MODEL_PREDICTION_COLUMNS.items():
        for (market, commodity), group in [*series_groups, (("ALL", "ALL"), predictions_df)]:
            records.append({
                "model": model_name,
                market_col: market,
                commodity_col: commodity,
                "scope": "pooled" if market == "ALL" and commodity == "ALL" else "series",
                "n_rows": len(group),
                **compute_metrics(group["target"], group[prediction_col], group["origin_price"], group["true_price"]),
            })
    return pd.DataFrame(records)


def _ensure_test_not_closed(
    horizon: str,
    output_dir: Path,
    model_dir: Path,
    drive_output_dir: Path | None,
    drive_model_dir: Path | None,
) -> None:
    result_name = f"results_{horizon}.csv"
    model_name = f"stacking_{horizon}.pkl"
    marker_name = f"test_closed_{horizon}.json"
    for directory, filenames in (
        (output_dir, (marker_name, result_name)), (model_dir, (model_name,)),
        (drive_output_dir, (marker_name, result_name)), (drive_model_dir, (model_name,)),
    ):
        if directory is not None:
            for filename in filenames:
                if (directory / filename).exists():
                    raise RuntimeError(
                        f"{horizon.upper()} evaluation is closed or already started: {directory / filename}. "
                        "Read saved results instead of repeating inference; do not delete closure markers."
                    )


def _load_evaluation_inputs(horizon: str, config: dict) -> tuple[pd.DataFrame, dict, dict]:
    tuning = json.loads(Path(config["tuning_path"]).read_text(encoding="utf-8"))
    if tuning["horizon"] != horizon:
        raise ValueError("Tuning artifact horizon mismatch")
    table = pd.read_csv(config["data_path"], parse_dates=["month", "target_month"])
    folds = json.loads(Path(config["fold_path"]).read_text(encoding="utf-8"))
    return table, tuning, folds


def _validate_evaluation_inputs(
    table: pd.DataFrame, tuning: dict, folds: dict,
) -> tuple[np.ndarray, np.ndarray]:
    if folds["table_row_position"] != list(range(len(table))):
        raise ValueError("Saved folds do not match supervised-table row positions")
    development_mask = np.asarray(folds["development_mask"])
    test_mask = np.asarray(folds["test_mask"])
    if any(
        mask.dtype != np.dtype(bool) or mask.shape != (len(table),)
        for mask in (development_mask, test_mask)
    ):
        raise ValueError("Saved fold masks must be aligned boolean vectors")
    if (development_mask & test_mask).any() or not (development_mask | test_mask).all():
        raise ValueError("Development/test masks must be disjoint and cover the supervised table")
    expected_development = table["month"].between("2010-01-01", "2018-10-01").to_numpy()
    expected_test = table["month"].between("2018-11-01", "2019-10-01").to_numpy()
    if (
        not np.array_equal(development_mask, expected_development)
        or not np.array_equal(test_mask, expected_test)
    ):
        raise ValueError("Saved origin-month masks do not match the frozen Task 1 protocol")
    if (
        int(development_mask.sum()) != tuning["development_rows"]
        or int(test_mask.sum()) != tuning["test_rows_reserved_not_scored"]
    ):
        raise ValueError("Task 2 row counts do not match Task 1 supervised tables")
    return development_mask, test_mask


def _save_model_bundle(horizon: str, models: dict, directories: tuple[Path, ...]) -> None:
    import joblib

    for directory in directories:
        joblib.dump(models, directory / f"stacking_{horizon}.pkl")


def _mark_test_closed(
    horizon: str, feature_cols: list[str], directories: tuple[Path, ...],
) -> dict:
    closure = {
        "horizon": horizon, "status": "test evaluation started; test closed to further tuning",
        "feature_cols": feature_cols,
        "test_origin_start": "2018-11-01", "test_origin_end": "2019-10-01",
    }
    for directory in directories:
        with (directory / f"test_closed_{horizon}.json").open("x", encoding="utf-8") as marker:
            json.dump(closure, marker, indent=2)
    return closure


def _build_evaluation_manifest(
    closure: dict, tuning: dict, folds: dict, development_rows: int, test_rows: int,
) -> dict:
    import sys

    import joblib
    import sklearn
    import xgboost

    return {
        **closure, "status": "completed; held-out test closed",
        "development_rows": development_rows, "test_rows": test_rows,
        "rf_params": tuning["random_forest"]["best_params"],
        "xgb_params": tuning["xgboost"]["best_params"],
        "ridge_alpha": tuning["ridge_meta"]["alpha"],
        "fold_boundaries": folds["folds"],
        "metric_units": {
            "mae_log": "log change", "rmse_log": "log change",
            "mape_price_pct": "percentage (not bounded above by 100)",
            "mae_price": "KES/kg", "rmse_price": "KES/kg",
        },
        "versions": {
            "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "joblib": joblib.__version__,
        },
    }


def _save_evaluation_outputs(
    horizon: str,
    results_table: pd.DataFrame,
    predictions: pd.DataFrame,
    manifest: dict,
    directories: tuple[Path, ...],
) -> None:
    for directory in directories:
        results_table.to_csv(directory / f"results_{horizon}.csv", index=False)
        predictions.to_csv(directory / f"test_predictions_{horizon}.csv", index=False)
        (directory / f"evaluation_manifest_{horizon}.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )


def run_final_evaluation(
    horizon: str,
    config: dict,
    output_dir: str | PathLike[str],
    model_dir: str | PathLike[str],
    drive_output_dir: str | PathLike[str] | None = None,
    drive_model_dir: str | PathLike[str] | None = None,
) -> pd.DataFrame:
    """Run and persist one frozen held-out evaluation, enforcing permanent closure."""
    from agrisence.training.ensemble.stacking import refit_final_stack, predict_test_set

    if horizon not in {"h1", "h2"}:
        raise ValueError("horizon must be 'h1' or 'h2'")
    output_dir, model_dir = Path(output_dir), Path(model_dir)
    drive_output_dir = Path(drive_output_dir) if drive_output_dir is not None else None
    drive_model_dir = Path(drive_model_dir) if drive_model_dir is not None else None
    _ensure_test_not_closed(horizon, output_dir, model_dir, drive_output_dir, drive_model_dir)

    table, tuning, folds = _load_evaluation_inputs(horizon, config)
    development_mask, test_mask = _validate_evaluation_inputs(table, tuning, folds)
    result_dirs = tuple(path for path in (output_dir, drive_output_dir) if path is not None)
    model_dirs = tuple(path for path in (model_dir, drive_model_dir) if path is not None)
    for directory in (output_dir, model_dir, drive_output_dir, drive_model_dir):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    feature_cols = tuning["feature_cols"]
    models = refit_final_stack(
        table.loc[development_mask].copy(), folds, feature_cols, config["target_col"],
        tuning["random_forest"]["best_params"], tuning["xgboost"]["best_params"],
        ridge_alpha=tuning["ridge_meta"]["alpha"],
    )
    models["horizon"] = horizon
    _save_model_bundle(horizon, models, model_dirs)
    closure = _mark_test_closed(horizon, feature_cols, result_dirs)

    predictions = predict_test_set(models, table.loc[test_mask].copy(), feature_cols)
    results_table = build_results_table(predictions, "market", "commodity")
    if set(results_table["model"]) != set(MODEL_PREDICTION_COLUMNS):
        raise ValueError("Final results must contain all five models, including persistence")
    results_table.insert(0, "horizon", horizon)
    manifest = _build_evaluation_manifest(
        closure, tuning, folds, int(development_mask.sum()), len(predictions),
    )
    _save_evaluation_outputs(horizon, results_table, predictions, manifest, result_dirs)
    _save_model_bundle(horizon, models, model_dirs)
    return results_table
