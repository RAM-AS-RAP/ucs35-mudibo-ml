"""Pure feature preparation and exact stacked-forecast explanations."""

from datetime import date
from typing import Any
import warnings

import numpy as np
import pandas as pd

from agrisence.explainability.shap_explainer import compute_stack_shap_values, shap_to_farmer_explanation
from agrisence.preprocessing.feature_engineering import apply_calendar_distance_guard, build_features


class FeatureError(ValueError):
    """An origin row cannot be represented by the saved model features."""


def _first_of_month(value) -> pd.Timestamp:
    return pd.Timestamp(value).to_period("M").to_timestamp()


def prepare_series_panel(prices_df: pd.DataFrame, market: str, crop: str, upto_month: date) -> pd.DataFrame:
    """Create a complete, causally truncated panel for one market/crop series."""
    required = {"market", "crop", "month", "avg_price_per_kg"}
    missing = required.difference(prices_df.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")
    if market not in {"Nairobi", "Kisumu", "Mombasa", "Nakuru"}:
        raise ValueError(f"Unsupported market name: {market}")
    crop_name = {"maize": "Maize", "beans": "Beans"}.get(crop.lower())
    if crop_name is None:
        raise ValueError(f"Unsupported crop: {crop}")
    data = prices_df.loc[
        prices_df["market"].eq(market) & prices_df["crop"].astype(str).str.lower().eq(crop.lower()),
        ["market", "crop", "month", "avg_price_per_kg"],
    ].copy()
    data["month"] = pd.to_datetime(data["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if data["month"].isna().any():
        raise ValueError(f"Invalid month in stored prices for {market}/{crop}")
    limit = _first_of_month(upto_month)
    data = data.loc[data["month"] <= limit]
    if data.empty:
        raise ValueError(f"No stored price history through {limit.date()} for {market}/{crop}")
    data["avg_price_per_kg"] = pd.to_numeric(data["avg_price_per_kg"], errors="coerce")
    data = data.sort_values("month", kind="stable")
    if data["month"].duplicated().any():
        raise ValueError(f"Duplicate stored price months for {market}/{crop}")
    calendar = pd.date_range(data["month"].min(), limit, freq="MS", name="month")
    panel = data.set_index("month")["avg_price_per_kg"].reindex(calendar).rename("avg_price_per_kg").reset_index()
    panel["market"] = market
    panel["commodity"] = crop_name
    return panel[["market", "commodity", "month", "avg_price_per_kg"]]


def build_origin_features(panel: pd.DataFrame, origin_month: date, feature_cols: list[str]) -> pd.DataFrame:
    """Build the saved training features, including its calendar-distance guard."""
    origin = _first_of_month(origin_month)
    market = str(panel["market"].iloc[0]) if "market" in panel and not panel.empty else "unknown"
    crop = str(panel["commodity"].iloc[0]) if "commodity" in panel and not panel.empty else "unknown"
    try:
        featured = build_features(panel)
        guarded = apply_calendar_distance_guard(featured)
        origin_rows = guarded.loc[pd.to_datetime(guarded["month"]).eq(origin)]
    except Exception as exception:
        raise FeatureError(
            f"Could not build features for {market}/{crop} origin {origin.date()}: {exception}"
        ) from exception
    if len(origin_rows) != 1:
        raise FeatureError(f"Expected one row for {market}/{crop} origin {origin.date()}, found {len(origin_rows)}")
    row = origin_rows.iloc[[0]].copy()
    values = row.reindex(columns=feature_cols, fill_value=0)
    try:
        values = values.astype("float64")
    except (TypeError, ValueError) as exception:
        raise FeatureError(
            f"Non-numeric features for {market}/{crop} origin {origin.date()}: {feature_cols}"
        ) from exception
    array = values.to_numpy(dtype=np.float64)
    invalid_columns = [feature_cols[index] for index in range(len(feature_cols)) if not np.isfinite(array[0, index])]
    if invalid_columns:
        raise FeatureError(
            f"Missing or non-finite features for {market}/{crop} origin {origin.date()}: {invalid_columns}"
        )
    return values


def predict_and_explain(models: dict, X: pd.DataFrame, origin_prices: list[float]) -> list[dict[str, Any]]:
    """Batch-predict one horizon and reconcile using the saved exact stack SHAP."""
    if len(X) == 0 or len(origin_prices) != len(X):
        raise ValueError("X and origin_prices must have the same nonzero row count")
    origin_values = np.asarray(origin_prices, dtype=np.float64)
    if not np.isfinite(origin_values).all() or (origin_values <= 0).any():
        raise ValueError("Origin prices must be finite and positive")
    rf_predictions = np.asarray(models["rf_model"].predict(X), dtype=np.float64).reshape(-1)
    xgb_predictions = np.asarray(models["xgb_model"].predict(X), dtype=np.float64).reshape(-1)
    stack_predictions = np.asarray(
        models["ridge_meta"].predict(np.column_stack([rf_predictions, xgb_predictions])),
        dtype=np.float64,
    ).reshape(-1)
    if not np.isfinite(stack_predictions).all():
        raise ValueError("Stack predictions must be finite")
    shap_result = compute_stack_shap_values(models, X)
    results = []
    for row_index, origin_price in enumerate(origin_values):
        explanation = shap_to_farmer_explanation(
            shap_result,
            row_index,
            float(origin_price),
            model_prediction=float(stack_predictions[row_index]),
        )
        if explanation["origin_price"] != round(float(origin_price), 2):
            raise ValueError(f"Explanation origin price differs from stored origin price at row {row_index}")
        predicted_price = float(explanation["predicted_price"])
        predicted_log_change = float(explanation["predicted_log_change"])
        if predicted_price <= 0:
            raise ValueError(f"Predicted price must be positive at row {row_index}")
        if abs(predicted_log_change) > 0.7:
            raise ValueError(f"Predicted log change exceeds 0.7 at row {row_index}: {predicted_log_change}")
        if abs(predicted_log_change) > 0.3:
            warnings.warn(
                f"Large predicted log change at row {row_index}: {predicted_log_change}",
                RuntimeWarning,
                stacklevel=2,
            )
        results.append({
            "predicted_log_change": explanation["predicted_log_change"],
            "predicted_price": explanation["predicted_price"],
            "shap_values": explanation,
        })
    return results