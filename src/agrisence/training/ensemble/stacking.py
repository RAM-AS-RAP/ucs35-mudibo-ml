"""Stacking ensemble builder for the crop price forecasting model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def generate_oof_predictions(
    df: pd.DataFrame,
    folds: dict,
    feature_cols: list[str],
    target_col: str,
    rf_params: dict,
    xgb_params: dict,
) -> pd.DataFrame:
    """Generate RF/XGBoost predictions only on later development folds."""
    from agrisence.training.folds import folds_to_sklearn_cv

    fold_pairs = folds_to_sklearn_cv(folds, df)
    records: list[pd.DataFrame] = []
    for train_indices, validation_indices in fold_pairs:
        rf = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_params)
        xgb = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            eval_metric="rmse",
            **xgb_params,
        )
        X_train = df.iloc[train_indices][feature_cols]
        y_train = df.iloc[train_indices][target_col]
        x_validation = df.iloc[validation_indices][feature_cols]
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        records.append(pd.DataFrame({
            "row_index": df.index[validation_indices],
            "origin_month": df.iloc[validation_indices]["month"].to_numpy(),
            "rf_oof_pred": rf.predict(x_validation),
            "xgb_oof_pred": xgb.predict(x_validation),
            "target": df.iloc[validation_indices][target_col].to_numpy(),
        }))
    if not records:
        raise ValueError("No out-of-fold predictions were generated")
    return pd.concat(records, ignore_index=True).sort_values("origin_month").reset_index(drop=True)


def fit_meta_model(oof_df: pd.DataFrame) -> Ridge:
    """Tune Ridge on chronological OOF base predictions and fit the meta-model."""
    required = {"rf_oof_pred", "xgb_oof_pred", "target", "origin_month"}
    missing = required.difference(oof_df.columns)
    if missing:
        raise ValueError(f"OOF table is missing columns: {sorted(missing)}")
    ordered = oof_df.sort_values("origin_month").reset_index(drop=True)
    origin_months = pd.to_datetime(ordered["origin_month"], errors="coerce")
    if origin_months.isna().any():
        raise ValueError("OOF origin_month values must all be valid dates")
    unique_months = pd.Series(origin_months.unique()).sort_values().reset_index(drop=True)
    if len(unique_months) < 3:
        raise ValueError("At least three distinct OOF origin months are required for Ridge CV")

    month_splitter = TimeSeriesSplit(n_splits=min(4, len(unique_months) - 1))
    cv = []
    for train_month_positions, validation_month_positions in month_splitter.split(unique_months):
        train_month_set = set(unique_months.iloc[train_month_positions])
        validation_month_set = set(unique_months.iloc[validation_month_positions])
        train_indices = np.flatnonzero(origin_months.isin(train_month_set).to_numpy())
        validation_indices = np.flatnonzero(origin_months.isin(validation_month_set).to_numpy())
        if len(train_indices) and len(validation_indices):
            cv.append((train_indices, validation_indices))
    search = GridSearchCV(
        Ridge(),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=cv,
        scoring="neg_mean_squared_error",
        refit=True,
    )
    search.fit(ordered[["rf_oof_pred", "xgb_oof_pred"]], ordered["target"])
    model = search.best_estimator_
    model.best_alpha_ = search.best_params_["alpha"]
    model.best_score_ = float(search.best_score_)
    return model


def refit_final_stack(
    df: pd.DataFrame,
    folds: dict,
    feature_cols: list[str],
    target_col: str,
    rf_params: dict,
    xgb_params: dict,
    ridge_alpha: float | None = None,
) -> dict:
    """Refit frozen base learners and a Ridge combiner using development rows only."""
    from agrisence.training.baselines.linear_regression import fit_linear_regression

    if not folds.get("folds"):
        raise ValueError("Saved chronological development folds are required")
    development_end = max(pd.Timestamp(fold["validation_end"]) for fold in folds["folds"])
    origin_months = pd.to_datetime(df["month"], errors="raise")
    if df.empty or origin_months.isna().any() or (origin_months > development_end).any():
        raise ValueError("Final refit accepts only nonempty development-period rows")
    if target_col in feature_cols or not feature_cols or len(set(feature_cols)) != len(feature_cols):
        raise ValueError("Features must be unique and exclude the target")
    if not np.isfinite(df[[*feature_cols, target_col]].to_numpy(dtype=float)).all():
        raise ValueError("Development features and targets must be finite")

    oof_df = generate_oof_predictions(df, folds, feature_cols, target_col, rf_params, xgb_params)
    if ridge_alpha is None:
        ridge_meta = fit_meta_model(oof_df)
    else:
        if not np.isfinite(ridge_alpha) or ridge_alpha < 0:
            raise ValueError("Frozen Ridge alpha must be finite and nonnegative")
        ridge_meta = Ridge(alpha=ridge_alpha).fit(
            oof_df[["rf_oof_pred", "xgb_oof_pred"]], oof_df["target"]
        )
        ridge_meta.best_alpha_ = ridge_alpha

    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1, **rf_params)
    xgb_model = XGBRegressor(
        objective="reg:squarederror", random_state=42, n_jobs=-1,
        eval_metric="rmse", **xgb_params,
    )
    rf_model.fit(df[feature_cols], df[target_col])
    xgb_model.fit(df[feature_cols], df[target_col])
    linear_model = fit_linear_regression(df[feature_cols], df[target_col])
    return {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "ridge_meta": ridge_meta,
        "linear_model": linear_model,
        "feature_cols": list(feature_cols),
        "target_col": target_col,
    }


def predict_test_set(models: dict, test_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Predict fitted models in one held-out pass; forbid reuse of the bundle."""
    if models.get("test_evaluated", False):
        raise RuntimeError("This model bundle has already evaluated the held-out test set")
    if list(feature_cols) != models["feature_cols"]:
        raise ValueError("Test feature order must match the fitted model bundle")
    target_col = models["target_col"]
    required = {"month", "market", "commodity", "avg_price_per_kg", target_col, *feature_cols}
    missing = required.difference(test_df.columns)
    if missing:
        raise ValueError(f"Test table is missing columns: {sorted(missing)}")
    origin_months = pd.to_datetime(test_df["month"], errors="raise")
    if test_df.empty or not origin_months.between("2018-11-01", "2019-10-01").all():
        raise ValueError("Test origins must be within November 2018-October 2019")
    if not np.isfinite(test_df[[*feature_cols, target_col, "avg_price_per_kg"]].to_numpy(dtype=float)).all():
        raise ValueError("Test features, targets, and prices must be finite")
    if (test_df["avg_price_per_kg"] <= 0).any():
        raise ValueError("Origin prices must be positive")

    models["test_evaluated"] = True
    predictions = test_df[["month", "market", "commodity"]].copy()
    predictions["target"] = test_df[target_col].to_numpy()
    predictions["origin_price"] = test_df["avg_price_per_kg"].to_numpy()
    predictions["true_price"] = predictions["origin_price"] * np.exp(predictions["target"])
    features = test_df[feature_cols]
    predictions["linear_regression_pred"] = models["linear_model"].predict(features)
    predictions["random_forest_pred"] = models["rf_model"].predict(features)
    predictions["xgboost_pred"] = models["xgb_model"].predict(features)
    meta_features = pd.DataFrame({
        "rf_oof_pred": predictions["random_forest_pred"],
        "xgb_oof_pred": predictions["xgboost_pred"],
    }, index=predictions.index)
    predictions["stacking_pred"] = models["ridge_meta"].predict(meta_features)
    return predictions


def build_stacking_ensemble(base_models: list[Any], meta_model: Any) -> Any:
    """Build a stacking ensemble with Random Forest and XGBoost base learners.

    The ensemble will combine tree-based base models and use a linear regression meta-learner
    to blend predictions for the final forecast.
    """
    if not base_models:
        raise ValueError("At least one base model is required")
    return {"base_models": base_models, "meta_model": meta_model}
