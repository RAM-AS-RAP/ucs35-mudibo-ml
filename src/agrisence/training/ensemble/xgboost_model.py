"""XGBoost ensemble trainer."""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor


XGB_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_weight": [1, 5],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.7, 1.0],
    "reg_lambda": [1, 5],
}


def tune_xgboost(
    df: Any,
    folds: dict,
    feature_cols: list[str],
    target_col: str,
) -> dict:
    """Tune XGBoost using only supplied chronological development folds."""
    from agrisence.training.folds import folds_to_sklearn_cv

    estimator = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
    )
    cv = folds_to_sklearn_cv(folds, df)
    search = GridSearchCV(
        estimator,
        param_grid=XGB_PARAM_GRID,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=False,
    )
    search.fit(df[feature_cols], df[target_col])
    return {"best_params": search.best_params_, "best_score": float(search.best_score_)}


def train_xgboost(X_train: Any, y_train: Any, param_grid: dict | None = None) -> Any:
    """Train an XGBoost regressor with optional hyperparameter tuning.

    This function is intended to use GridSearchCV with k-fold cross-validation on the
    training data once the modeling strategy is finalized.
    """
    estimator = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
    )
    if param_grid is None:
        return estimator.fit(X_train, y_train)
    return GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=param_grid.pop("cv"),
        n_jobs=-1,
    ).fit(X_train, y_train)
