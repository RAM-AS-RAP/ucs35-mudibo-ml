"""Random forest ensemble trainer."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


RF_PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [5, 10],
    "max_features": ["sqrt", 0.7],
}


def tune_random_forest(
    df: Any,
    folds: dict,
    feature_cols: list[str],
    target_col: str,
) -> dict:
    """Tune Random Forest using only supplied chronological development folds."""
    from agrisence.training.folds import folds_to_sklearn_cv

    estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
    cv = folds_to_sklearn_cv(folds, df)
    search = GridSearchCV(
        estimator,
        param_grid=RF_PARAM_GRID,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=False,
    )
    search.fit(df[feature_cols], df[target_col])
    return {"best_params": search.best_params_, "best_score": float(search.best_score_)}


def train_random_forest(X_train: Any, y_train: Any, param_grid: dict | None = None) -> Any:
    """Train a random forest regressor with optional hyperparameter tuning.

    This function is intended to use GridSearchCV with k-fold cross-validation on the
    training data once the modeling strategy is finalized.
    """
    estimator = RandomForestRegressor(random_state=42, n_jobs=-1)
    if param_grid is None:
        return estimator.fit(X_train, y_train)
    return GridSearchCV(
        estimator,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=param_grid.pop("cv"),
        n_jobs=-1,
    ).fit(X_train, y_train)
