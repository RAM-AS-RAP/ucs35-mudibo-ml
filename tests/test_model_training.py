"""Focused tests for Task 2 training utilities."""

import numpy as np
import pandas as pd

from agrisence.training.baselines.linear_regression import fit_linear_regression
from agrisence.training.baselines.persistence import predict_persistence
from agrisence.training.ensemble.stacking import fit_meta_model, generate_oof_predictions
from agrisence.training.folds import build_expanding_window_folds, folds_to_sklearn_cv


def _table(horizon="h1"):
    origins = pd.date_range("2010-01-01", "2018-10-01", freq="MS")
    target = origins + pd.DateOffset(months=1 if horizon == "h1" else 2)
    return pd.DataFrame({
        "month": origins,
        "target_month": target,
        "forecast_horizon": horizon,
        "x1": np.arange(len(origins), dtype=float),
        "x2": np.sin(np.arange(len(origins))),
        "target": np.cos(np.arange(len(origins))) / 10,
    })


def test_persistence_and_linear_regression():
    table = _table()
    assert np.array_equal(predict_persistence(table), np.zeros(len(table)))
    model = fit_linear_regression(table[["x1", "x2"]], table["target"])
    assert len(model.predict(table[["x1", "x2"]])) == len(table)


def test_sklearn_folds_are_temporal_and_non_overlapping():
    table = _table()
    folds = build_expanding_window_folds(table)
    splits = folds_to_sklearn_cv(folds, table)
    for train_indices, validation_indices in splits:
        assert max(train_indices) < min(validation_indices)
        assert table.iloc[train_indices]["month"].max() < table.iloc[validation_indices]["month"].min()


def test_oof_predictions_are_validation_aligned_and_meta_model_fits():
    table = _table()
    folds = build_expanding_window_folds(table)
    oof = generate_oof_predictions(
        table,
        folds,
        ["x1", "x2"],
        "target",
        {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 2, "max_features": 1.0},
        {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1, "min_child_weight": 1,
         "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1},
    )
    assert {"rf_oof_pred", "xgb_oof_pred", "target"}.issubset(oof.columns)
    assert oof["origin_month"].is_monotonic_increasing
    meta = fit_meta_model(oof)
    assert hasattr(meta, "best_alpha_")
    assert len(meta.predict(oof[["rf_oof_pred", "xgb_oof_pred"]])) == len(oof)
