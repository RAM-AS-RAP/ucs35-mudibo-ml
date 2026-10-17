"""Tests for exact SHAP decomposition of the linear Ridge stack."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from agrisence.explainability.shap_explainer import (
    BASELINE_GROUPS,
    DRIVER_GROUPS,
    compute_stack_shap_values,
    shap_to_farmer_explanation,
)
from agrisence.training.folds import build_expanding_window_folds


class _LinearTreeFixture:
    def __init__(self, base_value, contributions):
        self.base_value = base_value
        self.contributions = np.asarray(contributions, dtype=float)

    def predict(self, X):
        return self.base_value + X.to_numpy(dtype=float) @ self.contributions


class _TreeExplainerFixture:
    def __init__(self, model):
        self.model = model
        self.expected_value = model.base_value

    def shap_values(self, X):
        return X.to_numpy(dtype=float) * self.model.contributions


def _models_and_features():
    X = pd.DataFrame({
        "lag_return": [0.2, -0.1],
        "market_indicator": [1.0, 0.0],
        "month_sin": [0.5, -0.5],
    })
    rf_model = _LinearTreeFixture(0.1, [0.2, -0.1, 0.05])
    xgb_model = _LinearTreeFixture(-0.2, [-0.05, 0.3, 0.1])
    ridge = Ridge()
    ridge.coef_ = np.array([0.4, 0.7])
    ridge.intercept_ = 0.03
    return {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "ridge_meta": ridge,
        "feature_cols": list(X.columns),
    }, X


def test_combined_shap_is_exact_additive_decomposition(monkeypatch):
    models, X = _models_and_features()
    monkeypatch.setitem(
        __import__("sys").modules,
        "shap",
        SimpleNamespace(TreeExplainer=_TreeExplainerFixture),
    )

    result = compute_stack_shap_values(models, X)

    ridge = models["ridge_meta"]
    expected_values = (
        ridge.coef_[0] * models["rf_model"].contributions
        + ridge.coef_[1] * models["xgb_model"].contributions
    )
    expected_base = (
        ridge.intercept_
        + ridge.coef_[0] * models["rf_model"].base_value
        + ridge.coef_[1] * models["xgb_model"].base_value
    )
    expected_predictions = (
        ridge.intercept_
        + ridge.coef_[0] * models["rf_model"].predict(X)
        + ridge.coef_[1] * models["xgb_model"].predict(X)
    )
    np.testing.assert_allclose(result["shap_values"], X.to_numpy() * expected_values)
    assert result["base_value"] == pytest.approx(expected_base)
    np.testing.assert_allclose(result["base_value"] + result["shap_values"].sum(axis=1), expected_predictions)
    assert result["feature_names"] == list(X.columns)


def test_grouped_farmer_explanation_reconciles_and_sorts_contributions():
    feature_names = [
        "log_return_lag_1", "log_return_lag_2", "log_return_lag_3",
        "log_ratio_3m", "log_ratio_12m", "month_sin", "month_cos", "year",
        "market_Nairobi", "commodity_Beans",
    ]
    row_values = np.array([0.001, 0.002, -0.0005, 0.01, -0.004, 0.1, -0.02, 0.0005, 0.005, -0.001])
    result = {
        "shap_values": row_values.reshape(1, -1),
        "base_value": 0.005,
        "feature_names": feature_names,
    }

    explanation = shap_to_farmer_explanation(result, row_index=0, origin_price=100.0)

    assert explanation["origin_price"] == 100.0
    assert explanation["base_value"] == 0.005
    assert explanation["predicted_log_change"] == pytest.approx(0.098)
    assert explanation["predicted_price"] == pytest.approx(100 * np.exp(0.098), abs=0.01)
    assert explanation["expected_change_pct"] == pytest.approx((np.exp(0.098) - 1) * 100, abs=0.01)
    groups = explanation["groups"]
    assert [group["label"] for group in groups] == [
        "Time of year", "Price compared with its recent average", "Recent price momentum",
    ]
    assert groups[-1]["direction"] == "increases"
    assert explanation["typical_change"]["log_value"] == pytest.approx(0.0095)
    assert not set(BASELINE_GROUPS).intersection(group["label"] for group in groups)
    assert sum(group["log_contribution"] for group in groups) + explanation["typical_change"]["log_value"] == pytest.approx(
        explanation["predicted_log_change"], abs=1e-4,
    )


def test_grouping_rejects_unassigned_or_multiply_assigned_features(monkeypatch):
    from agrisence.explainability import shap_explainer

    shap_result = {
        "shap_values": np.zeros((1, 2)),
        "base_value": 0.0,
        "feature_names": ["year", "unexpected_feature"],
    }
    with pytest.raises(ValueError, match="unexpected_feature"):
        shap_to_farmer_explanation(shap_result, row_index=0, origin_price=10.0)

    monkeypatch.setattr(shap_explainer, "BASELINE_GROUPS", {
        **BASELINE_GROUPS,
        "Duplicate group": ["year"],
    })
    shap_result["feature_names"] = ["year"]
    shap_result["shap_values"] = np.zeros((1, 1))
    with pytest.raises(ValueError, match="year"):
        shap_to_farmer_explanation(shap_result, row_index=0, origin_price=10.0)


def test_stack_shap_requires_feature_order_and_ridge(monkeypatch):
    models, X = _models_and_features()
    monkeypatch.setitem(
        __import__("sys").modules,
        "shap",
        SimpleNamespace(TreeExplainer=_TreeExplainerFixture),
    )
    reordered = X[X.columns[::-1]]
    with pytest.raises(ValueError, match="feature_cols"):
        compute_stack_shap_values(models, reordered)

    nonlinear_models = {**models, "ridge_meta": object()}
    with pytest.raises(TypeError, match="Ridge"):
        compute_stack_shap_values(nonlinear_models, X)


@pytest.mark.parametrize("row_index", [-1, 1])
def test_farmer_explanation_rejects_invalid_row_index(row_index):
    with pytest.raises(IndexError):
        shap_to_farmer_explanation(
            {"shap_values": np.ones((1, 2)), "base_value": 0.0, "feature_names": ["a", "b"]},
            row_index=row_index,
            origin_price=10.0,
        )


def test_shap_notebook_processes_only_saved_test_rows_and_writes_artifacts(tmp_path, monkeypatch):
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/07_shap_explainability.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    shap_cell = next(
        "\n".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "shap_results = {}" in "\n".join(cell["source"])
    )
    feature_dir = tmp_path / "features"
    model_dir = tmp_path / "models"
    evaluation_dir = tmp_path / "evaluation"
    output_dir = tmp_path / "artifacts"
    drive_output_dir = tmp_path / "drive"
    for directory in (feature_dir, model_dir, evaluation_dir, output_dir, drive_output_dir):
        directory.mkdir()

    dates = pd.date_range("2010-01-01", "2019-10-01", freq="MS")
    month_numbers = dates.month.to_numpy()
    row_positions = np.arange(len(dates), dtype=float)
    features = pd.DataFrame({
        "month": dates,
        "market": ["Nairobi"] * len(dates),
        "commodity": ["Maize"] * len(dates),
        "forecast_horizon": "h1",
        "target_month": dates + pd.DateOffset(months=1),
        "target_h1": np.linspace(-0.02, 0.03, len(dates)),
        "avg_price_per_kg": np.linspace(40.0, 55.0, len(dates)),
    })
    feature_cols = [
        "log_return_lag_1", "log_return_lag_2", "log_return_lag_3",
        "log_ratio_3m", "log_ratio_12m", "month_sin", "month_cos", "year",
        "market_Nairobi", "market_Kisumu", "market_Mombasa", "market_Nakuru",
        "commodity_Maize", "commodity_Beans",
    ]
    for index, feature in enumerate(feature_cols[:5]):
        features[feature] = np.sin(row_positions / (index + 2))
    features["month_sin"] = np.sin(2 * np.pi * (month_numbers - 1) / 12)
    features["month_cos"] = np.cos(2 * np.pi * (month_numbers - 1) / 12)
    features["year"] = dates.year
    features["market_Nairobi"] = 1
    for feature in feature_cols[9:12]:
        features[feature] = 0
    features["commodity_Maize"] = 1
    features["commodity_Beans"] = 0
    folds = build_expanding_window_folds(features, dev_end="2018-10-01")
    fold_artifact = {
        "table_row_position": list(range(len(features))),
        "development_mask": folds["development_mask"].tolist(),
        "test_mask": folds["test_mask"].tolist(),
        "folds": folds["folds"],
    }
    for horizon in ("h1",):
        horizon_table = features.copy()
        horizon_table["forecast_horizon"] = horizon
        horizon_table[f"target_{horizon}"] = features["target_h1"]
        horizon_table.to_csv(feature_dir / f"model_df_{horizon}.csv", index=False)
        (feature_dir / f"folds_{horizon}.json").write_text(
            json.dumps(fold_artifact, default=str), encoding="utf-8",
        )
        X_test = horizon_table.loc[folds["test_mask"], feature_cols].reset_index(drop=True)
        rf_model = _LinearTreeFixture(0.1, np.linspace(0.2, -0.1, len(feature_cols)))
        xgb_model = _LinearTreeFixture(-0.2, np.linspace(-0.05, 0.3, len(feature_cols)))
        ridge = Ridge()
        ridge.coef_ = np.array([0.4, 0.7])
        ridge.intercept_ = 0.03
        models = {
            "rf_model": rf_model,
            "xgb_model": xgb_model,
            "ridge_meta": ridge,
            "linear_model": object(),
            "feature_cols": feature_cols,
            "target_col": f"target_{horizon}",
            "horizon": horizon,
            "test_evaluated": True,
        }
        joblib.dump(models, model_dir / f"stacking_{horizon}.pkl")
        (evaluation_dir / f"evaluation_manifest_{horizon}.json").write_text(json.dumps({
            "status": "completed; held-out test closed",
            "test_rows": int(folds["test_mask"].sum()),
        }), encoding="utf-8")
        test_table = horizon_table.loc[folds["test_mask"]].reset_index(drop=True)
        rf_predictions = rf_model.predict(X_test)
        xgb_predictions = xgb_model.predict(X_test)
        stack_predictions = ridge.intercept_ + ridge.coef_[0] * rf_predictions + ridge.coef_[1] * xgb_predictions
        test_table.assign(
            origin_price=test_table["avg_price_per_kg"], stacking_pred=stack_predictions,
        )[[
            "month", "market", "commodity", "origin_price", "stacking_pred",
        ]].to_csv(evaluation_dir / f"test_predictions_{horizon}.csv", index=False)

    def fake_summary_plot(*args, **kwargs):
        return None

    def fake_savefig(path, **kwargs):
        Path(path).write_bytes(b"synthetic plot")

    monkeypatch.setitem(
        sys.modules,
        "shap",
        SimpleNamespace(TreeExplainer=_TreeExplainerFixture, summary_plot=fake_summary_plot),
    )
    def legacy_farmer_call(shap_result, row_index, top_n=5):
        return []

    scope = {
        "Path": Path, "json": json, "np": np, "pd": pd, "joblib": joblib,
        "plt": SimpleNamespace(
            figure=lambda *args, **kwargs: None,
            title=lambda *args, **kwargs: None,
            tight_layout=lambda *args, **kwargs: None,
            savefig=fake_savefig,
            close=lambda *args, **kwargs: None,
        ),
        "shap": sys.modules["shap"], "display": lambda *args, **kwargs: None,
        "compute_stack_shap_values": compute_stack_shap_values,
        "shap_to_farmer_explanation": legacy_farmer_call,
        "horizon_paths": {
            horizon: {
                "table": feature_dir / f"model_df_{horizon}.csv",
                "folds": feature_dir / f"folds_{horizon}.json",
                "predictions": evaluation_dir / f"test_predictions_{horizon}.csv",
                "manifest": evaluation_dir / f"evaluation_manifest_{horizon}.json",
                "model": model_dir / f"stacking_{horizon}.pkl",
            }
            for horizon in ("h1",)
        },
        "shap_dir": output_dir,
        "drive_shap_dir": drive_output_dir,
    }
    exec(compile(shap_cell, str(notebook_path), "exec"), scope)

    grouped_cell = next(
        "\n".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "grouped_explanations = {}" in "\n".join(cell["source"])
    )
    scope["shap_to_farmer_explanation"] = shap_to_farmer_explanation
    exec(compile(grouped_cell, str(notebook_path), "exec"), scope)

    assert set(scope["shap_results"]) == {"h1"}
    for horizon in ("h1",):
        summary = pd.read_csv(output_dir / f"shap_summary_{horizon}.csv")
        values = pd.read_csv(output_dir / f"shap_values_{horizon}.csv")
        assert len(summary) == len(feature_cols)
        assert len(values) == int(folds["test_mask"].sum())
        assert summary["mean_abs_shap"].is_monotonic_decreasing
        assert (drive_output_dir / f"shap_beeswarm_{horizon}.png").exists()
        grouped_path = output_dir / f"shap_grouped_{horizon}.json"
        grouped_output = json.loads(grouped_path.read_text(encoding="utf-8"))
        assert len(grouped_output["explanations"]) == int(folds["test_mask"].sum())
        assert len(grouped_output["group_mean_abs_log_contribution"]) == len(DRIVER_GROUPS)
        assert (drive_output_dir / grouped_path.name).exists()