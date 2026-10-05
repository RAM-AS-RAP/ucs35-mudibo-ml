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


def test_farmer_explanation_sorts_absolute_contributions():
    result = {
        "shap_values": np.array([[0.1, -0.8, 0.3, 0.0]]),
        "feature_names": ["season", "lag", "market", "commodity"],
    }

    explanation = shap_to_farmer_explanation(result, row_index=0, top_n=3)

    assert explanation == [
        {"feature": "lag", "contribution": -0.8, "direction": "decreases"},
        {"feature": "market", "contribution": 0.3, "direction": "increases"},
        {"feature": "season", "contribution": 0.1, "direction": "increases"},
    ]
    assert shap_to_farmer_explanation(result, row_index=0, top_n=0) == []


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
            {"shap_values": np.ones((1, 2)), "feature_names": ["a", "b"]},
            row_index=row_index,
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
    features = pd.DataFrame({
        "month": dates,
        "market": ["Nairobi"] * len(dates),
        "commodity": ["Maize"] * len(dates),
        "forecast_horizon": "h1",
        "target_month": dates + pd.DateOffset(months=1),
        "target_h1": np.linspace(-0.02, 0.03, len(dates)),
        "avg_price_per_kg": np.linspace(40.0, 55.0, len(dates)),
        "feature_a": np.linspace(-1.0, 1.0, len(dates)),
        "feature_b": np.cos(np.arange(len(dates))),
    })
    folds = build_expanding_window_folds(features, dev_end="2018-10-01")
    fold_artifact = {
        "table_row_position": list(range(len(features))),
        "development_mask": folds["development_mask"].tolist(),
        "test_mask": folds["test_mask"].tolist(),
        "folds": folds["folds"],
    }
    feature_cols = ["feature_a", "feature_b"]

    for horizon in ("h1", "h2"):
        horizon_table = features.copy()
        horizon_table["forecast_horizon"] = horizon
        horizon_table[f"target_{horizon}"] = features["target_h1"]
        horizon_table.to_csv(feature_dir / f"model_df_{horizon}.csv", index=False)
        (feature_dir / f"folds_{horizon}.json").write_text(
            json.dumps(fold_artifact, default=str), encoding="utf-8",
        )
        X_test = horizon_table.loc[folds["test_mask"], feature_cols].reset_index(drop=True)
        rf_model = _LinearTreeFixture(0.1, [0.2, -0.1])
        xgb_model = _LinearTreeFixture(-0.2, [-0.05, 0.3])
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
        test_table.assign(stacking_pred=stack_predictions)[
            ["month", "market", "commodity", "stacking_pred"]
        ].to_csv(evaluation_dir / f"test_predictions_{horizon}.csv", index=False)

    def fake_summary_plot(*args, **kwargs):
        return None

    def fake_savefig(path, **kwargs):
        Path(path).write_bytes(b"synthetic plot")

    monkeypatch.setitem(
        sys.modules,
        "shap",
        SimpleNamespace(TreeExplainer=_TreeExplainerFixture, summary_plot=fake_summary_plot),
    )
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
        "shap_to_farmer_explanation": shap_to_farmer_explanation,
        "horizon_paths": {
            horizon: {
                "table": feature_dir / f"model_df_{horizon}.csv",
                "folds": feature_dir / f"folds_{horizon}.json",
                "predictions": evaluation_dir / f"test_predictions_{horizon}.csv",
                "manifest": evaluation_dir / f"evaluation_manifest_{horizon}.json",
                "model": model_dir / f"stacking_{horizon}.pkl",
            }
            for horizon in ("h1", "h2")
        },
        "shap_dir": output_dir,
        "drive_shap_dir": drive_output_dir,
    }
    exec(compile(shap_cell, str(notebook_path), "exec"), scope)

    assert set(scope["shap_results"]) == {"h1", "h2"}
    for horizon in ("h1", "h2"):
        summary = pd.read_csv(output_dir / f"shap_summary_{horizon}.csv")
        values = pd.read_csv(output_dir / f"shap_values_{horizon}.csv")
        assert len(summary) == 2
        assert len(values) == int(folds["test_mask"].sum())
        assert summary["mean_abs_shap"].is_monotonic_decreasing
        assert (drive_output_dir / f"shap_beeswarm_{horizon}.png").exists()