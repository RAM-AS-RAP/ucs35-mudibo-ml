"""Focused synthetic tests for Task 2 training and Task 3 evaluation."""

import ast
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

from agrisence.training.baselines.linear_regression import fit_linear_regression
from agrisence.training.baselines.persistence import predict_persistence
from agrisence.training.ensemble.stacking import (
    fit_meta_model, generate_oof_predictions, predict_test_set, refit_final_stack,
)
from agrisence.training.folds import build_expanding_window_folds, folds_to_sklearn_cv
from agrisence.training.evaluate import (
    MODEL_PREDICTION_COLUMNS, build_results_table, compute_metrics, reconstruct_price,
)


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


@pytest.mark.parametrize("horizon", ["h1", "h2"])
def test_final_refit_and_single_test_pass(horizon):
    table = _table(horizon)
    folds = build_expanding_window_folds(table)
    features = ["x1", "x2"]
    rf_params = {"n_estimators": 5, "max_depth": 2}
    xgb_params = {"n_estimators": 5, "max_depth": 2}
    models = refit_final_stack(table, folds, features, "target", rf_params, xgb_params, ridge_alpha=1.0)
    assert models["ridge_meta"].alpha == 1.0
    assert models["rf_model"].n_estimators == 5
    test = table.iloc[-3:].copy()
    test["month"] = pd.date_range("2018-11-01", periods=3, freq="MS")
    test["market"] = "Nairobi"
    test["commodity"] = "Maize"
    test["avg_price_per_kg"] = 40.0
    with pytest.raises(ValueError, match="feature order"):
        predict_test_set(models, test, features[::-1])
    predictions = predict_test_set(models, test, features)
    assert len(predictions) == len(test)
    assert "persistence_pred" not in predictions.columns
    expected = models["ridge_meta"].predict(pd.DataFrame({
        "rf_oof_pred": predictions["random_forest_pred"],
        "xgb_oof_pred": predictions["xgboost_pred"],
    }))
    np.testing.assert_allclose(predictions["stacking_pred"], expected)
    np.testing.assert_allclose(predictions["true_price"], 40 * np.exp(test["target"]))
    with pytest.raises(RuntimeError, match="already evaluated"):
        predict_test_set(models, test, features)
    with pytest.raises(ValueError, match="development-period"):
        refit_final_stack(test, folds, features, "target", rf_params, xgb_params)


def test_final_metrics_reconstruction_and_error_metrics():
    origins = pd.Series([100.0, 100.0, 100.0], index=[4, 5, 6])
    actual = pd.Series([0.02, -0.02, 0.01])
    predicted = pd.Series([0.03, 0.02, -0.01])
    actual_prices = reconstruct_price(origins, actual)
    assert actual_prices.index.equals(origins.index)
    metrics = compute_metrics(actual, predicted, origins, actual_prices)
    errors = predicted.to_numpy() - actual.to_numpy()
    price_errors = 100 * (np.exp(predicted) - np.exp(actual))
    assert metrics["mae_log"] == pytest.approx(np.mean(np.abs(errors)))
    assert metrics["rmse_log"] == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert metrics["mae_price"] == pytest.approx(np.mean(np.abs(price_errors)))
    assert metrics["rmse_price"] == pytest.approx(np.sqrt(np.mean(price_errors ** 2)))
    assert metrics["mape_price_pct"] == pytest.approx(100 * np.mean(np.abs(price_errors) / actual_prices.to_numpy()))
    assert set(metrics) == {"mae_log", "rmse_log", "mape_price_pct", "mae_price", "rmse_price"}
    with pytest.raises(ValueError, match="disagree"):
        compute_metrics(actual, predicted, origins, [100, 100, 100])
    with pytest.raises(ValueError, match="positive"):
        reconstruct_price(pd.Series([0.0]), pd.Series([0.0]))


def test_results_table_contains_all_fitted_models_eight_series_and_pooled_metrics():
    predictions = pd.DataFrame([
        {"market": market, "commodity": commodity, "target": 0.02,
         "origin_price": 50.0, "true_price": 50 * np.exp(0.02)}
        for market in ("Nairobi", "Kisumu", "Mombasa", "Nakuru")
        for commodity in ("Beans", "Maize")
    ])
    for column in MODEL_PREDICTION_COLUMNS.values():
        predictions[column] = 0.02
    results = build_results_table(predictions, "market", "commodity")
    assert len(results) == 36
    assert "directional_accuracy" not in results.columns
    assert set(results["model"]) == set(MODEL_PREDICTION_COLUMNS)
    assert "Persistence" not in set(results["model"])
    pooled = results.loc[results["scope"].eq("pooled")]
    assert len(pooled) == 4
    assert (pooled["n_rows"] == 8).all()
    assert (pooled["mae_price"] == 0).all()


@pytest.mark.parametrize("horizon", ["h1", "h2"])
def test_final_notebook_saves_models_metrics_and_permanent_closure(tmp_path, horizon, monkeypatch):
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/06_final_evaluation.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cell_types = [cell["cell_type"] for cell in notebook["cells"]]
    assert len(cell_types) % 2 == 0
    assert all(cell_types[index:index + 2] == ["markdown", "code"] for index in range(0, len(cell_types), 2))
    defined_functions = []
    for cell in notebook["cells"]:
        assert cell["metadata"]["id"]
        assert cell["metadata"]["language"] == ("python" if cell["cell_type"] == "code" else "markdown")
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            defined_functions.extend(node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    assert defined_functions == []
    config_source = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "evaluation_outputs = " in "".join(cell["source"])
    )
    horizon_source = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code" and f"results_{horizon} = " in "".join(cell["source"])
    )
    import_source = next(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "except ImportError as error:" in "".join(cell["source"])
    )
    series_tables = []
    for market in ("Nairobi", "Kisumu", "Mombasa", "Nakuru"):
        for commodity in ("Beans", "Maize"):
            table = _table(horizon)
            future = table.iloc[-3:].copy()
            future["month"] = pd.date_range("2018-11-01", periods=3, freq="MS")
            future["target_month"] = future["month"] + pd.DateOffset(months=int(horizon[1:]))
            table = pd.concat([table, future], ignore_index=True)
            table["market"] = market
            table["commodity"] = commodity
            table["avg_price_per_kg"] = 50.0
            table[f"target_{horizon}"] = table.pop("target")
            series_tables.append(table)
    table = pd.concat(series_tables, ignore_index=True)
    folds = build_expanding_window_folds(table)
    saved_folds = {
        "table_row_position": list(range(len(table))),
        "development_mask": folds["development_mask"].tolist(),
        "test_mask": folds["test_mask"].tolist(),
        "folds": folds["folds"],
    }
    table.to_csv(tmp_path / "data.csv", index=False)
    (tmp_path / "folds.json").write_text(json.dumps(saved_folds, default=str), encoding="utf-8")
    payload = {
        "horizon": horizon, "feature_cols": ["x1", "x2"],
        "development_rows": int(folds["development_mask"].sum()),
        "test_rows_reserved_not_scored": int(folds["test_mask"].sum()),
        "random_forest": {"best_params": {"n_estimators": 3, "max_depth": 2}},
        "xgboost": {"best_params": {"n_estimators": 3, "max_depth": 2}},
        "ridge_meta": {"alpha": 1.0},
    }
    (tmp_path / "tuning.json").write_text(json.dumps(payload), encoding="utf-8")
    directories = {name: tmp_path / name for name in (
        "output_dir", "model_dir", "drive_output_dir", "drive_model_dir",
    )}
    for directory in directories.values():
        directory.mkdir()
    calls = []

    def tracked_predictions(models, test_df, feature_cols):
        assert (directories["output_dir"] / f"test_closed_{horizon}.json").exists()
        assert (directories["drive_output_dir"] / f"test_closed_{horizon}.json").exists()
        calls.append(horizon)
        return shared_predictions(models, test_df, feature_cols)

    scope = {
        **directories, "pd": pd,
        "display": lambda results: None,
        "horizon_configs": {horizon: {
            "data_path": tmp_path / "data.csv", "fold_path": tmp_path / "folds.json",
            "tuning_path": tmp_path / "tuning.json", "target_col": f"target_{horizon}",
        }},
    }
    exec(compile(import_source, str(notebook_path), "exec"), scope)
    shared_predictions = predict_test_set
    monkeypatch.setattr("agrisence.training.ensemble.stacking.predict_test_set", tracked_predictions)
    exec(compile(config_source, str(notebook_path), "exec"), scope)
    exec(compile(horizon_source, str(notebook_path), "exec"), scope)
    results = scope[f"results_{horizon}"]
    assert len(results) == 36
    assert "Persistence" not in set(results["model"])
    assert calls == [horizon]
    for directory in (directories["output_dir"], directories["drive_output_dir"]):
        pd.testing.assert_frame_equal(results, pd.read_csv(directory / f"results_{horizon}.csv"))
        predictions = pd.read_csv(directory / f"test_predictions_{horizon}.csv")
        assert len(predictions) == 24
        manifest = json.loads((directory / f"evaluation_manifest_{horizon}.json").read_text())
        assert manifest["ridge_alpha"] == 1.0
        assert manifest["test_rows"] == 24
        assert "directional_accuracy" not in manifest["metric_units"]
    for directory in (directories["model_dir"], directories["drive_model_dir"]):
        bundle = joblib.load(directory / f"stacking_{horizon}.pkl")
        assert {"rf_model", "xgb_model", "ridge_meta", "linear_model", "feature_cols"}.issubset(bundle)
        assert bundle["feature_cols"] == ["x1", "x2"]
        assert bundle["test_evaluated"] is True
    save_source = next("".join(cell["source"]) for cell in notebook["cells"]
                       if "saved_models = []" in "".join(cell["source"]))
    assert "run_final_evaluation" not in save_source
    assert "predict_test_set" not in save_source
    save_tree = ast.parse(save_source)
    save_loop = next(node for node in save_tree.body if isinstance(node, ast.For))
    save_loop.iter = ast.Tuple(elts=[ast.Constant(value=horizon)], ctx=ast.Load())
    ast.fix_missing_locations(save_tree)
    (directories["model_dir"] / f"stacking_{horizon}.pkl").unlink()
    exec(compile(save_tree, str(notebook_path), "exec"), scope)
    assert len(scope["saved_models"]) == 2
    assert calls == [horizon]
    restored = joblib.load(directories["model_dir"] / f"stacking_{horizon}.pkl")
    assert restored["test_evaluated"] is True
    assert restored["feature_cols"] == ["x1", "x2"]
    with pytest.raises(RuntimeError, match="closed"):
        exec(compile(horizon_source, str(notebook_path), "exec"), scope)
    for key in ("output_dir", "model_dir"):
        scope[key] = tmp_path / f"restart_{key}"
        scope[key].mkdir()
    exec(compile(config_source, str(notebook_path), "exec"), scope)
    with pytest.raises(RuntimeError, match="closed"):
        exec(compile(horizon_source, str(notebook_path), "exec"), scope)
    assert calls == [horizon]


def test_final_notebook_colab_setup_uses_training_drive_paths(tmp_path, monkeypatch):
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/06_final_evaluation.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup_tree = ast.parse("".join(notebook["cells"][1]["source"]))
    setup_tree.body = [
        node for node in setup_tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "pathlib")
    ]
    repo = tmp_path / "repo"
    (repo / "src/agrisence").mkdir(parents=True)
    (repo / "pyproject.toml").touch()
    drive_root = tmp_path / "drive"
    feature_dir = drive_root / "capstone-data/feature_engineering"
    tuning_dir = drive_root / "capstone-data/model_training"
    feature_dir.mkdir(parents=True)
    tuning_dir.mkdir(parents=True)
    for horizon in ("h1", "h2"):
        for filename in (f"model_df_{horizon}.csv", f"folds_{horizon}.json"):
            (feature_dir / filename).touch()
        (tuning_dir / f"tuning_{horizon}.json").touch()

    def mapped_path(value):
        value = str(value)
        if value == "/content/AgriSence_model_training":
            return repo
        prefix = "/content/drive/MyDrive/"
        if value.startswith(prefix):
            return drive_root / value[len(prefix):]
        if value == "/content/.env":
            return tmp_path / ".env"
        return Path(value)

    mounts = []
    installs = []
    google = ModuleType("google")
    colab = ModuleType("google.colab")
    colab.drive = SimpleNamespace(mount=mounts.append)
    google.colab = colab
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    ipython = ModuleType("IPython")
    ipython_display = ModuleType("IPython.display")
    ipython_display.display = lambda *args, **kwargs: None
    ipython.display = ipython_display
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)
    monkeypatch.setattr("subprocess.check_call", installs.append)
    monkeypatch.setattr("os.chdir", lambda directory: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    scope = {"Path": mapped_path}
    exec(compile(setup_tree, str(notebook_path), "exec"), scope)
    assert mounts == ["/content/drive"]
    assert len(installs) == 1
    assert scope["feature_dir"] == feature_dir
    assert scope["tuning_dir"] == tuning_dir
    assert scope["drive_output_dir"] == drive_root / "capstone-data/final_evaluation"
    assert scope["drive_model_dir"] == drive_root / "capstone-data/models"
    assert len(scope["ENV_PATHS"]) == 4
    (tuning_dir / "tuning_h2.json").unlink()
    with pytest.raises(FileNotFoundError, match="tuning_h2.json"):
        exec(compile(setup_tree, str(notebook_path), "exec"), {"Path": mapped_path})


def test_colab_import_updates_checkout_and_clears_stale_module(tmp_path, monkeypatch):
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/06_final_evaluation.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "except ImportError as error:" in "".join(cell["source"]))
    commands = []

    def fake_git(command, **kwargs):
        commands.append(command)
        if "fetch" in command:
            assert "GIT_CONFIG_VALUE_0" in kwargs["env"]
            assert not any("AUTHORIZATION" in argument for argument in command)
        if "branch" in command:
            return SimpleNamespace(stdout="feature/model-training\n")
        if "rev-parse" in command:
            return SimpleNamespace(stdout="abc1234\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setenv("GITHUB_TOKEN", "test-only-not-a-real-token")
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_git)
    saved_modules = {name: module for name, module in sys.modules.items()
                     if name == "agrisence" or name.startswith("agrisence.")}
    cached = ModuleType("agrisence.training.evaluate")
    monkeypatch.setitem(sys.modules, "agrisence.training.evaluate", cached)
    scope = {
        "IN_COLAB": True, "REPO_BRANCH": "feature/model-training", "repo_root": tmp_path,
        "os": os, "subprocess": subprocess, "base64": base64,
    }
    try:
        exec(compile(source, str(notebook_path), "exec"), scope)
        assert any(command[3:] == ["fetch", "origin", "feature/model-training"] for command in commands)
        assert any(command[3:] == ["merge", "--ff-only", "origin/feature/model-training"] for command in commands)
        assert sys.modules["agrisence.training.evaluate"] is not cached
        assert scope["run_final_evaluation"].__module__ == "agrisence.training.evaluate"
        assert "github_token" not in scope
        assert "git_env" not in scope
    finally:
        sys.modules.update(saved_modules)


@pytest.mark.parametrize("branch,changes", [("main", ""), ("feature/model-training", " M tracked.py")])
def test_colab_import_preserves_wrong_branch_or_tracked_edits(tmp_path, monkeypatch, branch, changes):
    notebook_path = Path(__file__).resolve().parents[1] / "notebooks/06_final_evaluation.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "except ImportError as error:" in "".join(cell["source"]))
    commands = []

    def fake_git(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=branch if "branch" in command else changes)

    monkeypatch.setattr(subprocess, "run", fake_git)
    scope = {"IN_COLAB": True, "REPO_BRANCH": "feature/model-training", "repo_root": tmp_path,
             "os": os, "subprocess": subprocess, "base64": base64}
    with pytest.raises(RuntimeError, match="expected|tracked edits"):
        exec(compile(source, str(notebook_path), "exec"), scope)
    assert not any("fetch" in command or "merge" in command for command in commands)
