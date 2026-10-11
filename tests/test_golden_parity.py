"""Ensure portable model artifacts reproduce Colab golden outputs and SHAP additivity."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agrisence.explainability.shap_explainer import compute_stack_shap_values
from agrisence.inference.model_loader import CHECKSUM_PATH, MODEL_DIR, MODEL_PATHS, load_model


pytestmark = pytest.mark.models
REQUIRED_FILES = [
    *[path for horizon_paths in MODEL_PATHS.values() for path in horizon_paths.values()],
    MODEL_DIR / "golden_h1.csv",
    MODEL_DIR / "golden_h2.csv",
    CHECKSUM_PATH,
]
MISSING_FILES = [str(path) for path in REQUIRED_FILES if not Path(path).is_file()]
if MISSING_FILES:
    pytest.skip(f"Golden model parity requires portable bundles, golden CSVs, and checksum manifest: {MISSING_FILES}", allow_module_level=True)


def _assert_column_matches(actual, expected, label, atol):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    max_difference = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0,
        atol=atol,
        err_msg=f"{label} parity failed; max absolute difference={max_difference:.12g}",
    )


@pytest.mark.parametrize("horizon", ["h1", "h2"])
def test_portable_model_matches_golden_predictions_and_exact_shap(horizon):
    models = load_model(horizon)
    golden = pd.read_csv(MODEL_DIR / f"golden_{horizon}.csv")
    X = golden[models["feature_cols"]].astype("float64")

    rf_predictions = models["rf_model"].predict(X)
    xgb_predictions = models["xgb_model"].predict(X)
    stack_predictions = models["ridge_meta"].predict(np.column_stack([rf_predictions, xgb_predictions]))
    _assert_column_matches(rf_predictions, golden["rf_pred"], f"{horizon} RF", atol=1e-6)
    _assert_column_matches(xgb_predictions, golden["xgb_pred"], f"{horizon} XGBoost", atol=1e-6)
    _assert_column_matches(stack_predictions, golden["stack_pred"], f"{horizon} stack", atol=1e-6)

    shap_result = compute_stack_shap_values(models, X)
    shap_reconstruction = shap_result["base_value"] + shap_result["shap_values"].sum(axis=1)
    _assert_column_matches(shap_reconstruction, stack_predictions, f"{horizon} SHAP reconstruction", atol=1e-5)