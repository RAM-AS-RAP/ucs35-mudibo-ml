"""Inspect trusted saved model bundles without fitting or predicting.

Run in the same Colab session/environment used for training:
    !python scripts/inspect_models.py
For this repository's layout:
    !python scripts/inspect_models.py --artifact-dir results/artifacts
The script also runs locally, but environment versions describe the machine
running it, not necessarily the training machine. Only --export-xgb writes
model exports; the original pickle files are never re-saved. Load only trusted
files, since both pickle and joblib deserialization can execute code.

After loading, sklearn __getstate__ reports the installed sklearn version,
not reliable historical provenance. Original version evidence is reported
separately when available in captured InconsistentVersionWarning instances.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import pickle
import platform
import sys
import warnings


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("scikit-learn", "xgboost", "shap", "numpy", "pandas", "scipy", "joblib")
XGB_PARAMETERS = (
    "objective", "n_estimators", "max_depth", "learning_rate", "subsample",
    "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
    "tree_method", "device", "random_state", "n_jobs", "eval_metric",
)


def _error(exception):
    return {"type": type(exception).__name__, "message": str(exception)}


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return str(value)


def _class_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _read(operation):
    try:
        return _json_value(operation())
    except Exception as exception:
        return {"error": _error(exception)}


def _load(path):
    attempts = []
    for loader_name in ("joblib", "pickle"):
        loaded = None
        error = None
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            try:
                if loader_name == "joblib":
                    import joblib

                    loaded = joblib.load(path)
                else:
                    with path.open("rb") as model_file:
                        loaded = pickle.load(model_file)
            except Exception as exception:
                error = _error(exception)
        warning_records = []
        for warning in captured:
            record = {
                "category": warning.category.__name__,
                "message": str(warning.message),
                "filename": warning.filename,
                "lineno": warning.lineno,
            }
            for attribute in ("estimator_name", "original_sklearn_version", "current_sklearn_version"):
                if hasattr(warning.message, attribute):
                    record[attribute] = getattr(warning.message, attribute)
            warning_records.append(record)
        attempts.append({"loader": loader_name, "error": error, "warnings": warning_records})
        if error is None:
            return loaded, loader_name, attempts
    return None, None, attempts


def _feature_list(bundle):
    preferred_keys = ("feature_cols", "feature_columns", "feature_names", "features")
    candidates = []
    for key, value in bundle.items():
        if not isinstance(value, (list, tuple)) and not hasattr(value, "tolist"):
            continue
        converted = _json_value(value)
        if isinstance(converted, list) and converted and all(isinstance(name, str) for name in converted):
            candidates.append(str(key))
    key = next((key for key in preferred_keys if key in candidates), None)
    if key is None and len(candidates) == 1:
        key = candidates[0]
    return key, _json_value(bundle[key]) if key is not None else None, candidates


def _inspect(path, installed_sklearn, export_path=None):
    bundle, loader, attempts = _load(path)
    result = {"path": str(path.relative_to(REPO_ROOT)), "loader": loader, "load_attempts": attempts}
    if loader is None:
        result["status"] = "FAIL: both loaders failed"
        return result
    result["object_type"] = _class_name(bundle)
    result["status"] = "loaded"
    if not isinstance(bundle, dict):
        result["status"] = "FAIL: expected a dict bundle"
        return result
    result["keys"] = [str(key) for key in bundle]
    feature_key, features, candidates = _feature_list(bundle)
    result.update({
        "feature_list_key": feature_key,
        "feature_list_candidates": candidates,
        "feature_columns": features,
        "feature_count": len(features) if features is not None else None,
        "missing_model_keys": [key for key in ("rf_model", "xgb_model", "ridge_meta") if key not in bundle],
    })
    model_details = {}
    for key, attributes in (
        ("rf_model", ("n_estimators", "max_depth", "min_samples_leaf", "max_features", "n_features_in_", "feature_names_in_")),
        ("xgb_model", ("n_features_in_",)),
        ("ridge_meta", ("alpha", "coef_", "intercept_", "n_features_in_")),
    ):
        estimator = bundle.get(key)
        if estimator is None:
            model_details[key] = {"error": "Estimator is missing"}
            continue
        detail = {"class": _class_name(estimator)}
        for attribute in attributes:
            detail[attribute] = _read(lambda: getattr(estimator, attribute, None))
        if key == "xgb_model":
            detail["main_params"] = _read(lambda: {
                name: value for name, value in estimator.get_params().items() if name in XGB_PARAMETERS
            })
            detail["booster_feature_names"] = _read(lambda: estimator.get_booster().feature_names)
            if export_path is not None:
                detail["json_export"] = _read(lambda: _export_xgb(estimator, export_path))
        model_details[key] = detail
    result["models"] = model_details
    checks = {}
    for label, names in (
        ("feature_list_matches_rf", model_details["rf_model"].get("feature_names_in_")),
        ("feature_list_matches_xgb", model_details["xgb_model"].get("booster_feature_names")),
    ):
        if features is None:
            checks[label] = "FAIL: feature list unavailable"
        elif isinstance(names, dict):
            checks[label] = "FAIL: could not read model feature names"
        elif names is None:
            checks[label] = "SKIP: model feature names not present"
        else:
            checks[label] = "PASS" if features == names else "FAIL: names or order differ"
    checks["ridge_has_two_features"] = (
        "PASS" if model_details["ridge_meta"].get("n_features_in_") == 2 else "FAIL"
    )
    result["checks"] = checks
    result["year_present"] = "year" in features if features is not None else None
    sklearn_versions = {}
    for key, estimator in bundle.items():
        if not any(base.__module__.startswith("sklearn.") for base in type(estimator).__mro__):
            continue
        state_version = _read(lambda: estimator.__getstate__().get("_sklearn_version"))
        original_evidence = [
            warning for attempt in attempts for warning in attempt["warnings"]
            if warning.get("estimator_name") == type(estimator).__name__
            and "original_sklearn_version" in warning
        ]
        record = {
            "class": _class_name(estimator),
            "getstate_sklearn_version": state_version,
            "installed_sklearn_version": installed_sklearn,
            "load_warning_version_evidence": original_evidence,
            "provenance_note": "Post-load __getstate__ is not proof of the pickle creation version.",
        }
        if isinstance(state_version, str) and state_version != installed_sklearn:
            record["warning"] = "WARNING: __getstate__ sklearn version differs from installed sklearn"
        if any(item["original_sklearn_version"] != installed_sklearn for item in original_evidence):
            record["creation_version_warning"] = "WARNING: recorded original sklearn version differs from installed sklearn"
        sklearn_versions[str(key)] = record
    result["sklearn_estimator_versions"] = sklearn_versions
    return result


def _export_xgb(estimator, path):
    estimator.get_booster().save_model(str(path))
    return str(path.relative_to(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-dir", default="artifacts", help="Artifact directory relative to the repository root")
    parser.add_argument("--export-xgb", action="store_true", help="Export the booster to a new xgb_h1.json file")
    args = parser.parse_args()
    artifact_dir = (REPO_ROOT / args.artifact_dir).resolve()
    if not artifact_dir.is_relative_to(REPO_ROOT):
        parser.error("--artifact-dir must resolve inside the repository")
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    report = {
        "environment": {"python": sys.version, "platform": platform.platform(), "packages": versions},
        "models": {},
    }
    for horizon in ("h1",):
        path = artifact_dir / "models" / f"stacking_{horizon}.pkl"
        export_path = artifact_dir / "models" / f"xgb_{horizon}.json" if args.export_xgb else None
        if export_path is not None and export_path.exists():
            parser.error(f"Refusing to overwrite existing export: {export_path}")
        try:
            report["models"][horizon] = _inspect(path, versions["scikit-learn"], export_path)
        except Exception as exception:
            report["models"][horizon] = {"path": str(path.relative_to(REPO_ROOT)), "error": _error(exception)}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output = artifact_dir / "model_inspection.json"
    readable = json.dumps(_json_value(report), indent=2, allow_nan=False)
    print(readable)
    output.write_text(readable + "\n", encoding="utf-8")
    print(f"Report saved: {output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()