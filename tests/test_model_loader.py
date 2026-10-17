"""Checksum and runtime-version guards for portable stack model files."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from agrisence.inference import model_loader


FEATURES = model_loader.EXPECTED_FEATURE_COLS


class FakeRF:
    feature_names_in_ = FEATURES


class FakeRidge:
    n_features_in_ = 2


class FakeBooster:
    feature_names = FEATURES

    def num_features(self):
        return 14


class FakeXGBRegressor:
    def load_model(self, path):
        self.path = path

    def get_booster(self):
        return FakeBooster()


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configured_loader(tmp_path, monkeypatch, bundle=None):
    xgb_path = tmp_path / "xgb_h1.json"
    sklearn_path = tmp_path / "stack_h1_sklearn.joblib"
    xgb_path.write_bytes(b'{"synthetic":"xgb model"}')
    sklearn_path.write_bytes(b"synthetic sklearn model")
    checksum_path = tmp_path / "MODEL_CHECKSUMS.json"
    checksum_path.write_text(json.dumps({
        xgb_path.name: _digest(xgb_path),
        sklearn_path.name: _digest(sklearn_path),
    }), encoding="utf-8")
    monkeypatch.setattr(model_loader, "MODEL_PATHS", {
        "h1": {"xgb": xgb_path, "sklearn": sklearn_path},
    })
    monkeypatch.setattr(model_loader, "CHECKSUM_PATH", checksum_path)
    model_bundle = bundle or {
        "rf_model": FakeRF(),
        "ridge_meta": FakeRidge(),
        "feature_cols": list(FEATURES),
        "target_col": "target_h1",
        "horizon": "h1",
    }
    monkeypatch.setitem(sys.modules, "joblib", SimpleNamespace(load=lambda path: model_bundle))
    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=FakeXGBRegressor))
    monkeypatch.setattr(model_loader.metadata, "version", lambda name: {
        "scikit-learn": "1.6.1", "xgboost": "3.4.1",
    }[name])
    return xgb_path, sklearn_path, checksum_path, model_bundle


def test_wrong_checksum_is_refused_before_loading(monkeypatch, tmp_path):
    xgb_path, _, checksum_path, _ = _configured_loader(tmp_path, monkeypatch)
    checksum_path.write_text(json.dumps({xgb_path.name: "0" * 64}), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "joblib", SimpleNamespace(load=lambda unused: pytest.fail("Joblib must not load")))
    with pytest.raises(model_loader.ModelLoadError, match="SHA-256 mismatch"):
        model_loader.load_model("h1")


def test_missing_checksum_entry_is_refused(monkeypatch, tmp_path):
    _, _, checksum_path, _ = _configured_loader(tmp_path, monkeypatch)
    checksum_path.write_text("{}", encoding="utf-8")
    with pytest.raises(model_loader.ModelLoadError, match="(?i)checksum entry.*missing"):
        model_loader.load_model("h1")


def test_missing_key_is_rejected(monkeypatch, tmp_path):
    _configured_loader(tmp_path, monkeypatch, {"rf_model": FakeRF()})
    with pytest.raises(model_loader.ModelLoadError, match="missing required keys"):
        model_loader.load_model("h1")


@pytest.mark.parametrize("horizon", ["h2", "h3"])
def test_unknown_horizon_raises_value_error(horizon):
    with pytest.raises(ValueError, match="horizon"):
        model_loader.load_model(horizon)


def test_installed_sklearn_version_mismatch_is_refused(monkeypatch, tmp_path):
    _configured_loader(tmp_path, monkeypatch)
    monkeypatch.setattr(model_loader.metadata, "version", lambda name: {
        "scikit-learn": "1.9.0", "xgboost": "3.4.1",
    }[name])
    with pytest.raises(model_loader.ModelLoadError, match="scikit-learn=1.9.0.*xgboost=3.4.1"):
        model_loader.load_model("h1")


def test_xgboost_version_mismatch_warns_but_loads(monkeypatch, tmp_path, capsys):
    _configured_loader(tmp_path, monkeypatch)
    monkeypatch.setattr(model_loader.metadata, "version", lambda name: {
        "scikit-learn": "1.6.1", "xgboost": "3.3.0",
    }[name])
    models = model_loader.load_model("h1")
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "xgboost=3.3.0" in output
    assert set(models) == {"rf_model", "xgb_model", "ridge_meta", "feature_cols", "horizon"}


def test_xgboost_feature_names_must_match_saved_order(monkeypatch, tmp_path):
    _configured_loader(tmp_path, monkeypatch)

    class MisorderedBooster(FakeBooster):
        feature_names = list(reversed(FEATURES))

    class MisorderedXGB(FakeXGBRegressor):
        def get_booster(self):
            return MisorderedBooster()

    monkeypatch.setitem(sys.modules, "xgboost", SimpleNamespace(XGBRegressor=MisorderedXGB))
    with pytest.raises(model_loader.ModelLoadError, match="XGBoost booster feature names"):
        model_loader.load_model("h1")