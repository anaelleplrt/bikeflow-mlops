"""Model loading against a real, local MLflow registry (SQLite in a temp dir).

The registry holds two approved versions and one unapproved version of a test
model. Each version predicts a different constant, so a prediction tells which
version was actually loaded.
"""

import math

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.dummy import DummyRegressor

from app.model_loader import (
    FEATURES,
    WARMUP_ROW,
    LoadedModel,
    ModelLoadError,
    check_can_predict,
    load_from_env,
    load_registry_model,
)

MODEL = "bikeflow-test"
ROW = pd.DataFrame([WARMUP_ROW], columns=FEATURES)


def constant_model(value):
    model = DummyRegressor(strategy="constant", constant=value)
    model.fit(ROW, [value])
    return model


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """Versions 1 (100, approved), 2 (200, approved), 3 (300, not approved)."""
    root = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLFLOW_TRACKING_URI", uri)
        patch.setenv("MLFLOW_REGISTRY_URI", uri)
        client = MlflowClient()
        experiment = client.create_experiment("loader-tests", artifact_location=(root / "artifacts").as_uri())

        for value, approved in [(100, True), (200, True), (300, False)]:
            with mlflow.start_run(experiment_id=experiment):
                info = mlflow.sklearn.log_model(constant_model(value), name="model")
            version = mlflow.register_model(info.model_uri, MODEL).version
            if approved:
                client.set_model_version_tag(MODEL, version, "approved", "true")
            client.set_model_version_tag(MODEL, version, "dataset_version", f"v{version}")

        client.set_registered_model_alias(MODEL, "champion", "1")
        yield client


def predicted(loaded):
    return float(loaded.model.predict(ROW)[0])


def test_champion_is_loaded_even_when_newer_versions_exist(registry):
    loaded = load_registry_model(MODEL, "champion", registry)

    assert loaded.info["model_version"] == "1"
    assert loaded.info["alias"] == "champion"
    assert loaded.info["dataset_version"] == "v1"
    assert loaded.info["run_id"]
    assert predicted(loaded) == 100


@pytest.mark.parametrize("target, expected", [("2", 200), ("1", 100)])
def test_moving_the_alias_changes_the_loaded_version(registry, target, expected):
    """Promotion (1 -> 2) and rollback (2 -> 1) are both just an alias move."""
    registry.set_registered_model_alias(MODEL, "champion", target)
    try:
        loaded = load_registry_model(MODEL, "champion", registry)

        assert loaded.info["model_version"] == target
        assert predicted(loaded) == expected
    finally:
        registry.set_registered_model_alias(MODEL, "champion", "1")


def test_unapproved_version_is_refused(registry):
    registry.set_registered_model_alias(MODEL, "champion", "3")
    try:
        with pytest.raises(ModelLoadError, match="not approved"):
            load_registry_model(MODEL, "champion", registry)
    finally:
        registry.set_registered_model_alias(MODEL, "champion", "1")


@pytest.mark.parametrize("name, alias", [(MODEL, "missing-alias"), ("no-such-model", "champion")])
def test_missing_alias_or_model_is_refused(registry, name, alias):
    with pytest.raises(ModelLoadError, match="holds alias"):
        load_registry_model(name, alias, registry)


def test_load_from_env_uses_the_registry(registry, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", registry.tracking_uri)
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", registry.tracking_uri)

    loaded = load_from_env({"MODEL_SOURCE": "mlflow", "MODEL_NAME": MODEL, "MODEL_ALIAS": "champion"})

    assert loaded.info["source"] == "mlflow"
    assert loaded.info["model_version"] == "1"
    assert loaded.info["loaded_at"]


# ---------------------------------------------------------------- without registry

def test_ci_source_loads_a_local_file(tmp_path):
    path = tmp_path / "ci_model.joblib"
    joblib.dump(constant_model(123), path)

    loaded = load_from_env({"MODEL_SOURCE": "ci", "CI_MODEL_PATH": str(path)})

    assert loaded.info["model_version"] == "ci-test-1"
    assert predicted(loaded) == 123


@pytest.mark.parametrize(
    "env, message",
    [
        ({"MODEL_SOURCE": "s3"}, "unknown MODEL_SOURCE"),
        ({"MODEL_SOURCE": "ci", "CI_MODEL_PATH": "does/not/exist.joblib"}, "cannot read CI model"),
    ],
)
def test_bad_configuration_is_refused(env, message):
    with pytest.raises(ModelLoadError, match=message):
        load_from_env(env)


class Raises:
    def predict(self, features):
        raise ValueError("feature names mismatch")


class Returns:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return self.value


@pytest.mark.parametrize(
    "model, message",
    [
        (Raises(), "warm-up prediction"),
        (Returns([math.nan]), "non-finite"),
        (Returns([math.inf]), "non-finite"),
        (Returns([]), "warm-up prediction"),
        (Returns(["abc"]), "warm-up prediction"),
    ],
)
def test_model_that_cannot_predict_is_refused(model, message):
    with pytest.raises(ModelLoadError, match=message):
        check_can_predict(LoadedModel(model))


def test_model_that_predicts_passes_the_check():
    check_can_predict(LoadedModel(Returns(np.array([87.5]))))
