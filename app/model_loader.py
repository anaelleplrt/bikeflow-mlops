"""Load the model served by the API.

In production the service loads the MLflow model version that holds the
``champion`` alias. The alias is resolved to a concrete version number when
the model is (re)loaded, and that version is what the API reports.

``MODEL_SOURCE=ci`` loads a local joblib file instead. It is only used by the
CI smoke test, which has no access to the shared registry.
"""

import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

# Column order of the training features (see ml/data.py).
FEATURES = ["mnth", "hr", "weekday", "workingday", "weathersit", "temp", "hum", "windspeed", "yr"]

# One realistic row, used to check that a loaded model can really predict.
WARMUP_ROW = {
    "mnth": 6,
    "hr": 8,
    "weekday": 1,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.64,
    "hum": 0.5,
    "windspeed": 0.15,
    "yr": 1,
}


class ModelLoadError(RuntimeError):
    """The approved model could not be loaded or could not predict."""


@dataclass
class LoadedModel:
    model: object
    info: dict = field(default_factory=dict)


def _now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_ci_model(path):
    import joblib

    try:
        model = joblib.load(path)
    except (OSError, ValueError) as error:
        raise ModelLoadError(f"cannot read CI model {path}: {error}") from error
    return LoadedModel(model, {"source": "ci", "model_name": "ci-test", "model_version": "ci-test-1"})


def load_registry_model(model_name, alias, client=None):
    """Resolve ``alias`` to a version of ``model_name`` and load that version."""
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = client or MlflowClient()
    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException as error:
        raise ModelLoadError(f"no version of {model_name!r} holds alias {alias!r}") from error

    if version.tags.get("approved") != "true":
        raise ModelLoadError(
            f"{model_name} version {version.version} holds {alias!r} but is not approved"
        )

    # Load the concrete version, not the alias: if the alias moves while we
    # load, we still serve (and report) exactly the version we resolved.
    uri = f"models:/{model_name}/{version.version}"
    try:
        model = mlflow.sklearn.load_model(uri)
    except Exception as error:  # network, artifact or deserialisation error
        raise ModelLoadError(f"cannot load {uri}: {error}") from error

    return LoadedModel(model, {
        "source": "mlflow",
        "model_name": model_name,
        "alias": alias,
        "model_version": str(version.version),
        "run_id": version.run_id,
        "dataset_version": version.tags.get("dataset_version"),
    })


def check_can_predict(loaded):
    """Raise ModelLoadError unless the model returns one finite number."""
    try:
        result = loaded.model.predict(pd.DataFrame([WARMUP_ROW], columns=FEATURES))
        value = float(result[0])
    except Exception as error:
        raise ModelLoadError(f"model failed the warm-up prediction: {error}") from error
    if not math.isfinite(value):
        raise ModelLoadError(f"model returned a non-finite warm-up prediction: {value}")


def load_from_env(env=None):
    """Load the model described by the environment and check it can predict."""
    env = os.environ if env is None else env
    source = env.get("MODEL_SOURCE", "mlflow")

    if source == "ci":
        loaded = load_ci_model(env["CI_MODEL_PATH"])
    elif source == "mlflow":
        loaded = load_registry_model(
            env.get("MODEL_NAME", "bikeflow"), env.get("MODEL_ALIAS", "champion")
        )
    else:
        raise ModelLoadError(f"unknown MODEL_SOURCE {source!r} (expected 'mlflow' or 'ci')")

    check_can_predict(loaded)
    loaded.info["loaded_at"] = _now()
    return loaded
