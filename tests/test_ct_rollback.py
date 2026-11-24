import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient
from sklearn.dummy import DummyRegressor

from ml.ct import plan
from ml.rollback import RollbackError, rollback

MODEL = "bikeflow-rollback-test"


def champion(dataset_version="v1", test_mae=50.0, reference_mae=50.0):
    return {"version": "1", "dataset_version": dataset_version, "test_mae": test_mae, "reference_mae": reference_mae}


# ---------------------------------------------------------------- CT decision

@pytest.mark.parametrize(
    "data_version, current, force, retrain, reason",
    [
        ("v1", None, False, True, "no champion"),                      # first release
        ("v2", champion("v1"), False, True, "new dataset version"),    # new data delivery
        ("v2", champion("v2"), False, False, "no performance drop"),   # nothing new: keep
        ("v2", champion("v2", 54.9), False, False, "no performance drop"),  # drop under 10%
        ("v2", champion("v2", 55.1), False, True, "degraded"),         # drop over 10%
        ("v2", champion("v2"), True, True, "forced"),                  # operator override
        ("v2", champion("v2", reference_mae=None), False, False, "no performance drop"),
    ],
)
def test_ct_retraining_decision(data_version, current, force, retrain, reason):
    decided, why = plan(data_version, current, force)

    assert decided is retrain
    assert reason in why


# ---------------------------------------------------------------- rollback

@pytest.fixture(scope="module")
def shared_registry(tmp_path_factory):
    """Versions 1 and 2 approved, 3 not approved. Built once for the module."""
    root = tmp_path_factory.mktemp("rollback")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MLFLOW_TRACKING_URI", uri)
        patch.setenv("MLFLOW_REGISTRY_URI", uri)
        client = MlflowClient()
        experiment = client.create_experiment("rollback-tests", artifact_location=(root / "artifacts").as_uri())

        model = DummyRegressor().fit(pd.DataFrame({"x": [0, 1]}), [0, 1])
        for approved in [True, True, False]:
            with mlflow.start_run(experiment_id=experiment):
                info = mlflow.sklearn.log_model(model, name="model")
            version = mlflow.register_model(info.model_uri, MODEL).version
            client.set_model_version_tag(MODEL, version, "dataset_version", f"v{version}")
            if approved:
                client.set_model_version_tag(MODEL, version, "approved", "true")
        yield client


@pytest.fixture
def registry(shared_registry):
    """Each test starts with the champion on version 2."""
    shared_registry.set_registered_model_alias(MODEL, "champion", "2")
    return shared_registry


def champion_version(client):
    return str(client.get_model_version_by_alias(MODEL, "champion").version)


def test_rollback_moves_the_alias_to_an_approved_version(registry):
    result = rollback("1", MODEL, registry)

    assert result == {
        "model_name": MODEL,
        "previous_version": "2",
        "new_version": "1",
        "dataset_version": "v1",
        "changed": True,
    }
    assert champion_version(registry) == "1"


def test_rollback_to_the_current_champion_changes_nothing(registry):
    result = rollback(2, MODEL, registry)

    assert result["changed"] is False
    assert champion_version(registry) == "2"


@pytest.mark.parametrize(
    "version, message",
    [
        ("3", "never approved"),
        ("99", "no version 99"),
        ("abc", "no version abc"),
    ],
)
def test_rollback_refuses_bad_versions(registry, version, message):
    with pytest.raises(RollbackError, match=message):
        rollback(version, MODEL, registry)

    assert champion_version(registry) == "2"


def test_rollback_without_a_current_champion(registry):
    registry.delete_registered_model_alias(MODEL, "champion")

    result = rollback("1", MODEL, registry)

    assert result["previous_version"] is None
    assert champion_version(registry) == "1"
