import math

import pytest
from fastapi.testclient import TestClient

from app import main
from app.main import PredictionInput, app, to_features
from app.model_loader import FEATURES, LoadedModel, ModelLoadError

client = TestClient(app)


class FakeModel:
    """Returns 42.7 for every row."""

    def predict(self, features):
        return [42.7] * len(features)


class HourModel:
    """Returns 10 x hour, so the day curve is known in advance."""

    def predict(self, features):
        return [10.0 * hour for hour in features["hr"]]


def fake_loaded(model=None, version="test-1"):
    return LoadedModel(model or FakeModel(), {"model_name": "bikeflow", "model_version": version})


@pytest.fixture
def served():
    """Serve a fake model during one test, then go back to 'no model'."""
    def serve(model=None, version="test-1"):
        app.state.loaded = fake_loaded(model, version)

    yield serve
    app.state.loaded = None
    app.state.load_status = "loading"
    app.state.load_error = None


def valid_payload(**changes):
    payload = {
        "year": 2012,
        "month": 6,
        "hour": 8,
        "weekday": 1,
        "workingday": 1,
        "weather_code": 1,
        "temperature_c": 22,
        "humidity_pct": 50,
        "wind_kmh": 10,
    }
    payload.update(changes)
    return payload


def day_payload(**changes):
    payload = valid_payload(**changes)
    payload.pop("hour")
    return payload


# ---------------------------------------------------------------- routes

def test_home_serves_the_web_page():
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "BikeFlow" in response.text
    assert "predict/day" in response.text


@pytest.mark.parametrize("path", ["/unknown", "/prediction", "/static/index.html"])
def test_unknown_routes(path):
    assert client.get(path).status_code == 404


def test_health_is_alive_even_without_model():
    assert client.get("/health").json() == {"status": "alive"}


@pytest.mark.parametrize("status, error", [("loading", None), ("failed", "no version holds alias")])
def test_not_ready_without_model(status, error):
    app.state.loaded = None
    app.state.load_status, app.state.load_error = status, error

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == {"status": status, "error": error}


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("get", "/model-info", None),
        ("post", "/predict", valid_payload()),
        ("post", "/predict/day", day_payload()),
    ],
)
def test_model_routes_refuse_without_model(method, path, body):
    app.state.loaded = None

    response = getattr(client, method)(path, json=body) if body else client.get(path)

    assert response.status_code == 503


def test_ready_and_model_info_report_the_loaded_version(served):
    served(version="7")

    assert client.get("/ready").json() == {"status": "ready", "model_version": "7"}
    assert client.get("/model-info").json() == {"model_name": "bikeflow", "model_version": "7"}


# ---------------------------------------------------------------- predictions

@pytest.mark.parametrize("hour", [0, 8, 23])
def test_prediction_with_model(served, hour):
    served()

    response = client.post("/predict", json=valid_payload(hour=hour))

    assert response.status_code == 200
    assert response.json() == {"predicted_rentals": 43, "model_version": "test-1"}


@pytest.mark.parametrize("raw, expected", [(-3.2, 0), (0.4, 0), (99.5, 100), (1234.49, 1234)])
def test_prediction_is_a_non_negative_whole_number(served, raw, expected):
    class Constant:
        def predict(self, features):
            return [raw] * len(features)

    served(Constant())

    assert client.post("/predict", json=valid_payload()).json()["predicted_rentals"] == expected


def test_day_prediction_returns_24_ordered_hours(served):
    served(HourModel(), version="2")

    response = client.post("/predict/day", json=day_payload())
    body = response.json()

    assert response.status_code == 200
    assert [row["hour"] for row in body["hours"]] == list(range(24))
    assert [row["predicted_rentals"] for row in body["hours"]] == [10 * h for h in range(24)]
    assert body["total_rentals"] == 10 * sum(range(24))
    assert body["peak_hour"] == 23
    assert body["model_version"] == "2"


def test_day_prediction_ignores_an_hour_field(served):
    served(HourModel())

    response = client.post("/predict/day", json=day_payload(hour=99))

    assert response.status_code == 200
    assert len(response.json()["hours"]) == 24


@pytest.mark.parametrize("field, value", [
    ("hour", -1),
    ("hour", 24),
    ("month", 13),
    ("weekday", 7),
    ("workingday", 2),
    ("humidity_pct", 101),
    ("weather_code", 5),
    ("weather_code", 0),
    ("year", 2010),
    ("temperature_c", 40),
    ("temperature_c", -9),
    ("wind_kmh", 68),
    ("hour", "eight"),
    ("month", None),
])
def test_invalid_input(served, field, value):
    served()

    response = client.post("/predict", json=valid_payload(**{field: value}))

    assert response.status_code == 422


@pytest.mark.parametrize("missing", ["year", "hour", "temperature_c"])
def test_missing_field_is_rejected(served, missing):
    served()
    payload = valid_payload()
    payload.pop(missing)

    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "changes, expected",
    [
        ({"temperature_c": -8}, {"temp": 0.0}),
        ({"temperature_c": 39}, {"temp": 1.0}),
        ({"humidity_pct": 50, "wind_kmh": 67}, {"hum": 0.5, "windspeed": 1.0}),
        ({"year": 2011}, {"yr": 0}),
        ({"year": 2012, "hour": 17, "weather_code": 3}, {"yr": 1, "hr": 17, "weathersit": 3}),
    ],
)
def test_inputs_are_converted_to_training_scales(changes, expected):
    features = to_features(PredictionInput(**valid_payload(**changes)))

    assert list(features.columns) == FEATURES
    for column, value in expected.items():
        assert math.isclose(features.loc[0, column], value)


# ---------------------------------------------------------------- reload

def test_reload_is_disabled_without_token(monkeypatch):
    monkeypatch.delenv("RELOAD_TOKEN", raising=False)

    assert client.post("/reload").status_code == 404


@pytest.mark.parametrize("header", [None, "Bearer wrong", "secret", "Bearer SECRET"])
def test_reload_rejects_bad_tokens(monkeypatch, header):
    monkeypatch.setenv("RELOAD_TOKEN", "secret")
    headers = {"Authorization": header} if header else {}

    assert client.post("/reload", headers=headers).status_code == 401


def test_reload_swaps_in_the_new_champion(monkeypatch, served):
    served(version="1")
    monkeypatch.setenv("RELOAD_TOKEN", "secret")
    monkeypatch.setattr(main, "load_from_env", lambda: fake_loaded(version="2"))

    response = client.post("/reload", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.json()["previous_version"] == "1"
    assert response.json()["model_version"] == "2"
    assert client.get("/model-info").json()["model_version"] == "2"


def test_failed_reload_keeps_serving_the_previous_model(monkeypatch, served):
    served(version="1")
    monkeypatch.setenv("RELOAD_TOKEN", "secret")

    def broken():
        raise ModelLoadError("model failed the warm-up prediction")

    monkeypatch.setattr(main, "load_from_env", broken)

    response = client.post("/reload", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 503
    assert response.json()["detail"]["still_serving"] == "1"
    assert client.get("/ready").json() == {"status": "ready", "model_version": "1"}
    assert client.post("/predict", json=valid_payload()).json()["model_version"] == "1"


# ---------------------------------------------------------------- start-up

def test_startup_retries_then_becomes_ready(monkeypatch):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ModelLoadError("registry unreachable")
        return fake_loaded(version="2")

    monkeypatch.setattr(main, "load_from_env", flaky)
    monkeypatch.setenv("MODEL_LOAD_RETRY_SECONDS", "0")
    app.state.loaded = None
    try:
        main.load_model_with_retries()

        assert len(calls) == 3
        assert client.get("/ready").json() == {"status": "ready", "model_version": "2"}
    finally:
        app.state.loaded = None


def test_startup_gives_up_and_stays_not_ready(monkeypatch):
    def broken():
        raise ModelLoadError("no version holds alias")

    monkeypatch.setattr(main, "load_from_env", broken)
    monkeypatch.setenv("MODEL_LOAD_ATTEMPTS", "2")
    monkeypatch.setenv("MODEL_LOAD_RETRY_SECONDS", "0")
    app.state.loaded = None

    main.load_model_with_retries()

    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == {"status": "failed", "error": "no version holds alias"}
