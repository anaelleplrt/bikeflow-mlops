import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("/", {"message": "Welcome to BikeFlow"}),
        ("/health", {"status": "alive"}),
    ],
)
def test_public_routes(path, expected):
    response = client.get(path)

    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.parametrize("path", ["/ready", "/ready?check=true"])
def test_not_ready_without_model(path):
    app.state.model = None

    response = client.get(path)

    assert response.status_code == 503


@pytest.mark.parametrize("path", ["/unknown", "/prediction"])
def test_unknown_routes(path):
    response = client.get(path)

    assert response.status_code == 404



class FakeModel:
    def predict(self, features):
        return [42.7]


def valid_payload():
    return {
        "month": 6,
        "hour": 8,
        "weekday": 1,
        "workingday": 1,
        "weather_code": 1,
        "temperature_c": 22,
        "humidity_pct": 50,
        "wind_kmh": 10,
    }


@pytest.mark.parametrize("hour", [0, 8, 23])
def test_prediction_with_model(hour):
    app.state.model = FakeModel()
    app.state.model_version = "test-1"

    try:
        payload = valid_payload()
        payload["hour"] = hour

        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert response.json() == {
            "predicted_rentals": 43,
            "model_version": "test-1",
        }
    finally:
        app.state.model = None
        app.state.model_version = None


@pytest.mark.parametrize("field,value", [
    ("hour", -1),
    ("month", 13),
    ("humidity_pct", 101),
    ("weather_code", 5),
])
def test_invalid_input(field, value):
    payload = valid_payload()
    payload[field] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

