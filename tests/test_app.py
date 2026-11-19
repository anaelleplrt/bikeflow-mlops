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
