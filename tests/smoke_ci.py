
import json
import time
from urllib.request import Request, urlopen

base_url = "http://127.0.0.1:8000"

for _ in range(30):
    try:
        with urlopen(f"{base_url}/ready", timeout=2) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(2)
else:
    raise RuntimeError("Container did not become ready")


with urlopen(f"{base_url}/health") as response:
    assert json.load(response) == {"status": "alive"}


with urlopen(f"{base_url}/model-info") as response:
    assert json.load(response)["model_version"] == "ci-test-1"


payload = {
    "month": 6,
    "hour": 8,
    "weekday": 1,
    "workingday": 1,
    "weather_code": 1,
    "temperature_c": 22,
    "humidity_pct": 50,
    "wind_kmh": 10,
}

request = Request(
    f"{base_url}/predict",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urlopen(request) as response:
    assert json.load(response) == {
        "predicted_rentals": 123,
        "model_version": "ci-test-1",
    }

print("Docker smoke test passed")
