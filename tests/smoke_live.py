"""Smoke test against a running BikeFlow service backed by the MLflow registry.

Used by the staging workflow (container started by Actions) and the
production workflow (live Azure URL). It checks that:

1. the service becomes ready;
2. it serves the version that holds the champion alias in the registry
   (or --expected-version when given);
3. a real prediction returns a plausible number with that version;
4. the day prediction and the web page work.

Usage:

    python tests/smoke_live.py --url https://bikeflow.example.io
    python tests/smoke_live.py --url http://127.0.0.1:8000 --expected-version 2

Without --expected-version, MLFLOW_TRACKING_URI and the DagsHub credentials
must be set so the script can read the champion alias.
"""

import argparse
import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SAMPLE = {
    "year": 2012,
    "month": 6,
    "hour": 8,
    "weekday": 1,
    "workingday": 1,
    "weather_code": 1,
    "temperature_c": 24,
    "humidity_pct": 55,
    "wind_kmh": 10,
}
# Hourly rentals never exceeded 977 in the dataset; leave some margin.
MAX_PLAUSIBLE = 2000


def call(url, payload=None, timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    request = Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    with urlopen(request, timeout=timeout) as response:
        body = response.read().decode()
        content_type = response.headers.get("Content-Type", "")
        return json.loads(body) if "json" in content_type else body


def wait_until_ready(base, timeout):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            return call(f"{base}/ready")
        except HTTPError as error:
            last = f"HTTP {error.code}: {error.read().decode()[:200]}"
        except (URLError, OSError) as error:
            last = str(error)
        time.sleep(5)
    raise SystemExit(f"service not ready after {timeout}s (last answer: {last})")


def champion_version(model_name, alias):
    from mlflow import MlflowClient

    return str(MlflowClient().get_model_version_by_alias(model_name, alias).version)


def check(condition, message):
    if not condition:
        raise SystemExit(f"FAILED: {message}")
    print(f"ok  {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected-version")
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "bikeflow"))
    parser.add_argument("--alias", default=os.getenv("MODEL_ALIAS", "champion"))
    parser.add_argument("--timeout", type=int, default=300, help="seconds to wait for /ready")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    expected = args.expected_version or champion_version(args.model_name, args.alias)
    print(f"expecting {args.model_name} version {expected} at {base}")

    ready = wait_until_ready(base, args.timeout)
    check(ready.get("model_version") == expected, f"/ready reports version {ready.get('model_version')}")

    info = call(f"{base}/model-info")
    check(info.get("model_version") == expected, f"/model-info reports version {info.get('model_version')}")
    check(info.get("alias") == args.alias, f"/model-info reports alias {info.get('alias')}")

    prediction = call(f"{base}/predict", SAMPLE)
    rentals = prediction.get("predicted_rentals")
    check(isinstance(rentals, int) and 0 <= rentals <= MAX_PLAUSIBLE, f"/predict returns {rentals} rentals")
    check(prediction.get("model_version") == expected, "/predict answers with the expected version")

    day_input = {key: value for key, value in SAMPLE.items() if key != "hour"}
    day = call(f"{base}/predict/day", day_input)
    check(len(day.get("hours", [])) == 24, "/predict/day returns 24 hours")
    check(day["hours"][8]["predicted_rentals"] == rentals, "/predict/day agrees with /predict at 8 h")

    check("BikeFlow" in call(f"{base}/"), "web page is served")
    print("live smoke test passed")


if __name__ == "__main__":
    try:
        main()
    except HTTPError as error:
        sys.exit(f"FAILED: HTTP {error.code} on {error.url}: {error.read().decode()[:300]}")
