import hmac
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.model_loader import FEATURES, ModelLoadError, load_from_env

logger = logging.getLogger("bikeflow")

STATIC_DIR = Path(__file__).parent / "static"

# The UCI hourly data normalises temperature with t_min = -8 and t_max = 39 °C,
# humidity by 100 and wind speed by 67 km/h.
TEMP_MIN_C, TEMP_MAX_C = -8, 39
HUMIDITY_MAX = 100
WIND_MAX_KMH = 67
BASE_YEAR = 2011


class Conditions(BaseModel):
    """Date and weather for one day, without the hour."""

    year: int = Field(ge=2011, le=2100)
    month: int = Field(ge=1, le=12)
    weekday: int = Field(ge=0, le=6, description="0 = Sunday ... 6 = Saturday")
    workingday: int = Field(ge=0, le=1, description="1 = neither weekend nor holiday")
    weather_code: int = Field(
        ge=1, le=4, description="1 clear, 2 mist or clouds, 3 light rain or snow, 4 heavy rain"
    )
    temperature_c: float = Field(ge=TEMP_MIN_C, le=TEMP_MAX_C)
    humidity_pct: float = Field(ge=0, le=HUMIDITY_MAX)
    wind_kmh: float = Field(ge=0, le=WIND_MAX_KMH)


class PredictionInput(Conditions):
    hour: int = Field(ge=0, le=23)


def feature_row(data, hour):
    return {
        "mnth": data.month,
        "hr": hour,
        "weekday": data.weekday,
        "workingday": data.workingday,
        "weathersit": data.weather_code,
        "temp": (data.temperature_c - TEMP_MIN_C) / (TEMP_MAX_C - TEMP_MIN_C),
        "hum": data.humidity_pct / HUMIDITY_MAX,
        "windspeed": data.wind_kmh / WIND_MAX_KMH,
        "yr": data.year - BASE_YEAR,
    }


def to_features(data, hours=None):
    hours = [data.hour] if hours is None else hours
    return pd.DataFrame([feature_row(data, hour) for hour in hours], columns=FEATURES)


def rentals(value):
    return max(0, round(float(value)))


# ---------------------------------------------------------------- model state

def set_loading_state(status, error=None):
    app.state.load_status = status
    app.state.load_error = error


def load_model_with_retries():
    """Background start-up load: the API answers /health meanwhile, /ready says 503."""
    attempts = int(os.getenv("MODEL_LOAD_ATTEMPTS", "3"))
    delay = float(os.getenv("MODEL_LOAD_RETRY_SECONDS", "5"))

    for attempt in range(1, attempts + 1):
        try:
            loaded = load_from_env()
        except (ModelLoadError, KeyError) as error:
            logger.error("model load attempt %d/%d failed: %s", attempt, attempts, error)
            set_loading_state("loading" if attempt < attempts else "failed", str(error))
            if attempt < attempts:
                time.sleep(delay)
            continue

        app.state.loaded = loaded
        set_loading_state("ready")
        logger.info("serving %s", loaded.info)
        return


@asynccontextmanager
async def lifespan(app):
    app.state.loaded = None
    set_loading_state("loading")
    threading.Thread(target=load_model_with_retries, daemon=True).start()
    yield


app = FastAPI(title="BikeFlow", lifespan=lifespan)
app.state.loaded = None
app.state.load_status = "loading"
app.state.load_error = None


def current_model():
    loaded = app.state.loaded
    if loaded is None:
        raise HTTPException(503, "Approved model not loaded")
    return loaded


# ---------------------------------------------------------------- routes

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    loaded = app.state.loaded
    if loaded is None:
        detail = {"status": app.state.load_status, "error": app.state.load_error}
        raise HTTPException(503, detail)

    return {"status": "ready", "model_version": loaded.info["model_version"]}


@app.get("/model-info")
def model_info():
    return current_model().info


@app.post("/predict")
def predict(data: PredictionInput):
    loaded = current_model()
    result = loaded.model.predict(to_features(data))

    return {
        "predicted_rentals": rentals(result[0]),
        "model_version": loaded.info["model_version"],
    }


@app.post("/predict/day")
def predict_day(data: Conditions):
    """Predict the 24 hours of one day in a single model call."""
    loaded = current_model()
    hours = list(range(24))
    result = loaded.model.predict(to_features(data, hours))
    values = [rentals(value) for value in result]

    return {
        "hours": [{"hour": hour, "predicted_rentals": value} for hour, value in zip(hours, values)],
        "total_rentals": sum(values),
        "peak_hour": max(hours, key=lambda hour: values[hour]),
        "model_version": loaded.info["model_version"],
    }


@app.post("/reload")
def reload(authorization: str | None = Header(default=None)):
    """Resolve the alias again and swap the model in, without a redeploy.

    Called by the CT and rollback workflows after the champion alias moves.
    If the new model cannot be loaded, the previous one keeps serving.
    """
    token = os.getenv("RELOAD_TOKEN")
    if not token:
        raise HTTPException(404, "Reload is disabled (RELOAD_TOKEN not set)")
    if not hmac.compare_digest(authorization or "", f"Bearer {token}"):
        raise HTTPException(401, "Invalid reload token")

    previous = app.state.loaded
    try:
        loaded = load_from_env()
    except ModelLoadError as error:
        logger.error("reload failed, keeping the current model: %s", error)
        raise HTTPException(503, {
            "error": str(error),
            "still_serving": previous.info["model_version"] if previous else None,
        }) from error

    app.state.loaded = loaded
    set_loading_state("ready")
    return {
        "previous_version": previous.info["model_version"] if previous else None,
        **loaded.info,
    }
