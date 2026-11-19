import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    month: int = Field(ge=1, le=12)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)
    weather_code: int = Field(ge=1, le=4)
    temperature_c: float = Field(ge=0, le=41)
    humidity_pct: float = Field(ge=0, le=100)
    wind_kmh: float = Field(ge=0, le=67)


def to_features(data):
    return pd.DataFrame([{
        "mnth": data.month,
        "hr": data.hour,
        "weekday": data.weekday,
        "workingday": data.workingday,
        "weathersit": data.weather_code,
        "temp": data.temperature_c / 41,
        "hum": data.humidity_pct / 100,
        "windspeed": data.wind_kmh / 67,
    }])


@asynccontextmanager
async def lifespan(app):
    app.state.model = None
    app.state.model_version = None

    # ci test model only - production will use mlflow
    if os.getenv("MODEL_SOURCE") == "ci":
        path = os.environ["CI_MODEL_PATH"]
        model = joblib.load(path)

        sample = PredictionInput(
            month=6,
            hour=8,
            weekday=1,
            workingday=1,
            weather_code=1,
            temperature_c=22,
            humidity_pct=50,
            wind_kmh=10,
        )

        model.predict(to_features(sample))
        app.state.model = model
        app.state.model_version = "ci-test-1"

    yield


app = FastAPI(title="BikeFlow", lifespan=lifespan)
app.state.model = None
app.state.model_version = None


@app.get("/")
def home():
    return {"message": "Welcome to BikeFlow"}


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    if app.state.model is None:
        raise HTTPException(503, "Approved model not loaded")

    return {
        "status": "ready",
        "model_version": app.state.model_version,
    }


@app.get("/model-info")
def model_info():
    if app.state.model is None:
        raise HTTPException(503, "Approved model not loaded")

    return {"model_version": app.state.model_version}


@app.post("/predict")
def predict(data: PredictionInput):
    if app.state.model is None:
        raise HTTPException(503, "Approved model not loaded")

    result = app.state.model.predict(to_features(data))

    return {
        "predicted_rentals": max(0, round(float(result[0]))),
        "model_version": app.state.model_version,
    }
