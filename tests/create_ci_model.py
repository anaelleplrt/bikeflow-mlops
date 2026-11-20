
import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor

features = pd.DataFrame([{
    "mnth": 6,
    "hr": 8,
    "weekday": 1,
    "workingday": 1,
    "weathersit": 1,
    "temp": 22 / 41,
    "hum": 0.5,
    "windspeed": 10 / 67,
    "yr": 1,
}])

model = DummyRegressor(strategy="constant", constant=123)
model.fit(features, [123])

joblib.dump(model, "ci_model.joblib")
