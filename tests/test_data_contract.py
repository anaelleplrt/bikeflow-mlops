import shutil

import numpy as np
import pandas as pd
import pytest

from ml.data import (
    DATA_DIR,
    FEATURES,
    DataContractError,
    add_features,
    check_frame,
    check_separation,
    load_dataset,
)
from ml.train import build_model


def make_frame(start="2011-01-01", days=2):
    dates = pd.date_range(start, periods=days, freq="D")
    rows = []
    for day in dates:
        for hour in range(24):
            rows.append({
                "dteday": day.strftime("%Y-%m-%d"),
                "mnth": day.month,
                "hr": hour,
                "weekday": day.weekday(),
                "workingday": int(day.weekday() < 5),
                "weathersit": 1 + hour % 3,
                "temp": 0.5,
                "hum": 0.6,
                "windspeed": 0.1,
                "cnt": 10 + hour,
            })
    return pd.DataFrame(rows)


def test_valid_frame_passes():
    check_frame(make_frame(), "train")


@pytest.mark.parametrize("date, year", [("2011-03-01", 0), ("2012-12-31", 1), ("2014-06-15", 3)])
def test_year_feature_is_derived_from_date(date, year):
    frame = add_features(make_frame(date, days=1))

    assert (frame["yr"] == year).all()
    assert list(frame[FEATURES].columns) == FEATURES


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda df: df.drop(columns=["temp"]), "missing columns"),
        (lambda df: df.iloc[0:0], "no rows"),
        (lambda df: df.assign(hum=np.nan), "missing values"),
        (lambda df: df.assign(hr=24), "hr outside"),
        (lambda df: df.assign(weathersit=5), "weathersit outside"),
        (lambda df: df.assign(temp=1.5), "temp outside"),
        (lambda df: df.assign(cnt=-1), "non-negative integer"),
        (lambda df: df.assign(cnt=1.5), "non-negative integer"),
        (lambda df: df.assign(dteday="not a date"), "invalid dates"),
        (lambda df: pd.concat([df, df.head(1)]), "duplicate"),
    ],
)
def test_invalid_frames_are_rejected(change, message):
    with pytest.raises(DataContractError, match=message):
        check_frame(change(make_frame()), "train")


def test_period_and_row_count_are_checked():
    frame = make_frame("2011-01-01", days=2)

    check_frame(frame, "train", ("2011-01-01", "2011-01-02", 48))

    with pytest.raises(DataContractError, match="expected 10 rows"):
        check_frame(frame, "train", ("2011-01-01", "2011-01-02", 10))

    with pytest.raises(DataContractError, match="dates outside"):
        check_frame(frame, "train", ("2011-01-02", "2011-01-31", 48))


@pytest.mark.parametrize(
    "train_start, validation_start, test_start, message",
    [
        ("2011-01-01", "2011-01-03", "2011-01-05", None),
        ("2011-01-01", "2011-01-02", "2011-01-05", "train overlaps validation"),
        ("2011-01-01", "2011-01-03", "2011-01-04", "validation overlaps"),
    ],
)
def test_split_separation(train_start, validation_start, test_start, message):
    train = make_frame(train_start)
    validation = make_frame(validation_start)
    test = make_frame(test_start)

    if message is None:
        check_separation(train, validation, test)
    else:
        with pytest.raises(DataContractError, match=message):
            check_separation(train, validation, test)


def test_wrong_dataset_version_is_rejected(tmp_path):
    shutil.copy(DATA_DIR / "current.dvc", tmp_path / "current.dvc")
    shutil.copy(DATA_DIR / "evaluation.dvc", tmp_path / "evaluation.dvc")

    with pytest.raises(DataContractError, match="unknown dataset version"):
        load_dataset("v3", tmp_path)

    # The checked-out pointer is one version; asking for the other must fail.
    for version in ["v1", "v2"]:
        try:
            load_dataset(version, tmp_path)
        except DataContractError as error:
            assert "not dataset" in str(error)
            break
        except FileNotFoundError:
            continue  # pointer matches: CSV files are not in tmp_path
    else:
        pytest.fail("one of v1/v2 should not match the checked-out pointer")


@pytest.mark.parametrize("kind", ["baseline", "candidate"])
def test_models_train_on_contract_features(kind):
    from ml.train import PARAMS

    frame = add_features(make_frame(days=5))
    model = build_model(kind, PARAMS[kind])
    model.fit(frame[FEATURES], frame["cnt"])

    predictions = model.predict(frame[FEATURES].head(3))

    assert len(predictions) == 3
    assert np.isfinite(predictions).all()
