"""Data loading, provenance and data contract checks for BikeFlow."""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Columns stored in the DVC snapshots.
SOURCE_FEATURES = [
    "mnth",
    "hr",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
]
TARGET = "cnt"
COLUMNS = ["dteday", *SOURCE_FEATURES, TARGET]

# Model inputs: the source features plus yr (years since 2011), derived from
# dteday so the model can learn the growth of demand between years.
BASE_YEAR = 2011
FEATURES = [*SOURCE_FEATURES, "yr"]

# Known dataset versions: DVC hash of data/current and expected periods.
DATASETS = {
    "v1": {
        "dvc_md5": "d0d9e62736306395adff5c47b6043278.dir",
        "train": ("2011-01-01", "2011-09-30", 6442),
        "validation": ("2011-10-01", "2011-12-31", 2203),
    },
    "v2": {
        "dvc_md5": "5c0dd375f863b5e603ab9a71a171490c.dir",
        "train": ("2011-01-01", "2012-06-30", 13003),
        "validation": ("2012-07-01", "2012-09-30", 2208),
    },
}
EVALUATION = {
    "dvc_md5": "389e6a13161da90fdd0719229fea4a80.dir",
    "test": ("2012-10-01", "2012-12-31", 2168),
}

# Allowed values and ranges (normalised scales used by the UCI dataset).
RANGES = {
    "mnth": (1, 12),
    "hr": (0, 23),
    "weekday": (0, 6),
    "workingday": (0, 1),
    "weathersit": (1, 4),
    "temp": (0.0, 1.0),
    "hum": (0.0, 1.0),
    "windspeed": (0.0, 1.0),
}


class DataContractError(ValueError):
    """Raised when a dataset does not respect the data contract."""


def dvc_hash(pointer):
    """Return the md5 stored in a .dvc pointer file."""
    content = yaml.safe_load(Path(pointer).read_text())
    return content["outs"][0]["md5"]


def check_frame(frame, name, period=None):
    """Validate one split against the data contract."""
    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        raise DataContractError(f"{name}: missing columns {missing}")

    if frame.empty:
        raise DataContractError(f"{name}: no rows")

    if frame[COLUMNS].isna().any().any():
        raise DataContractError(f"{name}: missing values")

    for column, (low, high) in RANGES.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.between(low, high).all():
            raise DataContractError(f"{name}: {column} outside [{low}, {high}]")

    target = pd.to_numeric(frame[TARGET], errors="coerce")
    if target.isna().any() or (target < 0).any() or (target % 1 != 0).any():
        raise DataContractError(f"{name}: {TARGET} must be a non-negative integer")

    dates = pd.to_datetime(frame["dteday"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise DataContractError(f"{name}: invalid dates")

    if frame.duplicated(["dteday", "hr"]).any():
        raise DataContractError(f"{name}: duplicate date and hour")

    if period is not None:
        start, end, rows = period
        if dates.min() < pd.Timestamp(start) or dates.max() > pd.Timestamp(end):
            raise DataContractError(f"{name}: dates outside {start} - {end}")
        if len(frame) != rows:
            raise DataContractError(f"{name}: expected {rows} rows, got {len(frame)}")


def add_features(frame):
    """Return a copy of a split with the derived model features."""
    frame = frame.copy()
    frame["yr"] = pd.to_datetime(frame["dteday"], format="%Y-%m-%d").dt.year - BASE_YEAR
    return frame


def check_separation(train, validation, test):
    """Training must end before validation, and validation before test."""
    train_end = pd.to_datetime(train["dteday"]).max()
    validation_start = pd.to_datetime(validation["dteday"]).min()
    validation_end = pd.to_datetime(validation["dteday"]).max()
    test_start = pd.to_datetime(test["dteday"]).min()

    if train_end >= validation_start:
        raise DataContractError("train overlaps validation")
    if validation_end >= test_start:
        raise DataContractError("validation overlaps final evaluation")


def load_dataset(version, data_dir=DATA_DIR):
    """Load and validate the checked-out dataset version.

    Returns the three splits and the provenance information to log.
    """
    if version not in DATASETS:
        raise DataContractError(f"unknown dataset version {version!r}")

    data_dir = Path(data_dir)
    expected = DATASETS[version]

    current_md5 = dvc_hash(data_dir / "current.dvc")
    evaluation_md5 = dvc_hash(data_dir / "evaluation.dvc")

    if current_md5 != expected["dvc_md5"]:
        raise DataContractError(
            f"data/current.dvc is {current_md5}, not dataset {version}. "
            f"Run: git checkout data-{version} -- data/current.dvc && dvc pull"
        )
    if evaluation_md5 != EVALUATION["dvc_md5"]:
        raise DataContractError("data/evaluation.dvc does not match the final test set")

    train = pd.read_csv(data_dir / "current" / "train.csv")
    validation = pd.read_csv(data_dir / "current" / "validation.csv")
    test = pd.read_csv(data_dir / "evaluation" / "test.csv")

    check_frame(train, "train", expected["train"])
    check_frame(validation, "validation", expected["validation"])
    check_frame(test, "test", EVALUATION["test"])
    check_separation(train, validation, test)

    train, validation, test = (add_features(f) for f in (train, validation, test))

    provenance = {
        "dataset_version": version,
        "dvc_current_md5": current_md5,
        "dvc_evaluation_md5": evaluation_md5,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    return train, validation, test, provenance
