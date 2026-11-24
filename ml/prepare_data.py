import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

FEATURES = [
    "mnth",
    "hr",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "hum",
    "windspeed",
]

COLUMNS = ["dteday", *FEATURES, "cnt"]

SPLITS = [
    ("v1", "train", "2011-01-01", "2011-09-30", 6442),
    ("v1", "validation", "2011-10-01", "2011-12-31", 2203),
    ("v2", "train", "2011-01-01", "2012-06-30", 13003),
    ("v2", "validation", "2012-07-01", "2012-09-30", 2208),
    ("v2", "test", "2012-10-01", "2012-12-31", 2168),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with ZipFile(args.source) as archive, archive.open("hour.csv") as file:
        data = pd.read_csv(file)

    required = set(COLUMNS + ["casual", "registered"])

    if not required.issubset(data.columns):
        raise ValueError("Missing required columns")

    if data[list(required)].isna().any().any():
        raise ValueError("Missing values detected")

    if not (data["casual"] + data["registered"] == data["cnt"]).all():
        raise ValueError("Invalid rental counts")

    data["dteday"] = pd.to_datetime(data["dteday"])

    if data.duplicated(["dteday", "hr"]).any():
        raise ValueError("Duplicate date and hour")

    data = data.sort_values(["dteday", "hr"])

    for version, split, start, end, expected in SPLITS:
        subset = data.loc[
            data["dteday"].between(start, end), COLUMNS
        ].copy()

        if len(subset) != expected:
            raise ValueError(f"Unexpected row count: {version}/{split}")

        subset["dteday"] = subset["dteday"].dt.strftime("%Y-%m-%d")

        folder = args.output / version
        folder.mkdir(parents=True, exist_ok=True)

        subset.to_csv(folder / f"{split}.csv", index=False)

        print(f"{version}/{split}: {len(subset)} rows")


if __name__ == "__main__":
    main()
