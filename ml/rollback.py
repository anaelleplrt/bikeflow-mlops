"""Move the champion alias to a previously approved model version.

Usage:

    python -m ml.rollback --version 1 --reason "v2 predictions drift"
    python -m ml.rollback --version 2 --model-name bikeflow-demo --output rollback.json

Only a version tagged approved=true (registered by ml.promote) can become
champion again. The switch is logged as an MLflow run (run_type=rollback).
The serving app keeps its current model until it is refreshed (POST /reload
or a revision restart); the rollback workflow does that and verifies the
version with a live prediction.
"""

import argparse
import json
import sys

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

ALIAS = "champion"
MODEL_NAME = "bikeflow"
EXPERIMENT = "bikeflow"


class RollbackError(ValueError):
    """The requested version cannot become champion."""


def rollback(version, model_name=MODEL_NAME, client=None):
    """Point the champion alias at ``version``. Returns what changed."""
    client = client or MlflowClient()
    version = str(version)

    try:
        target = client.get_model_version(model_name, version)
    except MlflowException as error:
        raise RollbackError(f"{model_name} has no version {version}") from error
    if target.tags.get("approved") != "true":
        raise RollbackError(f"{model_name} version {version} was never approved")

    try:
        previous = str(client.get_model_version_by_alias(model_name, ALIAS).version)
    except MlflowException:
        previous = None

    changed = previous != version
    if changed:
        client.set_registered_model_alias(model_name, ALIAS, version)

    return {
        "model_name": model_name,
        "previous_version": previous,
        "new_version": version,
        "dataset_version": target.tags.get("dataset_version"),
        "changed": changed,
    }


def log_rollback(result, reason):
    import mlflow

    from ml.train import git_info

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"rollback-{result['model_name']}-v{result['new_version']}"):
        mlflow.set_tags({
            **git_info(),
            "run_type": "rollback",
            "model_name": result["model_name"],
            "previous_champion_version": str(result["previous_version"]),
            "champion_version": result["new_version"],
            "reason": reason,
        })
        mlflow.log_dict({**result, "reason": reason}, "rollback.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--reason", default="manual rollback")
    parser.add_argument("--output", help="write the result as JSON to this file")
    args = parser.parse_args()

    try:
        result = rollback(args.version, args.model_name)
    except RollbackError as error:
        sys.exit(f"rollback refused: {error}")

    if result["changed"]:
        log_rollback(result, args.reason)
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
