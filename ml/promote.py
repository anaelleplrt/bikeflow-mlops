"""Compare a candidate run with the current champion and promote it if it passes.

Usage:

    python -m ml.promote --run-id <candidate run id> --data-version v1

Both models are scored on the same final evaluation rows
(data/evaluation/test.csv). The decision is logged as its own MLflow run.
"""

import argparse
import json

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from ml.data import FEATURES, TARGET, load_dataset
from ml.gates import decide_promotion, naive_mae, regression_metrics
from ml.train import EXPERIMENT

MODEL_NAME = "bikeflow"
ALIAS = "champion"


def current_champion(client, model_name=MODEL_NAME):
    """Return the model version holding the champion alias, or None."""
    try:
        return client.get_model_version_by_alias(model_name, ALIAS)
    except MlflowException:
        return None


def score(model_uri, test):
    model = mlflow.sklearn.load_model(model_uri)
    return regression_metrics(test[TARGET], model.predict(test[FEATURES]))


def promote(run_id, data_version, model_name=MODEL_NAME, dry_run=False):
    client = MlflowClient()
    train_df, validation_df, test_df, provenance = load_dataset(data_version)
    fit_df = pd.concat([train_df, validation_df], ignore_index=True)

    candidate_run = client.get_run(run_id)
    candidate_uri = candidate_run.data.tags.get("model_uri")
    if not candidate_uri:
        raise SystemExit(f"run {run_id} has no model_uri tag (was it created by ml.train?)")
    trained_on = candidate_run.data.tags.get("dataset_version")
    if trained_on != data_version:
        raise SystemExit(f"run {run_id} was trained on {trained_on}, not {data_version}")

    candidate = score(candidate_uri, test_df)
    # Naive reference uses the same data as the final model fit.
    naive = naive_mae(fit_df, test_df)

    champion = current_champion(client, model_name)
    champion_metrics = None
    if champion is not None:
        champion_metrics = score(f"models:/{model_name}/{champion.version}", test_df)

    decision = decide_promotion(
        candidate["mae"], naive, champion_metrics["mae"] if champion_metrics else None
    )

    new_version = None
    if decision.promote and not dry_run:
        registered = mlflow.register_model(candidate_uri, model_name)
        new_version = registered.version
        for key, value in {
            "approved": "true",
            "dataset_version": data_version,
            "source_run_id": run_id,
            "promotion_test_mae": f"{candidate['mae']:.4f}",
        }.items():
            client.set_model_version_tag(model_name, new_version, key, value)
        client.set_registered_model_alias(model_name, ALIAS, new_version)

    summary = {
        "decision": "promoted" if decision.promote else "rejected",
        "reason": decision.reason,
        "candidate_run_id": run_id,
        "candidate_test_mae": candidate["mae"],
        "naive_test_mae": naive,
        "previous_champion_version": champion.version if champion else None,
        "previous_champion_test_mae": champion_metrics["mae"] if champion_metrics else None,
        "new_champion_version": new_version,
        "dry_run": dry_run,
        **provenance,
    }

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"promotion-{data_version}"):
        mlflow.set_tags({
            "run_type": "promotion",
            "decision": summary["decision"],
            "candidate_run_id": run_id,
            "previous_champion_version": str(summary["previous_champion_version"]),
            "new_champion_version": str(new_version),
        })
        mlflow.log_metric("candidate_test_mae", candidate["mae"])
        mlflow.log_metric("naive_test_mae", naive)
        if champion_metrics:
            mlflow.log_metric("champion_test_mae", champion_metrics["mae"])
        mlflow.log_dict(summary, "promotion_decision.json")

    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-version", required=True, choices=["v1", "v2"])
    parser.add_argument("--dry-run", action="store_true", help="decide without registering")
    args = parser.parse_args()
    promote(args.run_id, args.data_version, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
