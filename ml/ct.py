"""Continuous training: validate a dataset version, decide, train, compare, promote.

Usage (from the repository root, with the dataset version pulled by DVC):

    python -m ml.ct --data-version v2
    python -m ml.ct --data-version v2 --model baseline --model-name bikeflow-demo
    python -m ml.ct --data-version v2 --force --output ct_result.json

Steps:

1. Fetch and validate: ``load_dataset`` checks the DVC hash of the selected
   version, the data contract and the train / validation / test separation.
   Any failure stops the run before the registry is touched.
2. Evaluate and decide: the current champion is scored on the held-out test
   split and ``should_retrain`` applies the retraining rule (new dataset
   version, or champion MAE more than 10% worse than at its promotion).
   The first release, with no champion, always trains.
3. Train and compare: ``ml.train`` trains the candidate, ``ml.promote``
   scores it and the champion on the same rows and applies the quality floor
   and the minimum improvement (lower MAE is better).
4. Promote: a passing candidate is registered, tagged approved and receives
   the champion alias; a failing one is not registered and the champion stays.

The whole decision is logged as an MLflow run (run_type=continuous-training)
and written to ``--output`` for the workflow, which then refreshes and
verifies the served model.
"""

import argparse
import json
import sys

import mlflow
from mlflow import MlflowClient

from ml.data import DataContractError, load_dataset
from ml.gates import PRIMARY_METRIC, should_retrain
from ml.promote import MODEL_NAME, current_champion, promote, score
from ml.train import EXPERIMENT, PARAMS, git_info, train


def champion_state(client, model_name, test):
    """Version, training data and current test MAE of the champion, or None."""
    champion = current_champion(client, model_name)
    if champion is None:
        return None

    metrics = score(f"models:/{model_name}/{champion.version}", test)
    reference = champion.tags.get("promotion_test_mae")
    return {
        "version": str(champion.version),
        "dataset_version": champion.tags.get("dataset_version"),
        "test_mae": metrics["mae"],
        "reference_mae": float(reference) if reference else None,
    }


def plan(data_version, champion, force=False):
    """Apply the retraining rule. Returns (retrain, reason)."""
    if champion is None:
        retrain, reason = should_retrain(data_version, None, None, None)
    else:
        retrain, reason = should_retrain(
            data_version, champion["dataset_version"], champion["test_mae"], champion["reference_mae"]
        )
    if force and not retrain:
        return True, f"forced by the operator ({reason})"
    return retrain, reason


def run(data_version, kind="candidate", model_name=MODEL_NAME, force=False):
    client = MlflowClient()

    # 1. Fetch and validate (raises DataContractError, nothing changed yet).
    _, _, test, provenance = load_dataset(data_version)

    # 2. Evaluate the champion and decide.
    champion = champion_state(client, model_name, test)
    retrain, reason = plan(data_version, champion, force)
    print(f"retrain={retrain}: {reason}")

    result = {
        "model_name": model_name,
        "dataset_version": data_version,
        "model_type": kind,
        "retrain": retrain,
        "retrain_reason": reason,
        "previous_champion_version": champion["version"] if champion else None,
        "previous_champion_test_mae": champion["test_mae"] if champion else None,
        "candidate_run_id": None,
        "promotion": "not-run",
        "promotion_reason": None,
        "new_champion_version": None,
        **provenance,
    }

    # 3 and 4. Train, compare with the champion on the same rows, promote.
    if retrain:
        run_id = train(data_version, kind)
        summary = promote(run_id, data_version, model_name=model_name)
        result.update({
            "candidate_run_id": run_id,
            "candidate_test_mae": summary["candidate_test_mae"],
            "promotion": summary["decision"],
            "promotion_reason": summary["reason"],
            "new_champion_version": (
                str(summary["new_champion_version"]) if summary["new_champion_version"] is not None else None
            ),
        })

    promoted = result["new_champion_version"] is not None
    result["promoted"] = promoted
    result["champion_version"] = result["new_champion_version"] if promoted else result["previous_champion_version"]

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"ct-{data_version}"):
        mlflow.set_tags({
            **git_info(),
            "run_type": "continuous-training",
            "model_name": model_name,
            "dataset_version": data_version,
            "dvc_current_md5": provenance["dvc_current_md5"],
            "retrain": str(retrain).lower(),
            "promotion": result["promotion"],
            "previous_champion_version": str(result["previous_champion_version"]),
            "champion_version": str(result["champion_version"]),
        })
        if champion:
            mlflow.log_metric(f"champion_test_{PRIMARY_METRIC}", champion["test_mae"])
        if "candidate_test_mae" in result:
            mlflow.log_metric(f"candidate_test_{PRIMARY_METRIC}", result["candidate_test_mae"])
        mlflow.log_dict(result, "ct_decision.json")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-version", required=True, choices=["v1", "v2"])
    parser.add_argument("--model", default="candidate", choices=list(PARAMS))
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--force", action="store_true", help="train even if the rule says no")
    parser.add_argument("--output", help="write the decision as JSON to this file")
    args = parser.parse_args()

    try:
        result = run(args.data_version, args.model, args.model_name, args.force)
    except DataContractError as error:
        sys.exit(f"data validation failed, nothing changed: {error}")

    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)


if __name__ == "__main__":
    main()
