"""Train a BikeFlow model and log it to MLflow.

Usage (from the repository root, with the dataset version checked out):

    python -m ml.train --data-version v1 --model baseline
    python -m ml.train --data-version v1 --model candidate

Model selection and tuning use the validation split only (model fitted on
the training split). The final model is then refitted with the same settings
on training + validation. The final evaluation split (data/evaluation/test.csv)
is scored and logged, but never used for fitting or tuning.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.data import FEATURES, TARGET, load_dataset
from ml.gates import naive_mae, regression_metrics

EXPERIMENT = "bikeflow"

# MLflow saves scikit-learn models with skops, which only loads types it
# trusts. These come from our own training code, so we declare them.
SKOPS_TRUSTED_TYPES = [
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]
CATEGORICAL = ["mnth", "hr", "weekday", "weathersit"]

# Hyperparameters chosen on the validation split.
PARAMS = {
    "baseline": {"alpha": 1.0},
    "candidate": {
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_leaf_nodes": 31,
        "random_state": 42,
    },
}


def build_model(kind, params):
    if kind == "baseline":
        encoder = ColumnTransformer(
            [("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
            remainder="passthrough",
        )
        return make_pipeline(encoder, Ridge(**params))

    if kind == "candidate":
        booster = HistGradientBoostingRegressor(
            categorical_features=[FEATURES.index(name) for name in CATEGORICAL],
            **params,
        )
        return TransformedTargetRegressor(
            regressor=booster, func=np.log1p, inverse_func=np.expm1
        )

    raise ValueError(f"unknown model type {kind!r}")


def git_info():
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        # data/current.dvc is switched between versions on purpose; the dataset
        # version is logged separately, so it does not count as a code change.
        "git_dirty": str(bool(run(
            "status", "--porcelain", "--untracked-files=no", "--", ".", ":!data/current.dvc"
        ))).lower(),
    }


def evaluation_plot(y_true, y_pred, path):
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.4)
    top = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([0, top], [0, top], color="black", linewidth=1)
    ax.set_xlabel("Actual rentals")
    ax.set_ylabel("Predicted rentals")
    ax.set_title("Final evaluation: Oct-Dec 2012")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train(data_version, kind):
    train_df, validation_df, test_df, provenance = load_dataset(data_version)
    params = PARAMS[kind]

    # 1. Validation: fit on the training split only and score the
    #    validation split. This is where models and settings are compared.
    model = build_model(kind, params)
    model.fit(train_df[FEATURES], train_df[TARGET])
    validation_metrics = regression_metrics(
        validation_df[TARGET], model.predict(validation_df[FEATURES])
    )

    # 2. Final model: refit with the same settings on training + validation,
    #    so it learns from the most recent months (demand grows over time).
    #    The final evaluation split is still never used for fitting.
    fit_df = pd.concat([train_df, validation_df], ignore_index=True)
    model = build_model(kind, params)
    model.fit(fit_df[FEATURES], fit_df[TARGET])

    test_predictions = np.clip(model.predict(test_df[FEATURES]), 0, None)
    test_metrics = regression_metrics(test_df[TARGET], test_predictions)

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{kind}-{data_version}") as run:
        mlflow.set_tags({**git_info(), **provenance, "model_type": kind})
        mlflow.log_params({
            **params,
            "features": ",".join(FEATURES),
            "final_fit": "train+validation",
            "final_fit_rows": len(fit_df),
        })
        mlflow.log_metrics({f"val_{k}": v for k, v in validation_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("val_naive_mae", naive_mae(train_df, validation_df))
        mlflow.log_metric("test_naive_mae", naive_mae(fit_df, test_df))

        with tempfile.TemporaryDirectory() as tmp:
            plot = Path(tmp) / "test_predictions.png"
            evaluation_plot(test_df[TARGET], test_predictions, plot)
            mlflow.log_artifact(plot, "evaluation")

            report = Path(tmp) / "metrics.json"
            report.write_text(json.dumps(
                {"validation": validation_metrics, "test": test_metrics, **provenance},
                indent=2,
            ))
            mlflow.log_artifact(report, "evaluation")

        example = train_df[FEATURES].head(5)
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        # DagsHub cannot resolve runs:/<id>/model with MLflow 3, so keep the
        # logged model URI (models:/m-...) on the run for promotion.
        mlflow.set_tag("model_uri", model_info.model_uri)

    print(f"run_id={run.info.run_id}")
    print(f"validation MAE={validation_metrics['mae']:.2f}  test MAE={test_metrics['mae']:.2f}")
    return run.info.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-version", required=True, choices=["v1", "v2"])
    parser.add_argument("--model", required=True, choices=list(PARAMS))
    args = parser.parse_args()
    train(args.data_version, args.model)


if __name__ == "__main__":
    main()
