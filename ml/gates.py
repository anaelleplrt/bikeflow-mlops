"""Quality gates and retraining rule.

These values are fixed before any model is scored on the final
evaluation set (October-December 2012). Lower MAE is better.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PRIMARY_METRIC = "mae"
LOWER_IS_BETTER = True

# Quality floor: a model must be at least 15% better than a naive
# predictor (average rentals per hour of day, fitted on the same training data).
QUALITY_FLOOR_RATIO = 0.85

# Minimum improvement: a candidate must reduce the champion MAE by at least 5%.
MIN_IMPROVEMENT = 0.05

# Retraining rule: retrain when a new dataset version arrives, or when the
# champion MAE on the final evaluation set degrades by more than 10%
# compared with the MAE recorded when it was promoted.
MAX_DEGRADATION = 0.10


@dataclass
class Decision:
    promote: bool
    reason: str


def regression_metrics(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def naive_mae(train, evaluation):
    """MAE of the naive predictor: mean rentals per hour of day."""
    hourly_mean = train.groupby("hr")["cnt"].mean()
    predictions = evaluation["hr"].map(hourly_mean).fillna(train["cnt"].mean())
    return float(mean_absolute_error(evaluation["cnt"], predictions))


def decide_promotion(candidate_mae, naive, champion_mae=None):
    """Apply the quality floor and the minimum improvement rule."""
    for name, value in [("candidate", candidate_mae), ("naive", naive)]:
        if value is None or not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name} MAE: {value}")

    floor = QUALITY_FLOOR_RATIO * naive
    if candidate_mae > floor:
        return Decision(
            False,
            f"quality floor failed: MAE {candidate_mae:.2f} > {floor:.2f} "
            f"({QUALITY_FLOOR_RATIO:.0%} of naive MAE {naive:.2f})",
        )

    if champion_mae is None:
        return Decision(True, f"no champion yet, quality floor passed ({candidate_mae:.2f} <= {floor:.2f})")

    if not np.isfinite(champion_mae) or champion_mae < 0:
        raise ValueError(f"invalid champion MAE: {champion_mae}")

    required = champion_mae * (1 - MIN_IMPROVEMENT)
    if candidate_mae > required:
        return Decision(
            False,
            f"not enough improvement: MAE {candidate_mae:.2f} > {required:.2f} "
            f"(champion {champion_mae:.2f} minus {MIN_IMPROVEMENT:.0%})",
        )

    return Decision(
        True,
        f"improvement passed: MAE {candidate_mae:.2f} <= {required:.2f} (champion {champion_mae:.2f})",
    )


def should_retrain(dataset_version, champion_dataset_version, champion_mae, reference_mae):
    """Decide whether continuous training should train a new candidate."""
    if champion_dataset_version is None:
        return True, "no champion yet: train the initial model"

    if dataset_version != champion_dataset_version:
        return True, f"new dataset version {dataset_version} (champion trained on {champion_dataset_version})"

    if reference_mae and champion_mae > reference_mae * (1 + MAX_DEGRADATION):
        return True, f"champion degraded: MAE {champion_mae:.2f} > {reference_mae:.2f} + {MAX_DEGRADATION:.0%}"

    return False, "same dataset version and no performance drop"
