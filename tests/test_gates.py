import math

import pandas as pd
import pytest

from ml.gates import decide_promotion, naive_mae, should_retrain

NAIVE = 100.0  # quality floor is then 85.0


@pytest.mark.parametrize(
    "candidate, champion, promote, reason",
    [
        (60.0, None, True, "no champion yet"),       # first release, passes floor
        (85.0, None, True, "no champion yet"),       # exactly on the floor
        (85.1, None, False, "quality floor"),        # just above the floor
        (70.0, 80.0, True, "improvement passed"),    # 12.5% better than champion
        (76.0, 80.0, True, "improvement passed"),    # exactly 5% better
        (77.0, 80.0, False, "not enough improvement"),  # only 3.75% better
        (82.0, 80.0, False, "not enough improvement"),  # worse than champion
        (90.0, 120.0, False, "quality floor"),       # beats champion but fails floor
    ],
)
def test_promotion_rule(candidate, champion, promote, reason):
    decision = decide_promotion(candidate, NAIVE, champion)

    assert decision.promote is promote
    assert reason in decision.reason


@pytest.mark.parametrize(
    "candidate, naive, champion",
    [
        (math.nan, NAIVE, None),
        (-1.0, NAIVE, None),
        (50.0, math.inf, None),
        (50.0, NAIVE, math.nan),
        (None, NAIVE, None),
    ],
)
def test_promotion_rejects_invalid_metrics(candidate, naive, champion):
    with pytest.raises(ValueError):
        decide_promotion(candidate, naive, champion)


@pytest.mark.parametrize(
    "version, champion_version, champion_mae, reference_mae, expected",
    [
        ("v1", None, None, None, True),     # no champion: initial training
        ("v2", "v1", 50.0, 50.0, True),     # new dataset version
        ("v1", "v1", 50.0, 50.0, False),    # same data, same performance
        ("v1", "v1", 54.9, 50.0, False),    # small drop under 10%
        ("v1", "v1", 56.0, 50.0, True),     # drop above 10%
    ],
)
def test_retraining_rule(version, champion_version, champion_mae, reference_mae, expected):
    retrain, reason = should_retrain(version, champion_version, champion_mae, reference_mae)

    assert retrain is expected
    assert reason


def test_naive_mae_uses_hourly_mean():
    train = pd.DataFrame({"hr": [0, 0, 1, 1], "cnt": [10, 30, 100, 300]})
    evaluation = pd.DataFrame({"hr": [0, 1, 2], "cnt": [20, 200, 110]})

    # predictions: hour 0 -> 20, hour 1 -> 200, unseen hour 2 -> global mean 110
    assert naive_mae(train, evaluation) == 0.0
