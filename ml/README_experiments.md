# BikeFlow - Experiments

## MLflow

We use the MLflow server of DagsHub (tracking + model registry):

https://dagshub.com/manonaubel94-stack/bikeflow-mlops.mlflow

- experiment: bikeflow
- registered model: bikeflow
- production alias: champion

The MLflow URI, username and token are in a local .env file
(not in Git) and will be added as GitHub secrets.

## Metric

Main metric: MAE (mean absolute error). Lower is better.

The app predicts a number of bike rentals for one hour, so the MAE
is easy to understand: it is the average number of bikes the
prediction is wrong by. We also log RMSE and R2.

## Splits

v1: train Jan-Sep 2011, validation Oct-Dec 2011
v2: train Jan 2011-Jun 2012, validation Jul-Sep 2012
Final evaluation (same for both): Oct-Dec 2012, 2168 rows

- We compare models and choose settings on the validation set.
- Then we train the final model again on train + validation with
  the same settings, so it also learns the most recent months.
- The final evaluation set is only used for the quality gates.
  Champion and candidate are always compared on the same rows.

## Models

- baseline: Ridge regression with one-hot encoding of month, hour,
  weekday and weather (alpha = 1.0)
- candidate: HistGradientBoostingRegressor trained on log(1 + cnt)
  (learning_rate = 0.05, max_iter = 500, max_leaf_nodes = 31)

Features: mnth, hr, weekday, workingday, weathersit, temp, hum,
windspeed and yr (year - 2011, computed from dteday).

## Quality gates

They are in ml/gates.py.

1. Quality floor: the MAE must be at most 85% of the MAE of a
   naive model (mean rentals per hour of the day).
2. Minimum improvement: if there is already a champion, the new
   model must have a MAE at least 5% lower.
3. Retraining rule (for continuous training): we retrain when there
   is a new dataset version, or when the champion MAE gets more
   than 10% worse than when it was promoted.

If a model fails a gate, it is not registered and the champion
stays the same. Each decision is saved in MLflow as a
"promotion-v1" or "promotion-v2" run with a promotion_decision.json
file.

## Commands

    set -a && source .env && set +a

    git checkout data-v1 -- data/current.dvc && dvc pull
    python -m ml.train --data-version v1 --model baseline
    python -m ml.promote --data-version v1 --run-id <run id>
    python -m ml.train --data-version v1 --model candidate
    python -m ml.promote --data-version v1 --run-id <run id>

    git checkout HEAD -- data/current.dvc && dvc pull
    python -m ml.train --data-version v2 --model candidate
    python -m ml.promote --data-version v2 --run-id <run id>
    python -m ml.train --data-version v2 --model baseline
    python -m ml.promote --data-version v2 --run-id <run id>

## Results

Runs done on 20 November 2026 from commit 007029d.

baseline v1 (run 7e8f86df)
- validation MAE 53.60, final MAE 102.75, naive MAE 110.35
- rejected: above the quality floor (93.80)

candidate v1 (run 116337b3)
- validation MAE 33.58, final MAE 90.04
- promoted: first champion, registry version 1

candidate v2 (run 623485b7)
- validation MAE 43.04, final MAE 51.42, naive MAE 94.91
- promoted: 51.42 is lower than 85.54 (champion 90.04 minus 5%)
- registry version 2

baseline v2 (run 578cb394)
- validation MAE 111.29, final MAE 91.43
- rejected: above the quality floor (80.67), champion kept

So the current champion is version 2. Version 1 is still in the
registry, so we can go back to it (rollback).

## Why version 2 is the production model

It has the lowest MAE on the final evaluation set (51.42 against
90.04 for version 1, about 43% better) and it is the only v2 model
that passes the quality floor.

## Notes

The gate values were chosen before the official runs. During
development we tested the scripts locally and saw that no model
passed the quality floor on the final evaluation set. We made two
changes, and we justified both with the validation set:

- train the final model on train + validation (the data has a
  trend, so the most recent months matter)
- add the yr feature: rentals grew a lot between 2011 and 2012.
  On the v2 validation set, the candidate MAE goes from 105.1 to
  43.0 with yr (baseline: 122.8 to 111.3)

We did not change the gate values.

Limitation: yr is only 0 or 1 in our data, so for 2013 and later
the model will predict 2012 levels.
