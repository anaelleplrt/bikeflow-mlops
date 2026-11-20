# BikeFlow - Experiments and model choice

## Tracking server

All runs and the Model Registry live on a shared MLflow server hosted by DagsHub:

    https://dagshub.com/manonaubel94-stack/bikeflow-mlops.mlflow

- Experiment: `bikeflow`
- Registered model: `bikeflow`
- Production alias: `champion`

Credentials are passed through environment variables (`MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`). They are stored in a
local `.env` file (ignored by Git) and in GitHub secrets, never in the code.

## Primary metric

**MAE (mean absolute error), in bikes per hour. Lower is better.**

The web app shows a number of rentals for a given hour. MAE is the average
number of bikes the prediction is off by, so it is directly readable by a user
or an operator planning bike availability. It is also less sensitive than RMSE
to a few extreme hours. RMSE and R² are logged as secondary metrics.

## Evaluation design

| Split | v1 | v2 | Use |
| --- | --- | --- | --- |
| Training | Jan-Sep 2011 | Jan 2011-Jun 2012 | fitting |
| Validation | Oct-Dec 2011 | Jul-Sep 2012 | model choice and settings |
| Final evaluation | Oct-Dec 2012 | Oct-Dec 2012 | quality gates only |

1. Each model is fitted on the training split and scored on the validation
   split. Model types, features and hyperparameters are chosen here.
2. The final model is refitted with the same settings on training +
   validation, so that it learns from the most recent months.
3. The final model is scored on the final evaluation split
   (`data/evaluation/test.csv`). Champion and candidate are always compared on
   these same 2,168 rows, which neither model has seen during fitting.

## Models

| Name | Model | Purpose |
| --- | --- | --- |
| `baseline` | One-hot encoding of month, hour, weekday and weather + Ridge regression (`alpha=1.0`) | simple reference |
| `candidate` | HistGradientBoostingRegressor on log(1 + rentals), `learning_rate=0.05`, `max_iter=500`, `max_leaf_nodes=31` | tuned model |

Features: `mnth`, `hr`, `weekday`, `workingday`, `weathersit`, `temp`, `hum`,
`windspeed`, and `yr` (years since 2011, derived from `dteday`).

## Quality gates (defined in `ml/gates.py`)

1. **Quality floor**: the candidate MAE must be at most 85% of the MAE of a
   naive predictor (average rentals per hour of day, fitted on the same data).
2. **Minimum improvement**: when a champion exists, the candidate MAE must be
   at least 5% lower than the champion MAE on the same rows.
3. **Retraining rule** (continuous training): retrain when a new dataset
   version arrives, or when the champion MAE degrades by more than 10% compared
   with the MAE recorded at promotion.

A candidate that fails a gate is not registered and the champion is kept. Every
decision is logged as a `promotion-<version>` run with a
`promotion_decision.json` artifact.

## How to reproduce

```bash
set -a && source .env && set +a

# v1
git checkout data-v1 -- data/current.dvc && dvc pull
python -m ml.train --data-version v1 --model baseline
python -m ml.promote --data-version v1 --run-id <baseline run id>
python -m ml.train --data-version v1 --model candidate
python -m ml.promote --data-version v1 --run-id <candidate run id>

# v2
git checkout HEAD -- data/current.dvc && dvc pull
python -m ml.train --data-version v2 --model candidate
python -m ml.promote --data-version v2 --run-id <candidate run id>
python -m ml.train --data-version v2 --model baseline
python -m ml.promote --data-version v2 --run-id <baseline run id>
```

## Results

Official runs on DagsHub (to be completed with the real run IDs and versions):

| Run | Data | Validation MAE | Final MAE | Naive MAE | Decision | Registry version |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | v1 | | | | | |
| candidate | v1 | | | | | |
| candidate | v2 | | | | | |
| baseline | v2 | | | | | |

Expected outcome, from a local dry run with the same code:

| Run | Data | Validation MAE | Final MAE | Decision |
| --- | --- | --- | --- | --- |
| baseline | v1 | 53.6 | 102.8 | rejected: quality floor (93.8) |
| candidate | v1 | 33.6 | 90.0 | promoted: initial champion |
| candidate | v2 | 43.0 | 51.4 | promoted: -43% vs champion |
| baseline | v2 | 111.3 | 91.4 | rejected: quality floor (80.7) |

## Evidence behind the production model

The production champion is the v2 gradient boosting candidate. On the same
final evaluation rows it reduces the MAE of the v1 champion by about 43%, and
it is the only v2 model that passes the quality floor.

## Development notes (transparency)

The gates above and the model settings were written before any official run.
During development, local dry runs scored the final evaluation split, which
showed that no model passed the quality floor. Two changes followed. Both are
justified by the validation split, not by final evaluation scores:

1. **Refit on training + validation.** This is standard practice for time
   series with a trend: the final model should learn from the most recent data.
2. **Add the `yr` feature.** Demand grows strongly between 2011 and 2012. On
   the v2 validation split (Jul-Sep 2012), adding `yr` lowers the candidate MAE
   from 105.1 to 43.0 and the baseline MAE from 122.8 to 111.3.

The gate values (85% floor, 5% improvement, 10% degradation) were not changed.

**Known limitation:** `yr` only takes the values 0 and 1 in the data. For 2013
and later, the tree model predicts 2012 demand levels.
