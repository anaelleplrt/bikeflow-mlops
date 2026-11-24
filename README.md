# BikeFlow

Final-year project for the Machine Learning in Production course at EFREI Paris.

URL: https://bikeflow.delightfulsmoke-399bc548.francecentral.azurecontainerapps.io/

Sample inputs: see [Sample inputs](#sample-inputs). Deployment, triggers and
rollback: [DEPLOYMENT.md](DEPLOYMENT.md). Data and experiments:
[ml/README_data.md](ml/README_data.md), [ml/README_experiments.md](ml/README_experiments.md).

## Project overview

BikeFlow is a web application that predicts hourly bike rental demand based on weather conditions, date and time.

The goal is to build and deploy a complete MLOps pipeline, including data versioning, model tracking, automated testing, continuous integration, deployment and retraining.

## Dataset

We use the Bike Sharing Dataset from the UCI Machine Learning Repository.

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Target variable: `cnt` (total hourly bike rentals).

The `casual` and `registered` columns will not be used as model inputs to avoid data leakage.

## Technologies

- Python
- Git and GitHub
- Docker
- GitHub Actions
- DVC with Azure Blob Storage
- MLflow and Model Registry
- FastAPI web app with an HTML interface

## Git workflow

We use four types of branches:

- `main`: production
- `staging`: release validation
- `dev`: development
- `feature/*`: individual development branches created from `dev`

All changes must go through reviewed pull requests:

`feature/*` -> `dev` -> `staging` -> `main`

`dev`, `staging` and `main` are protected: a pull request and a passing CI
check are required before merging.

## Team

Three students: Anaëlle POLLART, Manon AUBEL and Grégoire ALPEROVITCH.


## Continuous integration

The CI workflow runs automatically on pull requests targeting
`dev`, `staging`, or `main`.

It can also be started manually with `workflow_dispatch`.

The workflow runs:
- Automated tests with pytest
- Static code checks with Ruff
- Docker image build
- Container smoke test using a deterministic test model

The CI model is only used for testing. Production loads the
approved MLflow champion model.

## Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Pull request to `dev`, `staging` or `main`; manual | Tests, Ruff, Docker build, container smoke test with a deterministic test model. Required check before merging. |
| `staging.yml` | Push to `staging` (merged PR); manual | Builds the release candidate, starts it with the shared MLflow registry and runs `tests/smoke_live.py`: ready, serves the current `champion` version, real prediction. |
| `deploy.yml` | Push to `main` (merged PR); manual | Builds the image, verifies it the same way, pushes it to GHCR tagged with the commit SHA, deploys that image to Azure Container Apps from `deploy/containerapp.yaml`, waits for the new revision and runs the live smoke test on the public URL. |
| `ct.yml` | Manual, input `dataset_version` (`v1`, `v2`), optional `model`, `model_name`, `force_retrain` | Pulls the DVC snapshot from Azure, validates it, scores the champion, applies the retraining rule, trains and compares a candidate, promotes it if it passes the gates, refreshes the service (`/reload`) and verifies the new version with a live prediction. Restores the previous champion if that check fails. |
| `rollback.yml` | Manual, inputs `version`, `model_name`, `reason` | Moves `champion` to a previously approved version, refreshes the service and verifies it with a live prediction. Puts the alias back if the check fails. |

`model_name` other than `bikeflow` targets a separate registered model for
controlled demos; the new version is then verified in a container started by
the workflow instead of the live service.

Deployment settings and secrets (GitHub > Settings > Secrets and variables > Actions):

- variables: `MLFLOW_TRACKING_URI`, `PRODUCTION_URL`, `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`,
  `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_LOCATION`,
  `AZURE_CONTAINER_ENVIRONMENT`, `AZURE_CONTAINER_APP`
- secrets: `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `RELOAD_TOKEN`,
  `AZURE_STORAGE_CONNECTION_STRING` (DVC remote, read by `ct.yml`)


GitHub Actions signs in to Azure with OIDC as the managed identity
`bikeflow-github-deploy` (Contributor on `bikeflow-rg` only), so no Azure
credential is stored in GitHub. The one-time Azure setup is
`deploy/azure_setup.sh`.

## Web application

The service loads the MLflow model version that holds the `champion` alias
of the registered model `bikeflow`. The alias is resolved to a version number
at start-up (and on `/reload`); only versions tagged `approved=true` are
served. Loading runs in the background: `/health` answers at once, `/ready`
returns 503 until the model has passed a warm-up prediction.

| Route | Purpose |
|---|---|
| `GET /` | Web interface: date, hour and weather in, predicted rentals and the 24-hour curve out |
| `POST /predict` | Prediction for one hour |
| `POST /predict/day` | Predictions for the 24 hours of a day |
| `GET /model-info` | Loaded model: name, version, alias, run id, dataset version |
| `GET /ready` | 200 once the approved model can predict, 503 before |
| `GET /health` | Liveness |
| `POST /reload` | Re-resolve the alias and swap the model (`Authorization: Bearer $RELOAD_TOKEN`); the previous model keeps serving if the new one fails |

Inputs: `year` (2011+), `month`, `hour`, `weekday` (0 = Sunday),
`workingday`, `weather_code` (1 clear to 4 heavy rain), `temperature_c`
(-8 to 39), `humidity_pct`, `wind_kmh` (0 to 67). The model learned on
2011-2012: later years are predicted at 2012 levels.

### Sample inputs

| Case | Payload for `POST /predict` |
|---|---|
| Monday morning rush | `{"year": 2012, "month": 6, "hour": 8, "weekday": 1, "workingday": 1, "weather_code": 1, "temperature_c": 24, "humidity_pct": 55, "wind_kmh": 10}` |
| Saturday afternoon | `{"year": 2012, "month": 6, "hour": 15, "weekday": 6, "workingday": 0, "weather_code": 1, "temperature_c": 27, "humidity_pct": 45, "wind_kmh": 12}` |
| Weekday night | `{"year": 2012, "month": 3, "hour": 3, "weekday": 3, "workingday": 1, "weather_code": 2, "temperature_c": 8, "humidity_pct": 80, "wind_kmh": 6}` |
| Rainy evening | `{"year": 2012, "month": 11, "hour": 18, "weekday": 4, "workingday": 1, "weather_code": 3, "temperature_c": 7, "humidity_pct": 90, "wind_kmh": 25}` |

### Run locally

    cp .env.example .env   # then fill in the DagsHub credentials
    set -a && source .env && set +a
    uvicorn app.main:app --reload

Settings (environment variables): `MODEL_SOURCE` (`mlflow` by default, `ci`
for the CI test model), `MODEL_NAME`, `MODEL_ALIAS`, `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, `RELOAD_TOKEN`,
`MODEL_LOAD_ATTEMPTS`, `MODEL_LOAD_RETRY_SECONDS`, `PORT`.
