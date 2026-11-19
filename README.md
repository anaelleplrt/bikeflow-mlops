# BikeFlow

Final-year project for the Machine Learning in Production course at EFREI Paris.

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
- Web API and frontend (to be implemented)

## Git workflow

We use four types of branches:

- `main`: production
- `staging`: release validation
- `dev`: development
- `feature/*`: individual development branches created from `dev`

All changes must go through reviewed pull requests:

`feature/*` -> `dev` -> `staging` -> `main`

Automated CI checks will be required before merging.

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

The CI model is only used for testing. Production will load
the approved MLflow champion model.

Staging, production deployment and continuous training
workflows will be added in later project phases.
