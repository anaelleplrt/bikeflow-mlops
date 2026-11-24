# BikeFlow - Deployment

BikeFlow runs as one Docker container on **Azure Container Apps**. The
container serves the web interface and the API, and loads at start-up the
MLflow model version that holds the `champion` alias in the shared DagsHub
registry. Every production image is built by GitHub Actions from a merged
commit and tagged with that commit's SHA.

## Architecture

| Part | Where | Notes |
|---|---|---|
| Code, tests, workflows | GitHub `anaelleplrt/bikeflow-mlops` | Branches `feature/*` -> `dev` -> `staging` -> `main` |
| Images | GHCR `ghcr.io/anaelleplrt/bikeflow-mlops:<commit sha>` | Public package: the image holds no data and no secret |
| Model registry | DagsHub MLflow, model `bikeflow`, alias `champion` | Only versions tagged `approved=true` are served |
| Training data | Azure Blob Storage `bikeflowmlops01`, container `bikeflow-dvc` | DVC remote, never in the image |
| Production service | Azure Container Apps, app `bikeflow` in environment `bikeflow-env`, resource group `bikeflow-rg`, France Central | 1 replica, 0.5 vCPU, 1 GiB |
| Deploy identity | Managed identity `bikeflow-github-deploy` | Contributor on `bikeflow-rg` only, used by Actions through OIDC |

## Which branch triggers which job

| Event | Workflow | Jobs and checks | Result |
|---|---|---|---|
| Pull request to `dev`, `staging` or `main` | `ci.yml` | pytest, Ruff, Docker build, container smoke test with a deterministic test model (`tests/smoke_ci.py`) | Required check: a failing PR cannot merge |
| Push to `staging` (merged PR) | `staging.yml` | Build the release candidate, start it with the shared registry, `tests/smoke_live.py` | Release candidate validated: it becomes ready and serves the current champion |
| Push to `main` (merged PR) | `deploy.yml` job `build` | Build, start with the shared registry, `tests/smoke_live.py`, push `ghcr.io/...:<sha>` | Only a verified image is pushed |
| | `deploy.yml` job `deploy` (environment `production`) | Render `deploy/containerapp.yaml`, create or update the app with that image, wait for the revision `bikeflow--<sha7>-<attempt>` to be healthy, `tests/smoke_live.py` on the public URL | Live service checked with a real prediction and the expected model version |

All three workflows can also be started by hand (`workflow_dispatch`).

`tests/smoke_live.py` reads the version that holds `champion` in the registry
and fails unless `/ready`, `/model-info` and `/predict` all report that
version and the prediction is a plausible number.

## Configuration and secrets

Nothing environment-specific is in the image. The container reads:

| Variable | Production source | Purpose |
|---|---|---|
| `MODEL_SOURCE` | `mlflow` (in `containerapp.yaml`) | `ci` is only used by the CI smoke test |
| `MODEL_NAME`, `MODEL_ALIAS` | `bikeflow`, `champion` | Registry model and alias to serve |
| `MLFLOW_TRACKING_URI` | GitHub variable | DagsHub MLflow server |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | GitHub secrets, stored as Container Apps secrets | DagsHub token |
| `RELOAD_TOKEN` | GitHub secret, stored as a Container Apps secret | Enables `POST /reload` |
| `MODEL_LOAD_ATTEMPTS`, `MODEL_LOAD_RETRY_SECONDS` | `5` attempts, 5 s | Start-up retries |
| `PORT` | default `8000` | Listening port |

GitHub variables: `MLFLOW_TRACKING_URI`, `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`,
`AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_LOCATION`,
`AZURE_CONTAINER_ENVIRONMENT`, `AZURE_CONTAINER_APP`.
GitHub secrets: `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`,
`RELOAD_TOKEN`. No Azure credential is stored: Actions signs in with OIDC.

## Reproduce a deployment

### One-time Azure setup

Done once, by someone with Owner on `bikeflow-rg`:

    az login
    AZURE_SUBSCRIPTION="<subscription id>" bash deploy/azure_setup.sh

It creates the Container Apps environment and the deploy identity, trusts
GitHub tokens for this repository (`environment:production` and
`refs/heads/main`), and prints the GitHub variables to set. The subscription
owner must have registered the providers `Microsoft.App`,
`Microsoft.OperationalInsights` and `Microsoft.ManagedIdentity`.

### Normal deployment

Merge `staging` into `main` through a pull request. `deploy.yml` builds,
verifies, pushes and deploys the merged commit, then prints the public URL in
the run summary (environment `production`).

### Redeploy a given commit

Actions -> Deploy -> Run workflow, on the branch or tag to deploy. The image
is rebuilt from that ref and tagged with its SHA.

### Deploy by hand (same result as the workflow)

    export IMAGE=ghcr.io/anaelleplrt/bikeflow-mlops:<sha>
    export REVISION_SUFFIX=<sha7>-manual
    export AZURE_LOCATION=francecentral MODEL_NAME=bikeflow MODEL_ALIAS=champion
    export AZURE_ENVIRONMENT_ID=$(az containerapp env show -n bikeflow-env -g bikeflow-rg --query id -o tsv)
    set -a && source .env && set +a    # MLflow credentials and RELOAD_TOKEN (must be set)
    envsubst < deploy/containerapp.yaml > /tmp/app.yaml
    az containerapp update -n bikeflow -g bikeflow-rg --yaml /tmp/app.yaml
    rm /tmp/app.yaml
    python tests/smoke_live.py --url https://$(az containerapp show -n bikeflow -g bikeflow-rg --query properties.configuration.ingress.fqdn -o tsv)

### Run the production image locally

    docker run --rm -p 8000:8000 --env-file .env ghcr.io/anaelleplrt/bikeflow-mlops:<sha>

## Model changes, readiness and rollback

The service loads the model once at start-up, in the background:
`/health` answers at once, `/ready` returns 503 until the version holding
`champion` is loaded, tagged `approved=true` and has passed a warm-up
prediction. `/model-info` reports the version actually loaded.

**After the alias moves** (a promotion, or a rollback), the running service
must load the new version, in one of two ways:

- `POST /reload` with `Authorization: Bearer $RELOAD_TOKEN` re-resolves the
  alias and swaps the model in memory. It answers with the previous and new
  versions. The app runs a single replica, so one call refreshes the whole
  service.
- `az containerapp revision restart -n bikeflow -g bikeflow-rg --revision <active revision>`
  restarts the container, which loads the alias again.

Then `python tests/smoke_live.py --url <public url>` checks that the new
version is served.

**Manual rollback** to a previously approved version (the continuous
training phase will wrap this in a workflow):

    set -a && source .env && set +a    # DagsHub credentials and RELOAD_TOKEN
    python -c "from mlflow import MlflowClient; MlflowClient().set_registered_model_alias('bikeflow', 'champion', '1')"
    curl -X POST -H "Authorization: Bearer $RELOAD_TOKEN" https://<public url>/reload
    python tests/smoke_live.py --url https://<public url> --expected-version 1

### If the app does not become ready

| Situation | What happens | What to do |
|---|---|---|
| `/reload` with a version that cannot load or predict | `/reload` returns 503 with the error; the previous model keeps serving and `/ready` stays 200 with the previous version | Move the alias back, or fix the version, then reload |
| New revision (deploy or restart) cannot load the model | The container retries 5 times, then `/ready` stays 503. The readiness probe fails, so Container Apps sends no traffic to that revision; in single-revision mode the previous healthy revision keeps serving until a new one is ready. `deploy.yml` fails and lists the revisions | Read the logs (`az containerapp logs show -n bikeflow -g bikeflow-rg`), then redeploy a previous commit or move the alias back |
| Registry unreachable at start-up | Same as above: retries, then not ready | Restart the revision once DagsHub is reachable |

## The twelve factors

| Factor | How BikeFlow applies it |
|---|---|
| I. Codebase | One Git repository. The same commit runs in CI, as the staging container and in production; each deploy is a revision of that commit. |
| II. Dependencies | Declared in `requirements.txt` (serving) and `requirements-dev.txt` (training, tests). `scikit-learn` and `skops` are pinned to the versions recorded with the registered models so they always load; the image installs nothing at run time. |
| III. Config | Everything that varies between deploys is an environment variable (table above). Secrets live in GitHub secrets and Container Apps secrets; `.env` is local and ignored by Git, `.env.example` documents it. |
| IV. Backing services | The MLflow registry (DagsHub) and the DVC storage (Azure Blob) are attached resources reached by URL and credentials. A test registry (local SQLite) is swapped in by tests just by changing `MLFLOW_TRACKING_URI`. |
| V. Build, release, run | Build: Actions builds the image of a merged commit. Release: that image (`:<sha>`) plus the rendered `deploy/containerapp.yaml` becomes an immutable revision `bikeflow--<sha7>-<attempt>`. Run: Container Apps runs the revision. The model is chosen at run time by the alias, so a promotion needs no rebuild. |
| VI. Processes | The service is stateless: the model is read from the registry into memory and nothing is written to local disk. A restart rebuilds the same state. |
| VII. Port binding | The app is self-contained (Uvicorn) and binds the port given by `PORT`, 8000 by default; Container Apps ingress targets it and terminates HTTPS. |
| VIII. Concurrency | Limited relevance: one small process is enough for the expected load, and we keep exactly one replica so `/reload` refreshes the whole service. Scaling out would mean raising `maxReplicas` and refreshing by restarting the revision instead of `/reload`. |
| IX. Disposability | Fast start (about 20 s, mostly the model download), loading in the background behind `/ready`, Uvicorn started with `exec` so it receives SIGTERM. A revision can be killed and replaced at any time. |
| X. Dev/prod parity | The same image runs in CI, staging and production; staging and production use the same registry and alias. The only difference is the deterministic test model in the CI smoke test, which has no registry access. |
| XI. Logs | The app writes to stdout; Container Apps streams it (`az containerapp logs show`). No Log Analytics workspace, to stay within the student credit. |
| XII. Admin processes | One-off tasks run from the same codebase and dependencies: training and promotion (`python -m ml.train`, `python -m ml.promote`), the Azure setup script, and the rollback commands above. |
