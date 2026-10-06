# Track A: Data Science MLOps

This project implements a reproducible Telco churn workflow: data preparation, training, experiment comparison, model registry promotion, serving, and monitoring.

## Data and model selection

The workflow compares logistic regression, random forest, and decision tree candidates. Data is stratified into 60% training, 20% validation, and 20% test partitions. Candidates are selected by validation F1, with recall and ROC AUC also recorded. The winning candidate is refit on the 80% development partition. The untouched test partition is reserved for Evidently evaluation.

If the original dataset is unavailable, `src/mlops_project/data.py` generates a deterministic 7,000-row synthetic Telco-style dataset using seed 42. Treat metrics from this generated dataset as a reproducible demonstration, not real-world performance.

## Reproduce

From this folder, install uv if needed and sync the locked environment:

```powershell
uv sync
.\.venv\Scripts\Activate.ps1
```

Run the full workflow:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
uv run python -m mlops_project.train
```

The workflow logs all three comparison runs and a final refit run to `mlruns/`, then generates:

- `reports/model_comparison.csv`: validation metrics for all three candidates
- `reports/registry_stage_transitions.csv`: registry transitions from None to Staging to Production
- `reports/evidently_telco_report.html`: data drift and classification results using real predictions on the untouched test set

The Production model is also assigned the MLflow `champion` alias for serving. MLflow stages are deprecated in newer versions; this project records them because the assignment explicitly requests stage transitions, while retaining the alias as the serving reference.

## MLflow evidence

After running the workflow, start the tracking UI in a separate PowerShell terminal:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
uv run mlflow ui --backend-store-uri .\mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` to inspect the experiment run table and model registry. The committed comparison CSV contains metrics for all three candidates, and the registry transition CSV records promotion to Production. UI screenshots can be added to `reports/` as supplemental evidence if desired or requested by the instructor.

## Serve the champion

In another terminal, from this folder:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
uv run mlflow models serve -m "models:/telco-churn-model@champion" --host 127.0.0.1 --port 5001 --env-manager local
```

The scoring endpoint is `http://127.0.0.1:5001/invocations`. It accepts MLflow dataframe-split JSON with the model's feature columns. Send a test request from another terminal or an API client to demonstrate serving.

## Tests

```powershell
uv run pytest -q
```

## Submission contents

Include `.gitignore`, this README, `pyproject.toml`, `uv.lock`, `src/mlops_project/`, `tests/`, the dataset CSV if used, generated comparison and registry evidence CSVs, and the Evidently HTML report. MLflow UI screenshots are supplemental and can be added under `reports/` if requested. Do not include `.venv/`, `mlruns/`, `run_output.log`, `__pycache__/`, `.pytest_cache/`, or `src/*.egg-info/`.
