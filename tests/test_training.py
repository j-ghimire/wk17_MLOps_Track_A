from pathlib import Path

import mlflow
import pandas as pd
import pytest

from mlops_project.reporting import generate_evidently_report
from mlops_project.train import run_model_comparison


def test_run_model_comparison_returns_metrics(tmp_path):
    tracking_uri = f"file://{tmp_path / 'mlruns'}"

    result = run_model_comparison(tracking_uri=tracking_uri, reports_dir=tmp_path / "reports")

    assert len(result["runs"]) >= 3
    assert result["best_model_name"]
    assert 0 <= result["best_accuracy"] <= 1
    assert 0 <= result["best_f1"] <= 1
    assert result["best_run_id"]
    assert Path(tmp_path / "mlruns").exists()
    comparison = pd.read_csv(tmp_path / "reports/model_comparison.csv")
    assert len(comparison) == 3
    assert "validation_f1" in comparison.columns
    assert comparison.loc[comparison["validation_f1"].idxmax(), "model_name"] == result["best_model_name"]
    transitions = pd.read_csv(tmp_path / "reports/registry_stage_transitions.csv")
    assert transitions["from_stage"].tolist() == ["Unassigned", "Staging"]
    assert transitions["to_stage"].tolist() == ["Staging", "Production"]

    client = mlflow.tracking.MlflowClient(tracking_uri=result["tracking_uri"])
    version = client.get_model_version_by_alias("telco-churn-model", "champion")
    assert str(version.version) == result["registered_model_version"]
    assert version.current_stage == "Production"
    assert version.tags["selected_candidate"] == result["best_model_name"]
    assert version.tags["refit_on_development_data"] == "true"


def test_evidently_report_rejects_missing_real_predictions(tmp_path):
    frame = pd.DataFrame({"Churn": [0, 1]})

    with pytest.raises(ValueError, match="real target, prediction"):
        generate_evidently_report(frame, frame, tmp_path / "report.html")
