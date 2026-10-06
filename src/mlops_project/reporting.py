from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_evidently_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: str | Path) -> str:
    """Generate an Evidently HTML drift and classification report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    reference_df = reference_df.copy()
    current_df = current_df.copy()

    for frame in (reference_df, current_df):
        if "target" not in frame.columns and "Churn" in frame.columns:
            frame.rename(columns={"Churn": "target"}, inplace=True)
        required_columns = {"target", "prediction", "prediction_probability"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                "Evidently classification reports require real target, prediction, and "
                f"prediction_probability columns; missing: {', '.join(sorted(missing_columns))}"
            )

    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.core.datasets import BinaryClassification
        from evidently.presets import ClassificationPreset, DataDriftPreset
    except ImportError as exc:  # pragma: no cover - dependency should be present
        raise RuntimeError("Evidently is not installed. Add it to the project dependencies.") from exc

    data_definition = DataDefinition(
        numerical_columns=[
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ],
        categorical_columns=[
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ],
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels="prediction",
                prediction_probas="prediction_probability",
            )
        ],
    )

    report = Report(
        metrics=[
            DataDriftPreset(),
            ClassificationPreset(),
        ]
    )
    snapshot = report.run(
        current_data=Dataset.from_pandas(current_df, data_definition=data_definition),
        reference_data=Dataset.from_pandas(reference_df, data_definition=data_definition),
    )
    html_path = output.with_suffix(".html")
    snapshot.save_html(str(html_path))
    return str(html_path)
