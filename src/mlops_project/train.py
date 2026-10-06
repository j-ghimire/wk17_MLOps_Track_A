from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import ensure_telco_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "reports" / "evidently_telco_report.html"


def _normalize_tracking_uri(tracking_uri: str | None) -> str:
    if tracking_uri is None:
        return (PROJECT_ROOT / "mlruns").as_uri()

    normalized = tracking_uri.replace("\\", "/")
    if normalized.startswith("file://") and not normalized.startswith("file:///"):
        return "file:///" + normalized[len("file://") :]
    return normalized


def _prepare_features_and_target() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    list[str],
    list[str],
]:
    df = ensure_telco_dataset()
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    cat_columns = [
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
    ]
    num_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    feature_columns = num_columns + cat_columns
    target_column = "Churn"

    X = df[feature_columns]
    y = df[target_column]

    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=43,
        stratify=y_development,
    )
    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        num_columns,
        cat_columns,
    )


def _build_model_pipeline(model_name: str):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                [
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
            ),
        ]
    )

    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
        )
    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def run_model_comparison(
    tracking_uri: str | None = None,
    reports_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train candidates, compare them by F1, and register the selected model."""
    normalized_uri = _normalize_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(normalized_uri)

    X_train, y_train, X_validation, y_validation, _, _, _, _ = _prepare_features_and_target()
    run_records: list[dict[str, object]] = []

    for model_name, params in [
        ("logistic_regression", {"solver": "lbfgs", "max_iter": 1000}),
        ("random_forest", {"n_estimators": 250, "max_depth": 10, "min_samples_leaf": 5}),
        ("decision_tree", {"max_depth": 6, "min_samples_leaf": 10}),
    ]:
        model_pipe = _build_model_pipeline(model_name)

        with mlflow.start_run(run_name=f"telco-churn-{model_name}") as run:
            model_pipe.fit(X_train, y_train)
            predictions = model_pipe.predict(X_validation)
            probabilities = model_pipe.predict_proba(X_validation)[:, 1]
            metrics = _compute_metrics(y_validation, predictions, probabilities)

            mlflow.log_params({"model_name": model_name, **params})
            for key, value in metrics.items():
                mlflow.log_metric(f"validation_{key}", float(value))

            model_info = mlflow.sklearn.log_model(
                model_pipe,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            run_records.append(
                {
                    "model_name": model_name,
                    "run_id": run.info.run_id,
                    "model_uri": model_info.model_uri,
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "roc_auc": metrics["roc_auc"],
                    "params": params,
                }
            )

    best_run = max(
        run_records,
        key=lambda item: (
            float(item["f1"]),
            float(item["recall"]),
            float(item["roc_auc"]),
        ),
    )
    X_development = pd.concat([X_train, X_validation])
    y_development = pd.concat([y_train, y_validation])
    final_pipeline = _build_model_pipeline(str(best_run["model_name"]))
    final_pipeline.fit(X_development, y_development)
    with mlflow.start_run(run_name="telco-churn-champion-refit") as final_run:
        mlflow.log_params(
            {
                "model_name": best_run["model_name"],
                "selection_metric": "validation_f1",
                "selection_value": best_run["f1"],
                "training_rows": len(X_development),
                "refit_on_development_data": True,
            }
        )
        final_model_info = mlflow.sklearn.log_model(
            final_pipeline,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

    registered_model = mlflow.register_model(
        model_uri=final_model_info.model_uri,
        name="telco-churn-model",
        tags={
            "selection_metric": "validation_f1",
            "selection_value": str(best_run["f1"]),
            "selected_candidate": str(best_run["model_name"]),
            "refit_on_development_data": "true",
        },
    )
    client = mlflow.tracking.MlflowClient(tracking_uri=normalized_uri)
    registry_transitions = []
    model_version = registered_model
    for target_stage in ("Staging", "Production"):
        previous_stage = model_version.current_stage
        model_version = client.transition_model_version_stage(
            name="telco-churn-model",
            version=str(registered_model.version),
            stage=target_stage,
            archive_existing_versions=target_stage == "Production",
        )
        registry_transitions.append(
            {
                "model_name": "telco-churn-model",
                "version": str(model_version.version),
                "from_stage": previous_stage if previous_stage not in (None, "None") else "Unassigned",
                "to_stage": model_version.current_stage,
                "run_id": final_run.info.run_id,
            }
        )
    client.set_registered_model_alias(
        name="telco-churn-model",
        alias="champion",
        version=str(registered_model.version),
    )

    output_dir = Path(reports_dir) if reports_dir is not None else PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(run_records).drop(columns="params").rename(
        columns={
            "accuracy": "validation_accuracy",
            "f1": "validation_f1",
            "precision": "validation_precision",
            "recall": "validation_recall",
            "roc_auc": "validation_roc_auc",
        }
    )
    comparison.to_csv(
        output_dir / "model_comparison.csv",
        index=False,
    )
    pd.DataFrame(registry_transitions).to_csv(
        output_dir / "registry_stage_transitions.csv",
        index=False,
    )

    return {
        "runs": run_records,
        "best_model_name": best_run["model_name"],
        "best_accuracy": float(best_run["accuracy"]),
        "best_f1": float(best_run["f1"]),
        "best_run_id": best_run["run_id"],
        "final_run_id": final_run.info.run_id,
        "model_uri": final_model_info.model_uri,
        "registered_model_version": str(registered_model.version),
        "tracking_uri": normalized_uri,
    }


def make_evidently_report(model_uri: str, tracking_uri: str | None = None) -> str:
    """Evaluate the selected model on the untouched test set and report drift."""
    from .reporting import generate_evidently_report

    mlflow.set_tracking_uri(_normalize_tracking_uri(tracking_uri))
    X_train, y_train, X_validation, y_validation, X_test, y_test, _, _ = _prepare_features_and_target()
    model = mlflow.sklearn.load_model(model_uri)

    def evaluation_frame(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        frame = features.copy()
        frame["target"] = target.to_numpy()
        frame["prediction"] = model.predict(features).astype(int)
        frame["prediction_probability"] = model.predict_proba(features)[:, 1].astype(float)
        return frame

    X_development = pd.concat([X_train, X_validation])
    y_development = pd.concat([y_train, y_validation])
    reference_df = evaluation_frame(X_development, y_development)
    current_df = evaluation_frame(X_test, y_test)
    return generate_evidently_report(reference_df, current_df, REPORT_PATH)


if __name__ == "__main__":
    result = run_model_comparison()
    print(
        f"Best model: {result['best_model_name']} | "
        f"F1={result['best_f1']:.4f} | accuracy={result['best_accuracy']:.4f}"
    )
    print(f"Best run ID: {result['best_run_id']}")
    print(f"Registered model version: {result['registered_model_version']} (alias: champion)")
    report_path = make_evidently_report(
        str(result["model_uri"]),
        tracking_uri=str(result["tracking_uri"]),
    )
    print(f"Evidently report: {report_path}")
