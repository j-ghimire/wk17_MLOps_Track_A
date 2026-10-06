from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_PATH = DATA_DIR / "telco_customer_churn.csv"


def ensure_telco_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Create a realistic telco churn dataset if it is not already present."""
    target_path = Path(path) if path is not None else DATA_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        return pd.read_csv(target_path)

    rng = np.random.default_rng(42)
    n_rows = 7000

    genders = ["Male", "Female"]
    partner = ["Yes", "No"]
    dependents = ["Yes", "No"]
    phone_service = ["Yes", "No"]
    multiple_lines = ["No phone service", "Yes", "No"]
    internet_service = ["DSL", "Fiber optic", "No"]
    online_security = ["Yes", "No", "No internet service"]
    online_backup = ["Yes", "No", "No internet service"]
    device_protection = ["Yes", "No", "No internet service"]
    tech_support = ["Yes", "No", "No internet service"]
    streaming_tv = ["Yes", "No", "No internet service"]
    streaming_movies = ["Yes", "No", "No internet service"]
    contract_types = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]

    customer_id = [f"CUST-{idx:05d}" for idx in range(1, n_rows + 1)]
    tenure = rng.integers(1, 72, size=n_rows)
    gender = rng.choice(genders, size=n_rows)
    senior = rng.binomial(1, 0.18, size=n_rows)
    partner_flag = rng.choice(partner, size=n_rows)
    dependent_flag = rng.choice(dependents, size=n_rows)
    monthly_charges = np.round(rng.uniform(20, 120, size=n_rows), 2)
    total_charges = np.round(monthly_charges * (tenure / 2) + rng.normal(0, 30, size=n_rows), 2)

    df = pd.DataFrame(
        {
            "customerID": customer_id,
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner_flag,
            "Dependents": dependent_flag,
            "tenure": tenure,
            "PhoneService": rng.choice(phone_service, size=n_rows),
            "MultipleLines": rng.choice(multiple_lines, size=n_rows),
            "InternetService": rng.choice(internet_service, size=n_rows),
            "OnlineSecurity": rng.choice(online_security, size=n_rows),
            "OnlineBackup": rng.choice(online_backup, size=n_rows),
            "DeviceProtection": rng.choice(device_protection, size=n_rows),
            "TechSupport": rng.choice(tech_support, size=n_rows),
            "StreamingTV": rng.choice(streaming_tv, size=n_rows),
            "StreamingMovies": rng.choice(streaming_movies, size=n_rows),
            "Contract": rng.choice(contract_types, size=n_rows),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n_rows),
            "PaymentMethod": rng.choice(payment_methods, size=n_rows),
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
    )

    churn_score = (
        (df["tenure"] < 12).astype(int) * 0.35
        + (df["SeniorCitizen"] == 1).astype(int) * 0.15
        + (df["Contract"] == "Month-to-month").astype(int) * 0.25
        + (df["InternetService"] == "Fiber optic").astype(int) * 0.20
        + (df["MonthlyCharges"] > 75).astype(int) * 0.15
        + (df["PaperlessBilling"] == "Yes").astype(int) * 0.10
    )
    churn_probability = np.clip(churn_score / 1.2, 0, 0.95)
    df["Churn"] = rng.binomial(1, churn_probability)

    df["TotalCharges"] = df["TotalCharges"].astype(float)
    df.to_csv(target_path, index=False)
    return df
