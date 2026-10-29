import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# Project directories
ROOT = Path(__file__).resolve().parent.parent

CONTRACT_PATH = ROOT / "contracts" / "tlc_contract_v1.json"
DATA_PATH = ROOT / "data" / "raw" / "yellow_tripdata_2024-01.parquet"
RESULTS_DIR = ROOT / "results"

FEATURES = ["trip_distance", "passenger_count"]
TARGET = "fare_amount"


def validate_contract(data_path, contract_path):
    """Validate dataset columns and types against the data contract."""

    with open(contract_path, "r", encoding="utf-8") as f:
        contract = json.load(f)

    actual_schema = pq.read_schema(data_path)

    required_columns = contract["required_columns"]
    allow_additional = contract.get("allow_additional_columns", False)

    missing_columns = []
    type_changes = {}

    for column, expected_type in required_columns.items():
        if column not in actual_schema.names:
            missing_columns.append(column)
        else:
            actual_type = str(actual_schema.field(column).type)

            if actual_type != expected_type:
                type_changes[column] = {
                    "expected": expected_type,
                    "actual": actual_type
                }

    additional_columns = [
        column
        for column in actual_schema.names
        if column not in required_columns
    ]

    if additional_columns and not allow_additional:
        print("ERROR: Additional columns are not allowed.")
        print("Additional columns:", additional_columns)
        return False

    if missing_columns:
        print("ERROR: Required columns are missing:", missing_columns)

    if type_changes:
        print("ERROR: Data type mismatches:", type_changes)

    if missing_columns or type_changes:
        return False

    print("Contract validation PASSED.")
    print("Required columns and data types are compatible.")

    if additional_columns:
        print("Additional columns allowed by contract:", additional_columns)

    return True


def load_sample(data_path, batch_size=50000):
    """Load a sample of the dataset."""

    parquet_file = pq.ParquetFile(data_path)

    batches = parquet_file.iter_batches(
        batch_size=batch_size,
        columns=FEATURES + [TARGET]
    )

    for batch in batches:
        df = batch.to_pandas()

        for column in FEATURES + [TARGET]:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        df = df[
            (df["trip_distance"] > 0)
            & (df["fare_amount"] > 0)
            & (df["passenger_count"] >= 0)
        ]

        return df

    raise ValueError("The dataset contains no readable records.")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Validating the data contract...")

    if not validate_contract(DATA_PATH, CONTRACT_PATH):
        print("Pipeline stopped because contract validation failed.")
        sys.exit(1)

    print("\nStep 2: Loading the dataset...")
    df = load_sample(DATA_PATH)

    print("Usable records:", len(df))

    if len(df) < 10:
        raise ValueError("Not enough usable records for the ML experiment.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nStep 3: Training the model...")

    model = HistGradientBoostingRegressor(
        max_iter=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    results = {
        "contract_status": "VALID",
        "dataset": DATA_PATH.name,
        "records_used": len(df),
        "features": FEATURES,
        "target": TARGET,
        "mae": float(mae),
        "rmse": rmse
    }

    output_path = RESULTS_DIR / "contract_enforced_ml_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n=== Contract-Enforced ML Results ===")
    print("Contract status:", results["contract_status"])
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()