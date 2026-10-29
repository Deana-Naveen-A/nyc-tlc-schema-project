from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "raw" / "yellow_tripdata_2024-01.parquet"
CONTRACT_FILE = ROOT / "contracts" / "tlc_contract_v1.json"
RESULTS_DIR = ROOT / "results"

FEATURES = ["trip_distance", "passenger_count"]
TARGET = "fare_amount"

MAX_ROWS = 50000


def load_sample():
    """Load a limited sample instead of the entire Parquet dataset."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    parquet_file = pq.ParquetFile(DATA_FILE)

    available_columns = parquet_file.schema.names
    required = FEATURES + [TARGET]

    missing = sorted(set(required) - set(available_columns))

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    batches = parquet_file.iter_batches(
        batch_size=MAX_ROWS,
        columns=required
    )

    frames = []

    for batch in batches:
        frames.append(batch.to_pandas())

        if sum(len(frame) for frame in frames) >= MAX_ROWS:
            break

    df = pd.concat(frames, ignore_index=True)
    df = df.head(MAX_ROWS)

    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    # Keep plausible values for this simple demonstration.
    df = df[
        (df["trip_distance"] > 0)
        & (df["fare_amount"] > 0)
        & (df["passenger_count"] >= 0)
    ].copy()

    if len(df) < 100:
        raise ValueError(
            "Not enough usable records for the ML experiment."
        )

    return df


def contract_check():
    """Check whether the dataset contains the model's required fields."""

    contract = json.loads(
        CONTRACT_FILE.read_text(encoding="utf-8")
    )

    required = set(contract["required_columns"])
    model_required = set(FEATURES + [TARGET])

    missing_from_contract = model_required - required

    if missing_from_contract:
        raise ValueError(
            "The data contract does not require these model fields: "
            f"{sorted(missing_from_contract)}"
        )

    actual_columns = set(pq.read_schema(DATA_FILE).names)
    missing_from_data = model_required - actual_columns

    if missing_from_data:
        raise ValueError(
            "CONTRACT CHECK FAILED. Missing model fields: "
            f"{sorted(missing_from_data)}"
        )

    print("Contract check PASSED.")
    print("All required model fields exist in the dataset.")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading NYC TLC sample...")
    df = load_sample()

    print(f"Usable records: {len(df)}")

    X = df[FEATURES]
    y = df[TARGET]

    # Keep the same train/test split for both experiments.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = HistGradientBoostingRegressor(
        max_iter=100,
        random_state=42
    )

    print("\nTraining baseline model...")
    model.fit(X_train, y_train)

    baseline_predictions = model.predict(X_test)

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    # Simulate a downstream compatibility problem:
    # trip_distance becomes unavailable, and a faulty fallback
    # silently substitutes zero instead of rejecting the data.
    X_corrupted = X_test.copy()
    X_corrupted["trip_distance"] = 0

    corrupted_predictions = model.predict(X_corrupted)

    corrupted_mae = mean_absolute_error(
        y_test, corrupted_predictions
    )

    corrupted_rmse = np.sqrt(
        mean_squared_error(y_test, corrupted_predictions)
    )

    print("\nChecking the data contract...")
    contract_check()

    results = {
        "dataset": "NYC TLC Yellow Taxi January 2024",
        "target": TARGET,
        "features": FEATURES,
        "usable_records": int(len(df)),
        "test_records": int(len(X_test)),
        "experiment": (
            "Simulated missing trip_distance with a zero-value fallback"
        ),
        "baseline": {
            "mae": float(baseline_mae),
            "rmse": float(baseline_rmse)
        },
        "simulated_schema_problem": {
            "mae": float(corrupted_mae),
            "rmse": float(corrupted_rmse)
        },
        "mae_change": float(corrupted_mae - baseline_mae),
        "rmse_change": float(corrupted_rmse - baseline_rmse),
        "note": (
            "The corrupted experiment is a controlled simulation, "
            "not a claim about the actual impact of the 2025 column."
        )
    }

    output_file = RESULTS_DIR / "ml_experiment.json"

    output_file.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8"
    )

    print("\n=== ML Experiment Results ===")
    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Corrupted MAE: {corrupted_mae:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Corrupted RMSE: {corrupted_rmse:.4f}")

    print("\nMAE change:", f"{corrupted_mae - baseline_mae:.4f}")
    print("RMSE change:", f"{corrupted_rmse - baseline_rmse:.4f}")

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()