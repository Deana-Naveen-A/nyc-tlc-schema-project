import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
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


class MemoryMonitor:
    """Sample the current process's resident memory usage."""

    def __init__(self, interval=0.05):
        self.process = psutil.Process()
        self.interval = interval
        self.peak_memory_mb = 0.0
        self.stop_event = threading.Event()
        self.thread = None

    def sample_memory(self):
        while not self.stop_event.is_set():
            try:
                memory_mb = (
                    self.process.memory_info().rss / (1024 * 1024)
                )
                self.peak_memory_mb = max(
                    self.peak_memory_mb, memory_mb
                )
            except psutil.Error:
                pass

            self.stop_event.wait(self.interval)

    def start(self):
        self.peak_memory_mb = (
            self.process.memory_info().rss / (1024 * 1024)
        )
        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self.sample_memory,
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join()

        try:
            final_memory_mb = (
                self.process.memory_info().rss / (1024 * 1024)
            )
            self.peak_memory_mb = max(
                self.peak_memory_mb, final_memory_mb
            )
        except psutil.Error:
            pass

        return round(self.peak_memory_mb, 2)


def validate_contract(data_path, contract_path):
    """Validate dataset columns and types against the data contract."""

    with open(contract_path, "r", encoding="utf-8") as f:
        contract = json.load(f)

    actual_schema = pq.read_schema(data_path)

    required_columns = contract["required_columns"]
    allow_additional = contract.get(
        "allow_additional_columns", False
    )

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
        print(
            "Additional columns allowed by contract:",
            additional_columns
        )

    return True


def load_sample(data_path, target_rows, batch_size=50000):
    """
    Read Parquet batches until target_rows usable records have
    been collected or the dataset is exhausted.
    """

    parquet_file = pq.ParquetFile(data_path)

    collected = []
    usable_count = 0

    batches = parquet_file.iter_batches(
        batch_size=batch_size,
        columns=FEATURES + [TARGET]
    )

    for batch in batches:
        df = batch.to_pandas()

        for column in FEATURES + [TARGET]:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        df = df.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        df = df[
            (df["trip_distance"] > 0)
            & (df["fare_amount"] > 0)
            & (df["passenger_count"] >= 0)
        ]

        remaining = target_rows - usable_count

        if len(df) > remaining:
            df = df.iloc[:remaining]

        if not df.empty:
            collected.append(df)
            usable_count += len(df)

        print(
            f"\rUsable records collected: {usable_count:,} / "
            f"{target_rows:,}",
            end="",
            flush=True
        )

        if usable_count >= target_rows:
            break

    print()

    if not collected:
        raise ValueError("The dataset contains no usable records.")

    result = pd.concat(collected, ignore_index=True)

    if len(result) < target_rows:
        print(
            f"WARNING: Requested {target_rows:,} usable rows, "
            f"but only {len(result):,} were available."
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run a contract-enforced TLC scaling experiment."
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=50000,
        help="Number of usable rows to process."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Number of Parquet rows to read per batch."
    )

    args = parser.parse_args()

    if args.rows < 10:
        parser.error("--rows must be at least 10.")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    monitor = MemoryMonitor()
    monitor.start()
    start_time = time.perf_counter()

    try:
        print("Step 1: Validating the data contract...")

        if not validate_contract(DATA_PATH, CONTRACT_PATH):
            print(
                "Pipeline stopped because contract validation failed."
            )
            sys.exit(1)

        print("\nStep 2: Loading the dataset...")

        df = load_sample(
            DATA_PATH,
            target_rows=args.rows,
            batch_size=args.batch_size
        )

        records_used = len(df)
        print("Usable records:", f"{records_used:,}")

        if records_used < 10:
            raise ValueError(
                "Not enough usable records for the ML experiment."
            )

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
        rmse = float(
            np.sqrt(mean_squared_error(y_test, predictions))
        )

        elapsed_seconds = time.perf_counter() - start_time
        peak_memory_mb = monitor.stop()

        results = {
            "contract_status": "VALID",
            "dataset": DATA_PATH.name,
            "requested_rows": args.rows,
            "records_used": records_used,
            "batch_size": args.batch_size,
            "features": FEATURES,
            "target": TARGET,
            "mae": float(mae),
            "rmse": rmse,
            "execution_time_seconds": round(elapsed_seconds, 3),
            "peak_process_memory_mb": peak_memory_mb
        }

        output_path = (
            RESULTS_DIR / f"scaling_results_{records_used}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        print("\n=== Contract-Enforced ML Scaling Results ===")
        print("Contract status:", results["contract_status"])
        print("Requested rows:", f"{args.rows:,}")
        print("Actual usable records:", f"{records_used:,}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Execution time: {elapsed_seconds:.3f} seconds")
        print(
            "Peak sampled process memory:",
            f"{peak_memory_mb:.2f} MB"
        )
        print("Results saved to:", output_path)

    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
