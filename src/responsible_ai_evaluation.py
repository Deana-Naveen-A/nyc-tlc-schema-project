import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "yellow_tripdata_2024-01.parquet"
OUTPUT_DIR = ROOT / "results" / "responsible_ai"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    start_time = time.perf_counter()

    # 1. Load only the fields required for this experiment.
    print("Loading public NYC TLC trip data...")

    columns = [
        "trip_distance",
        "passenger_count",
        "fare_amount",
    ]

    df = pd.read_parquet(DATA_PATH, columns=columns)

    original_rows = len(df)

    # 2. Clean invalid and missing values.
    df = df.dropna(subset=columns)

    df = df[
        (df["trip_distance"] > 0)
        & (df["passenger_count"] >= 0)
        & (df["fare_amount"] > 0)
    ].copy()

    # Restrict extreme values to reduce the influence of
    # clearly implausible records on this demonstration.
    df = df[
        (df["trip_distance"] <= 100)
        & (df["fare_amount"] <= 500)
        & (df["passenger_count"] <= 10)
    ].copy()

    if len(df) < 100:
        raise ValueError(
            "Not enough valid records for the evaluation."
        )

    print(f"Original records: {original_rows:,}")
    print(f"Usable records: {len(df):,}")

    # 3. Define model inputs and target.
    X = df[["trip_distance", "passenger_count"]]
    y = df["fare_amount"]

    # 4. Use a fixed random seed for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # 5. Train the model.
    model = HistGradientBoostingRegressor(
        random_state=42
    )

    model.fit(X_train, y_train)

    # 6. Generate predictions on the held-out test set.
    predictions = model.predict(X_test)

    overall_mae = mean_absolute_error(
        y_test,
        predictions,
    )

    overall_rmse = mean_squared_error(
        y_test,
        predictions,
    ) ** 0.5

    # 7. Group test records by trip distance.
    evaluation = X_test.copy()

    evaluation["actual_fare"] = y_test
    evaluation["predicted_fare"] = predictions

    evaluation["distance_category"] = pd.cut(
        evaluation["trip_distance"],
        bins=[0, 2, 5, float("inf")],
        labels=["Short (0-2 miles)", "Medium (2-5 miles)",
                "Long (>5 miles)"],
        right=True,
    )

    # 8. Calculate errors for each category.
    results = []

    for category, group in evaluation.groupby(
        "distance_category",
        observed=False,
    ):
        if group.empty:
            continue

        mae = mean_absolute_error(
            group["actual_fare"],
            group["predicted_fare"],
        )

        rmse = mean_squared_error(
            group["actual_fare"],
            group["predicted_fare"],
        ) ** 0.5

        results.append(
            {
                "distance_category": str(category),
                "records": len(group),
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
            }
        )

    results_df = pd.DataFrame(results)

    # 9. Save the detailed results.
    csv_path = OUTPUT_DIR / "fairness_error_analysis.csv"
    results_df.to_csv(csv_path, index=False)

    # 10. Generate a graph comparing category errors.
    ax = results_df.set_index(
        "distance_category"
    )[["MAE", "RMSE"]].plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_title("Prediction Error by Trip-Distance Category")
    ax.set_xlabel("Trip-distance category")
    ax.set_ylabel("Prediction error (fare units)")
    ax.legend(title="Metric")
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()

    graph_path = OUTPUT_DIR / "fairness_error_analysis.png"
    plt.savefig(graph_path, dpi=150)
    plt.close()

    # 11. Save a summary for the report.
    summary = {
        "experiment": "Trip-distance-group error analysis",
        "dataset": "NYC TLC Yellow Taxi, January 2024",
        "original_records": int(original_rows),
        "usable_records": int(len(df)),
        "training_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "features": ["trip_distance", "passenger_count"],
        "target": "fare_amount",
        "random_state": 42,
        "overall_MAE": round(float(overall_mae), 4),
        "overall_RMSE": round(float(overall_rmse), 4),
        "group_results": results,
        "execution_time_seconds": round(
            time.perf_counter() - start_time, 3
        ),
        "interpretation_note": (
            "Differences in prediction error across distance "
            "categories indicate performance variation. They do "
            "not independently establish demographic unfairness."
        ),
    }

    json_path = OUTPUT_DIR / "responsible_ai_summary.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # 12. Display the measured evidence.
    print("\nOverall model performance")
    print(f"MAE:  {overall_mae:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")

    print("\nPrediction errors by trip-distance category")
    print(results_df.to_string(index=False))

    print("\nResponsible AI evaluation completed.")
    print(f"CSV results: {csv_path}")
    print(f"Graph:       {graph_path}")
    print(f"JSON summary:{json_path}")


if __name__ == "__main__":
    main()