import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.contract_enforced_pipeline import validate_contract


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = PROJECT_ROOT / "contracts" / "tlc_contract_v1.json"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2024-01.parquet"
RESULTS_PATH = PROJECT_ROOT / "results" / "failure_detection_results.json"


def create_test_parquet(output_path, schema):
    """Create an empty Parquet file with the supplied schema."""
    table = pa.Table.from_batches([], schema=schema)
    pq.write_table(table, output_path)


def main():
    # Load the real contract.
    with open(CONTRACT_PATH, "r", encoding="utf-8") as file:
        contract = json.load(file)

    required_columns = contract["required_columns"]

    # Read the real baseline schema so the test uses the correct data types.
    baseline_schema = pq.read_schema(DATA_PATH)

    missing_from_baseline = [
        column
        for column in required_columns
        if column not in baseline_schema.names
    ]

    if missing_from_baseline:
        raise ValueError(
            "The baseline dataset is missing contract columns: "
            f"{missing_from_baseline}"
        )

    # Build a complete schema containing every required contract column.
    correct_fields = [
        baseline_schema.field(column)
        for column in required_columns
    ]

    # The three scenarios differ by only one schema change each.
    scenarios = [
        {
            "name": "Missing required column",
            "change": "missing",
            "expected_valid": False,
        },
        {
            "name": "Wrong data type",
            "change": "wrong_type",
            "expected_valid": False,
        },
        {
            "name": "Additional compatible column",
            "change": "additional",
            "expected_valid": contract.get(
                "allow_additional_columns", False
            ),
        },
    ]

    results = []
    breaking_detected = 0
    breaking_total = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        for scenario in scenarios:
            print(f"\nTesting: {scenario['name']}")

            fields = list(correct_fields)

            if scenario["change"] == "missing":
                # Remove exactly one required column.
                fields = [
                    field for field in fields
                    if field.name != next(iter(required_columns))
                ]

            elif scenario["change"] == "wrong_type":
                # Change only trip_distance to string, preserving other fields.
                if "trip_distance" not in required_columns:
                    raise ValueError(
                        "'trip_distance' is not in the contract."
                    )

                fields = [
                    pa.field(field.name, pa.string())
                    if field.name == "trip_distance"
                    else field
                    for field in fields
                ]

            elif scenario["change"] == "additional":
                # Keep every required column and add one extra field.
                fields.append(
                    pa.field("experiment_extra_column", pa.string())
                )

            test_path = temp_dir / (
                scenario["change"] + ".parquet"
            )
            test_schema = pa.schema(fields)
            create_test_parquet(test_path, test_schema)

            actual_valid = validate_contract(
                test_path,
                CONTRACT_PATH,
            )

            expected_valid = scenario["expected_valid"]
            correct = actual_valid == expected_valid

            print("Expected valid:", expected_valid)
            print("Actual valid:", actual_valid)
            print("Correct outcome:", correct)

            is_breaking = scenario["change"] in {
                "missing",
                "wrong_type",
            }

            if is_breaking:
                breaking_total += 1
                if not actual_valid:
                    breaking_detected += 1

            results.append({
                "scenario": scenario["name"],
                "expected_valid": expected_valid,
                "actual_valid": actual_valid,
                "correct_outcome": correct,
                "is_breaking_change": is_breaking,
            })

    detection_rate = (
        breaking_detected / breaking_total * 100
        if breaking_total
        else 0.0
    )

    summary = {
        "results": results,
        "breaking_scenarios_detected": breaking_detected,
        "breaking_scenarios_total": breaking_total,
        "breaking_change_detection_rate_percent": detection_rate,
        "all_scenarios_correct": all(
            result["correct_outcome"] for result in results
        ),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n=== Failure Detection Summary ===")
    print(f"Breaking scenarios detected: {breaking_detected}/{breaking_total}")
    print(f"Breaking-change detection rate: {detection_rate:.1f}%")
    print("All scenarios correct:", summary["all_scenarios_correct"])
    print("Results saved to:", RESULTS_PATH)


if __name__ == "__main__":
    main()