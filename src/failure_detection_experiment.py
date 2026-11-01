import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.contract_enforced_pipeline import validate_contract


ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "contracts" / "tlc_contract_v1.json"

OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def create_parquet(path, fields):
    schema = pa.schema(fields)

    data = {
        field.name: pa.array([], type=field.type)
        for field in schema
    }

    table = pa.table(data, schema=schema)
    pq.write_table(table, path)


def main():
    scenarios = [
        {
            "name": "Missing required column",
            "fields": [
                ("trip_distance", pa.float64()),
                ("passenger_count", pa.int64()),
            ],
            "expected_valid": False,
            "breaking": True,
        },
        {
            "name": "Wrong data type",
            "fields": [
                ("trip_distance", pa.string()),
                ("passenger_count", pa.int64()),
                ("fare_amount", pa.float64()),
            ],
            "expected_valid": False,
            "breaking": True,
        },
        {
            "name": "Additional compatible column",
            "fields": [
                ("trip_distance", pa.float64()),
                ("passenger_count", pa.int64()),
                ("fare_amount", pa.float64()),
                ("new_optional_field", pa.float64()),
            ],
            "expected_valid": True,
            "breaking": False,
        },
    ]

    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for index, scenario in enumerate(scenarios):
            data_path = Path(temp_dir) / f"scenario_{index}.parquet"

            create_parquet(data_path, scenario["fields"])

            print(f"\nTesting: {scenario['name']}")

            actual_valid = validate_contract(
                data_path,
                CONTRACT_PATH,
            )

            detected_correctly = (
                actual_valid == scenario["expected_valid"]
            )

            results.append({
                "scenario": scenario["name"],
                "expected_valid": scenario["expected_valid"],
                "actual_valid": actual_valid,
                "breaking_change": scenario["breaking"],
                "detected_correctly": detected_correctly,
            })

            print("Expected valid:", scenario["expected_valid"])
            print("Actual valid:", actual_valid)
            print("Correct outcome:", detected_correctly)

    breaking_results = [
        result for result in results
        if result["breaking_change"]
    ]

    detected = sum(
        result["detected_correctly"]
        for result in breaking_results
    )

    total = len(breaking_results)

    summary = {
        "total_scenarios": len(results),
        "breaking_scenarios": total,
        "breaking_scenarios_detected_correctly": detected,
        "breaking_change_detection_rate_percent": (
            round(100 * detected / total, 2)
            if total else None
        ),
        "compatible_scenarios_accepted": sum(
            result["detected_correctly"]
            for result in results
            if not result["breaking_change"]
        ),
        "results": results,
    }

    output_path = OUTPUT_DIR / "failure_detection_results.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n=== Failure Detection Summary ===")
    print(f"Breaking scenarios detected: {detected}/{total}")
    print(
        "Breaking-change detection rate:",
        f"{summary['breaking_change_detection_rate_percent']}%",
    )
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()