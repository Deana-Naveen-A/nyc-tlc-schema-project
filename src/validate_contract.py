from pathlib import Path
import json
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

CONTRACT_FILE = ROOT / "contracts" / "tlc_contract_v1.json"

DATASETS = {
    "January 2024": ROOT / "data" / "raw" /
                    "yellow_tripdata_2024-01.parquet",

    "January 2025": ROOT / "data" / "raw" /
                    "yellow_tripdata_2025-01.parquet",
}


def validate_dataset(dataset_path, contract):
    schema = pq.read_schema(dataset_path)

    actual = {
        field.name: str(field.type)
        for field in schema
    }

    expected = contract["required_columns"]

    missing = sorted(set(expected) - set(actual))

    type_changes = {
        column: {
            "expected": expected[column],
            "actual": actual[column]
        }
        for column in expected.keys() & actual.keys()
        if expected[column] != actual[column]
    }

    additional = sorted(set(actual) - set(expected))

    errors = []

    if missing:
        errors.append(f"Missing required columns: {missing}")

    if type_changes:
        errors.append(f"Data type mismatches: {type_changes}")

    if additional and not contract["allow_additional_columns"]:
        errors.append(f"Unexpected columns: {additional}")

    return {
        "valid": len(errors) == 0,
        "missing_columns": missing,
        "additional_columns": additional,
        "type_changes": type_changes,
        "errors": errors
    }


def main():
    contract = json.loads(
        CONTRACT_FILE.read_text(encoding="utf-8")
    )

    results = {}

    for name, path in DATASETS.items():
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        result = validate_dataset(path, contract)
        results[name] = result

        print(f"\n=== {name} ===")
        print("Status:", "VALID" if result["valid"] else "INVALID")
        print("Missing columns:", result["missing_columns"])
        print("Additional columns:", result["additional_columns"])
        print("Type changes:", result["type_changes"])

        for error in result["errors"]:
            print("ERROR:", error)

    output_path = ROOT / "results" / "contract_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(results, indent=4),
        encoding="utf-8"
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()