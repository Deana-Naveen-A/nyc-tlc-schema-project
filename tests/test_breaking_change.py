import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

CONTRACT_FILE = ROOT / "contracts" / "tlc_contract_v1.json"
DATA_FILE = ROOT / "data" / "raw" / "yellow_tripdata_2024-01.parquet"


def validate_columns(actual_columns, contract):
    expected_columns = set(contract["required_columns"])

    missing = sorted(expected_columns - set(actual_columns))

    return {
        "valid": len(missing) == 0,
        "missing_columns": missing
    }


def test_missing_required_column_is_detected():
    contract = json.loads(
        CONTRACT_FILE.read_text(encoding="utf-8")
    )

    actual_columns = pq.read_schema(DATA_FILE).names

    # Simulate a breaking change without modifying the source file.
    removed_column = actual_columns[0]

    simulated_columns = [
        column
        for column in actual_columns
        if column != removed_column
    ]

    result = validate_columns(simulated_columns, contract)

    assert result["valid"] is False
    assert removed_column in result["missing_columns"]

    print(f"\nDetected removed column: {removed_column}")


def test_additional_column_is_allowed():
    contract = json.loads(
        CONTRACT_FILE.read_text(encoding="utf-8")
    )

    actual_columns = pq.read_schema(DATA_FILE).names

    simulated_columns = actual_columns + ["example_new_field"]

    result = validate_columns(simulated_columns, contract)

    assert result["valid"] is True