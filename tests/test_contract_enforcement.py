import json
import pyarrow as pa

from src.contract_enforced_pipeline import validate_contract


def create_test_contract(tmp_path, required_columns, allow_additional=True):
    """Create a temporary contract for testing."""

    contract = {
        "name": "Test Contract",
        "version": "1.0",
        "required_columns": required_columns,
        "allow_additional_columns": allow_additional
    }

    contract_path = tmp_path / "contract.json"

    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f)

    return contract_path


def create_test_parquet(tmp_path, schema):
    """Create a small temporary Parquet dataset."""

    import pyarrow.parquet as pq

    data = {
        field.name: pa.array([], type=field.type)
        for field in schema
    }

    table = pa.table(data, schema=schema)
    parquet_path = tmp_path / "test_data.parquet"

    pq.write_table(table, parquet_path)

    return parquet_path


def test_missing_required_column_is_rejected(tmp_path):
    schema = pa.schema([
        ("trip_distance", pa.float64())
    ])

    parquet_path = create_test_parquet(tmp_path, schema)

    contract_path = create_test_contract(
        tmp_path,
        {
            "trip_distance": "double",
            "fare_amount": "double"
        }
    )

    assert validate_contract(parquet_path, contract_path) is False


def test_wrong_data_type_is_rejected(tmp_path):
    schema = pa.schema([
        ("trip_distance", pa.string())
    ])

    parquet_path = create_test_parquet(tmp_path, schema)

    contract_path = create_test_contract(
        tmp_path,
        {
            "trip_distance": "double"
        }
    )

    assert validate_contract(parquet_path, contract_path) is False


def test_additional_column_is_allowed(tmp_path):
    schema = pa.schema([
        ("trip_distance", pa.float64()),
        ("cbd_congestion_fee", pa.float64())
    ])

    parquet_path = create_test_parquet(tmp_path, schema)

    contract_path = create_test_contract(
        tmp_path,
        {
            "trip_distance": "double"
        },
        allow_additional=True
    )

    assert validate_contract(parquet_path, contract_path) is True