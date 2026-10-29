from pathlib import Path
import json
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "raw" / "yellow_tripdata_2024-01.parquet"
CONTRACT_FILE = ROOT / "contracts" / "tlc_contract_v1.json"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    schema = pq.read_schema(DATA_FILE)

    contract = {
        "contract_name": "NYC TLC Yellow Taxi",
        "version": "1.0",
        "description": "Baseline schema derived from January 2024 data",
        "allow_additional_columns": True,
        "required_columns": {
            field.name: str(field.type)
            for field in schema
        }
    }

    CONTRACT_FILE.parent.mkdir(parents=True, exist_ok=True)

    CONTRACT_FILE.write_text(
        json.dumps(contract, indent=4),
        encoding="utf-8"
    )

    print("Data contract created successfully!")
    print(f"Required columns: {len(contract['required_columns'])}")
    print(f"Saved to: {CONTRACT_FILE}")


if __name__ == "__main__":
    main()