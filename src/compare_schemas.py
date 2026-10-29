from pathlib import Path
import json
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

FILES = {
    "2024": RAW_DIR / "yellow_tripdata_2024-01.parquet",
    "2025": RAW_DIR / "yellow_tripdata_2025-01.parquet",
}


def get_schema(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    schema = pq.read_schema(file_path)

    return {
        field.name: str(field.type)
        for field in schema
    }


def main():
    schema_2024 = get_schema(FILES["2024"])
    schema_2025 = get_schema(FILES["2025"])

    columns_2024 = set(schema_2024)
    columns_2025 = set(schema_2025)

    added = sorted(columns_2025 - columns_2024)
    removed = sorted(columns_2024 - columns_2025)

    common = columns_2024 & columns_2025

    changed_types = {
        col: {
            "2024": schema_2024[col],
            "2025": schema_2025[col],
        }
        for col in sorted(common)
        if schema_2024[col] != schema_2025[col]
    }

    report = {
        "columns_2024": len(schema_2024),
        "columns_2025": len(schema_2025),
        "added_in_2025": added,
        "removed_in_2025": removed,
        "changed_types": changed_types,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_file = RESULTS_DIR / "schema_comparison.json"

    output_file.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\n=== NYC TLC Schema Comparison ===")

    print(f"\n2024 columns: {len(schema_2024)}")
    print(f"2025 columns: {len(schema_2025)}")

    print("\nColumns added in 2025:")
    print(added or "None")

    print("\nColumns removed in 2025:")
    print(removed or "None")

    print("\nColumns with changed data types:")
    print(json.dumps(changed_types, indent=4))

    print(f"\nFull report saved to: {output_file}")


if __name__ == "__main__":
    main()