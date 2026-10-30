import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
ANALYSIS_DIR = RESULTS_DIR / "scaling_analysis"

ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# STEP 1: LOAD ALL SCALING RESULTS
# --------------------------------------------------

def load_results():
    """Load all scaling experiment JSON files."""

    result_files = sorted(
        RESULTS_DIR.glob("scaling_results_*.json")
    )

    if not result_files:
        raise FileNotFoundError(
            "No scaling result files found in the results folder."
        )

    records = []

    for file_path in result_files:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        records.append({
            "dataset": data.get("dataset", "Unknown"),
            "contract_status": data.get("contract_status", "Unknown"),
            "requested_rows": data.get("requested_rows"),
            "records_used": data.get("records_used"),
            "batch_size": data.get("batch_size"),
            "execution_time_seconds": data.get(
                "execution_time_seconds"
            ),
            "peak_process_memory_mb": data.get(
                "peak_process_memory_mb"
            ),
            "mae": data.get("mae"),
            "rmse": data.get("rmse"),
            "source_file": file_path.name
        })

    df = pd.DataFrame(records)

    df["records_used"] = pd.to_numeric(
        df["records_used"], errors="coerce"
    )

    df["requested_rows"] = pd.to_numeric(
        df["requested_rows"], errors="coerce"
    )

    df["execution_time_seconds"] = pd.to_numeric(
        df["execution_time_seconds"], errors="coerce"
    )

    df["peak_process_memory_mb"] = pd.to_numeric(
        df["peak_process_memory_mb"], errors="coerce"
    )

    df["mae"] = pd.to_numeric(
        df["mae"], errors="coerce"
    )

    df["rmse"] = pd.to_numeric(
        df["rmse"], errors="coerce"
    )

    df = df.sort_values(
        by="records_used"
    ).reset_index(drop=True)

    return df


# --------------------------------------------------
# STEP 2: GENERATE COMPARISON TABLES
# --------------------------------------------------

def save_tables(df):
    """Save scaling results as CSV and Markdown tables."""

    csv_path = ANALYSIS_DIR / "scaling_comparison.csv"
    markdown_path = ANALYSIS_DIR / "scaling_comparison.md"

    df.to_csv(csv_path, index=False)

    with open(
        markdown_path, "w", encoding="utf-8"
    ) as file:
        file.write("# NYC TLC Scaling Experiment Results\n\n")
        file.write(
            df.to_markdown(index=False, floatfmt=".4f")
        )
        file.write("\n")

    print("\nComparison table:")
    print(
        df[
            [
                "requested_rows",
                "records_used",
                "contract_status",
                "execution_time_seconds",
                "peak_process_memory_mb",
                "mae",
                "rmse"
            ]
        ].to_string(index=False)
    )

    print("\nCSV saved to:", csv_path)
    print("Markdown table saved to:", markdown_path)


# --------------------------------------------------
# STEP 3: CREATE SCALING GRAPHS
# --------------------------------------------------

def create_graphs(df):
    """Generate graphs from the recorded experiments."""

    # Plot 1: Dataset size vs execution time
    timing_df = df.dropna(
        subset=["records_used", "execution_time_seconds"]
    )

    if not timing_df.empty:
        plt.figure(figsize=(9, 5))

        plt.plot(
            timing_df["records_used"],
            timing_df["execution_time_seconds"],
            marker="o"
        )

        plt.title("Dataset Size vs Execution Time")
        plt.xlabel("Number of Usable Records")
        plt.ylabel("Execution Time (seconds)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = ANALYSIS_DIR / "scaling_execution_time.png"
        plt.savefig(path, dpi=300)
        plt.close()

        print("Execution-time graph saved to:", path)

    # Plot 2: Dataset size vs peak memory
    memory_df = df.dropna(
        subset=["records_used", "peak_process_memory_mb"]
    )

    if not memory_df.empty:
        plt.figure(figsize=(9, 5))

        plt.plot(
            memory_df["records_used"],
            memory_df["peak_process_memory_mb"],
            marker="o"
        )

        plt.title("Dataset Size vs Peak Sampled Process Memory")
        plt.xlabel("Number of Usable Records")
        plt.ylabel("Peak Sampled Memory (MB)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = ANALYSIS_DIR / "scaling_memory_usage.png"
        plt.savefig(path, dpi=300)
        plt.close()

        print("Memory graph saved to:", path)

    # Plot 3: Dataset size vs MAE and RMSE
    metric_df = df.dropna(
        subset=["records_used", "mae", "rmse"]
    )

    if not metric_df.empty:
        plt.figure(figsize=(9, 5))

        plt.plot(
            metric_df["records_used"],
            metric_df["mae"],
            marker="o",
            label="MAE"
        )

        plt.plot(
            metric_df["records_used"],
            metric_df["rmse"],
            marker="s",
            label="RMSE"
        )

        plt.title("Dataset Size vs Model Error")
        plt.xlabel("Number of Usable Records")
        plt.ylabel("Error (fare units)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = ANALYSIS_DIR / "scaling_model_performance.png"
        plt.savefig(path, dpi=300)
        plt.close()

        print("Model-performance graph saved to:", path)


# --------------------------------------------------
# STEP 4: GENERATE A SUMMARY REPORT
# --------------------------------------------------

def create_report(df):
    """Create a text report using the actual experiment results."""

    report_path = ANALYSIS_DIR / "scaling_report.txt"

    successful_contracts = (
        df["contract_status"].astype(str).str.upper() == "VALID"
    )

    actual_rows = df["records_used"].dropna()

    lines = [
        "NYC TLC SCHEMA EVOLUTION AND DATA CONTRACTS",
        "SCALING EXPERIMENT REPORT",
        "=" * 55,
        "",
        f"Number of experiment result files: {len(df)}",
        f"Contract validation passed: {successful_contracts.all()}",
        "",
    ]

    if not actual_rows.empty:
        lines.extend([
            f"Smallest experiment: {int(actual_rows.min()):,} rows",
            f"Largest experiment: {int(actual_rows.max()):,} rows",
            "",
        ])

    fastest = df.dropna(
        subset=["execution_time_seconds"]
    )

    if not fastest.empty:
        fastest_row = fastest.loc[
            fastest["execution_time_seconds"].idxmin()
        ]

        lines.extend([
            "Fastest recorded experiment:",
            f"  Rows: {int(fastest_row['records_used']):,}",
            (
                "  Execution time: "
                f"{fastest_row['execution_time_seconds']:.3f} seconds"
            ),
            "",
        ])

    largest = df.dropna(
        subset=["records_used"]
    )

    if not largest.empty:
        largest_row = largest.loc[
            largest["records_used"].idxmax()
        ]

        lines.extend([
            "Largest recorded experiment:",
            f"  Rows processed: {int(largest_row['records_used']):,}",
            (
                "  Execution time: "
                f"{largest_row['execution_time_seconds']:.3f} seconds"
                if pd.notna(largest_row["execution_time_seconds"])
                else "  Execution time: unavailable"
            ),
            (
                "  Peak sampled memory: "
                f"{largest_row['peak_process_memory_mb']:.2f} MB"
                if pd.notna(largest_row["peak_process_memory_mb"])
                else "  Peak sampled memory: unavailable"
            ),
            (
                f"  MAE: {largest_row['mae']:.4f}"
                if pd.notna(largest_row["mae"])
                else "  MAE: unavailable"
            ),
            (
                f"  RMSE: {largest_row['rmse']:.4f}"
                if pd.notna(largest_row["rmse"])
                else "  RMSE: unavailable"
            ),
            "",
        ])

    lines.extend([
        "INTERPRETATION",
        "-" * 55,
        (
            "Execution time and memory measurements describe the "
            "recorded runs on this machine."
        ),
        (
            "Repeat experiments to assess timing variability before "
            "drawing strong scalability conclusions."
        ),
        (
            "Peak memory is sampled periodically and may miss brief "
            "memory spikes."
        ),
        (
            "The pipeline collects selected records into a pandas "
            "DataFrame before training, so it is not fully out-of-core."
        ),
        (
            "Model metrics may change with dataset size. Compare them "
            "carefully, and use a consistent evaluation split when "
            "making formal model-performance claims."
        ),
        "",
        "Result files included:",
    ])

    for filename in df["source_file"]:
        lines.append(f"  - {filename}")

    with open(
        report_path, "w", encoding="utf-8"
    ) as file:
        file.write("\n".join(lines) + "\n")

    print("\nReport saved to:", report_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("=" * 55)
    print("NYC TLC SCALING ANALYSIS")
    print("=" * 55)

    df = load_results()

    print(f"\nLoaded {len(df)} scaling result files.")

    save_tables(df)
    create_graphs(df)
    create_report(df)

    print("\nAll analysis tasks completed.")
    print("Output directory:", ANALYSIS_DIR)


if __name__ == "__main__":
    main()