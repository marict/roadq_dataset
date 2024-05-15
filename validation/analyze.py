import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

GROUND_TRUTH_TIMESTAMP = pd.Timestamp("2024-04-23 00:00:00")


def parse_args() -> argparse.ArgumentParser:
    """Parse command line arguments.

    Returns:
        argparse.ArgumentParser: The command line arguments parser.
    """
    parser = argparse.ArgumentParser(
        description="Scatter plot PCI and PCI_pred values from a CSV file."
    )
    parser.add_argument(
        "--predictions-csv",
        type=str,
        required=True,
        help="Path to the predictions CSV file.",
    )
    return parser


def load_data(csv_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(csv_path)


def plot(df: pd.DataFrame) -> None:
    """Scatter plot the PCI and PCI_pred values.

    Args:
        df (pd.DataFrame): DataFrame containing 'PCI', 'PCI_pred', and 'PCI_pred_normalized' columns.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["PCI"],
        df["PCI_pred"],
        alpha=0.5,
        edgecolors="k",
        color="blue",
        label="PCI vs PCI_pred",
    )
    plt.xlabel("PCI")
    plt.ylabel("PCI_pred / PCI_pred_normalized")
    plt.title("Scatter Plot of PCI vs PCI_pred and PCI vs PCI_pred_normalized")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.clf()

    # Create a scatter plot of PCI_SquaredError vs TimeStampDelta
    if "PCI_SquaredError" in df.columns and "TimeStampDeltaMonths" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df["TimeStampDeltaMonths"],
            df["PCI_SquaredError"],
            alpha=0.5,
            edgecolors="k",
            color="green",
            label="PCI_SquaredError vs TimeStampDeltaMonths",
        )
        plt.xlabel("TimeStampDeltaMonths")
        plt.ylabel("PCI_SquaredError")
        plt.title("Scatter Plot of PCI_SquaredError vs TimeStampDeltaMonths")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.clf()


def main() -> None:
    """Main function to parse arguments, load data, and plot predictions."""
    parser = parse_args()
    args = parser.parse_args()

    csv_path = args.predictions_csv
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f"The file {csv_path} does not exist.")

    df = load_data(csv_path)

    min_value = df["PCI_pred"].min()
    max_value = df["PCI_pred"].max()
    df["PCI_pred_normalized"] = (
        (df["PCI_pred"] - min_value) / (max_value - min_value)
    ) * 100

    df["PRED_TIMESTAMP"] = pd.to_datetime(df["PRED_TIMESTAMP"])
    df["TimeStampDelta"] = (
        GROUND_TRUTH_TIMESTAMP - df["PRED_TIMESTAMP"]
    ).dt.total_seconds()
    # Convert TimeStampDelta from seconds to months
    seconds_per_month = 30.44 * 24 * 60 * 60
    df["TimeStampDeltaMonths"] = df["TimeStampDelta"] / seconds_per_month

    df["PCI_SquaredError"] = (df["PCI_pred"] - df["PCI"]) ** 2

    # Order by TimestampDelta
    df = df.sort_values(by="TimeStampDeltaMonths")

    if "PCI" not in df.columns or "PCI_pred" not in df.columns:
        raise ValueError("The CSV file must contain 'PCI' and 'PCI_pred' columns.")

    plot(df)


if __name__ == "__main__":
    main()
