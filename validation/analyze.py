import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def plot_predictions(df: pd.DataFrame) -> None:
    """Scatter plot the PCI and PCI_pred values.

    Args:
        df (pd.DataFrame): DataFrame containing 'PCI' and 'PCI_pred' columns.
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
    plt.ylabel("PCI_pred")
    plt.title("Scatter Plot of PCI vs PCI_pred")
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    """Main function to parse arguments, load data, and plot predictions."""
    parser = parse_args()
    args = parser.parse_args()

    csv_path = args.predictions_csv
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f"The file {csv_path} does not exist.")

    df = load_data(csv_path)

    if "PCI" not in df.columns or "PCI_pred" not in df.columns:
        raise ValueError("The CSV file must contain 'PCI' and 'PCI_pred' columns.")

    plot_predictions(df)


if __name__ == "__main__":
    main()
