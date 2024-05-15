# Scores the IRI results of our model compared to the validation output

import argparse
import pathlib

import numpy as np
import pandas as pd
import simple_cache
from tqdm import tqdm

import get_images
import get_predictions
import show_img

PREDICTIONS_DIR = pathlib.Path(__file__).parent / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

SEED = 42
# Set random seed for reproducibility
np.random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse validation and predictions CSV files"
    )
    parser.add_argument(
        "--validation-csv",
        default="validation/dummy_validation_data.csv",
        type=str,
        help="Path to the validation CSV file",
    )
    parser.add_argument(
        "--n-samples", default=None, type=int, help="Number of samples to use"
    )
    parser.add_argument(
        "--linear-validation-set",
        action="store_true",
        help="Use a validation set that scales lineraly from 0 to 100 PCI",
    )
    return parser.parse_args()


def resample_data(
    df: pd.DataFrame, latitude_resolution: float, longitude_resolution: float
) -> pd.DataFrame:
    """Resample the data to the nearest latitude and longitude resolution."""
    df["QUANTIZED_LAT"] = (
        np.floor(df["LATITUDE"] / latitude_resolution) * latitude_resolution
    )
    df["QUANTIZED_LON"] = (
        np.floor(df["LONGITUDE"] / longitude_resolution) * longitude_resolution
    )
    print(
        f"Resampled data from {len(df)} to {len(df.groupby(['QUANTIZED_LAT', 'QUANTIZED_LON']))} rows"
    )
    if len(df) == 0:
        raise ValueError(
            f"No data after resampling with latitude resolution {latitude_resolution} and longitude resolution {longitude_resolution}"
        )
    return df.groupby(["QUANTIZED_LAT", "QUANTIZED_LON"]).first().reset_index(drop=True)


def get_predictions_(
    validation_csv: pathlib.Path,
    n_samples: int = None,
    linear_validation_set: bool = False,
) -> pd.DataFrame:
    """Get predictions from the model for the given validation CSV."""
    print(f"Getting predictions for {validation_csv}")
    # Load validation csv as pandas dataframe.
    val_df = pd.read_csv(validation_csv)
    print(f"Loaded {len(val_df)} rows from {validation_csv}")

    if linear_validation_set:
        print(f"Using a linear validation set")
        new_val_df = pd.DataFrame()
        # Filter the DataFrame to get one row per PCI value from 0 to 100
        new_val_df = (
            val_df[val_df["PCI"].isin(range(101))]
            .groupby("PCI")
            .head(1)
            .reset_index(drop=True)
        )
        print(f"Using a linear validation set with {len(new_val_df)} samples")
        # Order by PCI
        val_df = new_val_df

    if n_samples is None:
        n_samples = len(val_df)

    # Sample top n-samples
    val_df = val_df.sample(n_samples, random_state=SEED)

    val_df = val_df.sort_values(by="PCI")

    # Estimated date of PCI measurement
    timestamp = pd.to_datetime("2024-04-23")
    predictions = []
    for lat, lon, pci in tqdm(
        val_df[["LATITUDE", "LONGITUDE", "PCI"]].values,
        total=len(val_df),
        unit="row",
    ):
        print(f"Getting predictions for latitude: {lat}, longitude: {lon}")
        image_paths = get_images.get_images(lat, lon, num_images=3)
        pci_preds = get_predictions.get_predictions(image_paths)

        valid_preds = []
        # If any predictions are strings instead of numbers
        for pci_pred, image_path in zip(pci_preds, image_paths):
            show_img.show_images(image_paths)
            if isinstance(pci_pred, str):
                print(f"Invalid prediction: {pci_pred} for image {image_path}")
            else:
                valid_preds.append(pci_pred)
        pci_preds = valid_preds
        if len(pci_preds) == 0:
            print(
                f"No valid predictions for image at {lat}, {lon}. pci_preds: {pci_preds}"
            )
            pci_pred = np.nan
        else:
            pci_pred = np.min(pci_preds)
        print(
            f"\tPredictions: {pci_preds}, final prediction: {pci_pred}, actual PCI: {pci}"
        )
        predictions.append(
            {
                "TIMESTAMP": timestamp,
                "LATITUDE": lat,
                "LONGITUDE": lon,
                "PCI": pci,
                "PCI_pred": pci_pred,
            }
        )
    predictions_df = pd.DataFrame(predictions)
    return predictions_df


def get_metrics(predictions_df: pd.DataFrame) -> dict:
    """Calculate the metrics for the given predictions."""
    # Remove nan rows
    # Get number of nan rows
    n_nan = predictions_df.isna().sum().sum()
    if n_nan > 0:
        print(f"Removed {n_nan} rows with NaN values")
    predictions_df = predictions_df.dropna()
    mae = np.mean(np.abs(predictions_df["PCI"] - predictions_df["PCI_pred"]))
    rmse = np.sqrt(np.mean((predictions_df["PCI"] - predictions_df["PCI_pred"]) ** 2))
    return {"MAE": mae, "RMSE": rmse}


if __name__ == "__main__":
    # Example usage
    args = parse_args()
    validation_csv = pathlib.Path(args.validation_csv)
    predictions_df = get_predictions_(
        validation_csv,
        n_samples=args.n_samples,
        linear_validation_set=args.linear_validation_set,
    )
    # Get datetime.now formatted nicely for a file name
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    predictions_csv = (
        PREDICTIONS_DIR / f"{validation_csv.stem}_{timestamp}_predictions.csv"
    )
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"Saved predictions to {predictions_csv}")

    metrics = get_metrics(predictions_df)
    print(f"Metrics: {metrics}")
