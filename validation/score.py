# Scores the IRI results of our model compared to the validation output

import argparse
import math
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
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan along segments to determine the score for the entire segment. Note that every sample is 10 meters apart.",
    )
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Display images that the model is predicting on.",
    )
    return parser.parse_args()


def get_prediction(lat: int, lon: int, show_images: bool = False):
    print(f"Getting predictions for latitude: {lat}, longitude: {lon}")
    image_paths, timestamp = get_images.get_images(lat, lon, num_images=2)
    pci_preds = get_predictions.get_predictions(image_paths)

    valid_preds = []
    for pci_pred, image_path in zip(pci_preds, image_paths):
        if show_images:
            show_img.show_images(image_paths)
        # If any predictions are strings instead of numbers.
        if isinstance(pci_pred, str):
            print(f"Invalid prediction: {pci_pred} for image {image_path}")
        else:
            valid_preds.append(pci_pred)
    pci_preds = valid_preds
    if len(pci_preds) == 0:
        print(f"No valid predictions for image at {lat}, {lon}. pci_preds: {pci_preds}")
        pci_pred = np.nan
    else:
        pci_pred = np.min(pci_preds)
    print(f"\tPredictions: {pci_preds}, final prediction: {pci_pred}")
    return pci_pred, timestamp


def get_predictions_(
    validation_csv: pathlib.Path,
    n_samples: int = None,
    linear_validation_set: bool = False,
    scan: bool = False,
    show_images: bool = False,
) -> pd.DataFrame:
    """Get predictions from the model for the given validation CSV."""
    print(f"Getting predictions for {validation_csv}")
    # Load validation csv as pandas dataframe.
    val_df = pd.read_csv(validation_csv)
    val_df = val_df.sort_values(by="PCI")

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

    if scan:
        # Group by segment ID
        segments = val_df.groupby("SEGID").first().reset_index()
        # Order by PCI
        segments = segments.sort_values(by="PCI")
        if n_samples is None:
            n_samples = len(segments)
        segments = segments.sample(n_samples, random_state=SEED)
        predictions = []
        # For each segment, get all samples
        for i, (seg_id, pci) in enumerate(segments[["SEGID", "PCI"]].values):
            segment_df = val_df[val_df["SEGID"] == seg_id].reset_index(drop=True)
            print(
                f"Scanning segment {seg_id} (segment {i}/{len(segments)}) of length: {len(segment_df)} with PCI {pci}"
            )
            score = math.inf  # Initialize score to infinity
            min_timestamp = pd.Timestamp.max
            for index, segment in segment_df.iterrows():
                lat, lon = segment["LATITUDE"], segment["LONGITUDE"]
                pci_pred, timestamp = get_prediction(lat, lon, show_images=show_images)
                score = min(score, pci_pred)
                min_timestamp = min(min_timestamp, timestamp)
                print(
                    f"sample {index+1}/{len(segment_df)} Current score: {score}, real pci: {pci}"
                )
            predictions.append(
                {
                    "SEGID": seg_id,
                    "PCI": pci,
                    "PCI_pred": score,
                    "PRED_TIMESTAMP": min_timestamp,
                }
            )
            print(
                f"Final pred_pci for segment {seg_id}: {score} with actual PCI {pci}, min_timestamp {min_timestamp}"
            )
    else:
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
            pci_pred, pred_timestamp = get_prediction(lat, lon)
            predictions.append(
                {
                    "TIMESTAMP": timestamp,
                    "LATITUDE": lat,
                    "LONGITUDE": lon,
                    "PCI": pci,
                    "PCI_pred": pci_pred,
                    "PRED_TIMESTAMP": pred_timestamp,
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
        scan=args.scan,
        show_images=args.show_images,
    )
    # Convert args into a format that can be used in the file name
    args_str = "_".join(
        [
            f"{key}={value}"
            for key, value in vars(args).items()
            if key != "validation_csv"
        ]
    )
    # Get datetime.now formatted nicely for a file name
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    predictions_csv = (
        PREDICTIONS_DIR
        / f"{timestamp}_{validation_csv.stem}_{args_str}_predictions.csv"
    )
    predictions_df.to_csv(predictions_csv, index=False)
    print(f"Saved predictions to {predictions_csv}")

    metrics = get_metrics(predictions_df)
    print(f"Metrics: {metrics}")
