import argparse
from pathlib import Path
from typing import List

import folium
import pandas as pd
from folium.plugins import HeatMap

import get_images


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a PCI heat map using Folium.")
    parser.add_argument(
        "csv_file", type=Path, help="Path to the CSV file containing the data."
    )
    parser.add_argument(
        "output_html",
        type=Path,
        help="Path to save the output HTML file with the heat map.",
    )
    return parser


def create_heat_map(csv_file: Path, output_html: Path) -> None:
    """Create a PCI heat map from a CSV file using Folium."""
    # Load the data
    data = pd.read_csv(csv_file)

    # Filter out rows with missing PCI values
    data = data.dropna(subset=["pci"])

    # Convert PCI to float
    data["pci"] = data["pci"].astype(float)
    data["pci_normalized"] = 100 - data["pci"]

    # Find the location with the lowest PCI value
    lowest_pci_row = data.loc[data["pci"].idxmin()]

    # Create a base map
    m = folium.Map(location=[data["lat"].mean(), data["lon"].mean()], zoom_start=15)

    # Create a heat map
    heat_data = [
        [row["lat"], row["lon"], row["pci_normalized"]]
        for index, row in data.iterrows()
    ]
    HeatMap(heat_data, radius=20, blur=15).add_to(
        m
    )  # Adjusted radius and blur for more spread out heat map

    # Add a marker for the location with the lowest PCI
    folium.Marker(
        location=[lowest_pci_row["lat"], lowest_pci_row["lon"]],
        popup=f"Lowest PCI: {lowest_pci_row['pci']}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # Save the map to an HTML file
    m.save(output_html)

    # Call get_images function for the location with the lowest PCI
    get_images.get_images(
        lat=lowest_pci_row["lat"],
        lon=lowest_pci_row["lon"],
        show_image=True,
        num_images=3,
    )

    # Print the location with the lowest PCI
    print(
        f"Location with the lowest PCI: {lowest_pci_row['lat']}, {lowest_pci_row['lon']}"
    )


if __name__ == "__main__":
    args = parse_args().parse_args()
    create_heat_map(args.csv_file, args.output_html)
