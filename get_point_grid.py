import argparse
from typing import List, Tuple

import numpy as np
import requests

import creds
import plot

MAX_POINTS = 10000
MIN_POINTS = 2


def parse_args() -> argparse.ArgumentParser:
    """Parse command line arguments for the grid coordinates and API key, including grid resolution and maximum number of points."""
    parser = argparse.ArgumentParser(
        description="Generate a grid of coordinates and snap them to the nearest roads."
    )
    parser.add_argument("lat_start", type=float, help="Start latitude of the grid.")
    parser.add_argument("lon_start", type=float, help="Start longitude of the grid.")
    parser.add_argument("lat_end", type=float, help="End latitude of the grid.")
    parser.add_argument("lon_end", type=float, help="End longitude of the grid.")
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.01,
        help="Resolution of the grid in degrees.",
    )
    parser.add_argument(
        "--plot-points", action="store_true", help="Plot the snapped points on a map."
    )
    return parser.parse_args()


def generate_grid(
    lat_start: float, lat_end: float, lon_start: float, lon_end: float, step: float
) -> List[Tuple[float, float]]:
    """Generate a grid of latitude and longitude coordinates, correctly handling both positive and negative steps."""
    # Determine the direction of the step for latitude and longitude
    lat_step = step if lat_end > lat_start else -step
    lon_step = step if lon_end > lon_start else -step

    # Generate points using adjusted steps
    lat_points = np.arange(lat_start, lat_end, lat_step)
    lon_points = np.arange(lon_start, lon_end, lon_step)

    # Create grid using list comprehension
    return [(lat, lon) for lat in lat_points for lon in lon_points]


def snap_to_roads(coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Snap each coordinate independently to the nearest road using Google Roads API and return a list of lat/lon pairs."""
    base_url = "https://roads.googleapis.com/v1/snapToRoads"
    snapped_points = []
    print(f"Snapping {len(coordinates)} coordinates to the nearest road...")
    for lat, lon in coordinates:
        # Construct the path parameter for a single coordinate
        path_param = f"{lat},{lon}"
        params = {"path": path_param, "interpolate": False, "key": creds.GOOGLE_API_KEY}
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            # Parse response to extract the nearest road point
            results = response.json().get("snappedPoints", [])
            if results:
                point = results[
                    0
                ]  # Take the first snapped point if multiple are returned
                snapped_points.append(
                    (point["location"]["latitude"], point["location"]["longitude"])
                )
        else:
            print(f"API Request Failed for {lat},{lon}: {response.text}")

    return snapped_points


def get_point_grid(
    lat_start: float,
    lon_start: float,
    lat_end: float,
    lon_end: float,
    resolution: float,
    plot_points: bool = False,
) -> List[Tuple[float, float]]:
    if not (
        -90 <= lat_start <= 90
        and -90 <= lat_end <= 90
        and -180 <= lon_start <= 180
        and -180 <= lon_end <= 180
    ):
        print(
            f"Invalid latitude or longitude values: {lat_start}, {lon_start}, {lat_end}, {lon_end}"
        )
        return

    # Use the resolution from args
    grid_coordinates = generate_grid(lat_start, lat_end, lon_start, lon_end, resolution)
    print(f"Generated grid of size {len(grid_coordinates)}")
    if len(grid_coordinates) > MAX_POINTS:
        print(
            f"Generated grid of size {len(grid_coordinates)} exceeds the maximum number of allowed points: {MAX_POINTS}"
        )
        return None
    if len(grid_coordinates) < MIN_POINTS:
        print(
            f"Generated grid of size {len(grid_coordinates)} has less than the minimum number of required points: {MIN_POINTS}"
        )
        return None

    snapped_points = snap_to_roads(grid_coordinates)
    print(f"Received {len(snapped_points)} snapped points from the API.")
    if plot_points:
        # Define rectangle from the start and end points
        rectangle = [((lat_start, lon_start), (lat_end, lon_end))]
        plot.plot_coordinates_on_map(
            snapped_points, rectangles=rectangle, map_title="Snapped Points"
        )
    return snapped_points


def main():
    args = parse_args()
    snapped_points = get_point_grid(
        args.lat_start,
        args.lon_start,
        args.lat_end,
        args.lon_end,
        args.resolution,
        args.plot_points,
    )
    print(f"Snapped points: {snapped_points}")


if __name__ == "__main__":
    main()
