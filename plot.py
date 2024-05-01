import pathlib
import webbrowser
from typing import List, Tuple

import folium
import argparse

# Get location of this file
THIS_DIR = pathlib.Path(__file__).parent
IMAGES_DIR = THIS_DIR / "maps"
IMAGES_DIR.mkdir(exist_ok=True)

def parse_args() -> argparse.ArgumentParser:
    """Parse command line arguments for rectangle coordinates."""
    parser = argparse.ArgumentParser(description="Plot coordinates and rectangles on a map.")
    parser.add_argument('--rectangles', type=str, nargs='+', help="Pairs of tuples for rectangles specified as 'lat1,lon1,lat2,lon2'")
    return parser.parse_args()

def plot_coordinates_on_map(
    coordinates: List[Tuple[float, float]] = None,
    rectangles: List[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    map_title: str = "Map of Coordinates",
) -> None:
    """Plot a list of latitude and longitude tuples and rectangles on a map using Folium and automatically open it in a browser."""
    if not coordinates:
        # Get coordinates from recstangles
        coordinates = []
        for rect in rectangles:
            coordinates.extend(rect)

    # Calculate the center of the map
    lat_avg = sum(coord[0] for coord in coordinates) / len(coordinates)
    lon_avg = sum(coord[1] for coord in coordinates) / len(coordinates)

    # Create a map centered around the average latitude and longitude
    map_obj = folium.Map(
        location=[lat_avg, lon_avg], zoom_start=12, tiles="OpenStreetMap"
    )

    # Add a marker for each coordinate
    for lat, lon in coordinates:
        folium.Marker([lat, lon], popup=f"({lat}, {lon})").add_to(map_obj)

    if rectangles:
        # Add rectangles to the map
        for bottom_left, top_right in rectangles:
            bounds = [bottom_left, top_right]
            folium.Rectangle(
                bounds=bounds,
                color="blue",
                fill=True,
                fill_color="cyan",
                fill_opacity=0.5,
            ).add_to(map_obj)

    # Add a title to the map
    title_html = (
        f"""<h3 align="center" style="font-size:20px"><b>{map_title}</b></h3>"""
    )
    map_obj.get_root().html.add_child(folium.Element(title_html))

    # Save to HTML and automatically open it in a web browser
    output_file = IMAGES_DIR / "map.html"
    map_obj.save(output_file)
    webbrowser.open(f"file://{output_file}")

def main() -> None:
    args = parse_args()
    
    # Process the rectangle arguments into a list of tuples
    rectangles = []
    if args.rectangles:
        for rect in args.rectangles:
            lat1, lon1, lat2, lon2 = map(float, rect.split(','))
            rectangles.append(((lat1, lon1), (lat2, lon2)))
    
    plot_coordinates_on_map(rectangles=rectangles)

if __name__ == "__main__":
    main()