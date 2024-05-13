import argparse
import csv
import io
import pathlib

import requests
import simple_cache
from PIL import Image

import creds
import show_img

# Get location of this file
THIS_DIR = pathlib.Path(__file__).parent
IMAGES_DIR = THIS_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)


# Add arguments for latitude and longitude
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lat", type=float, help="Latitude of the location to fetch Street View image."
    )
    parser.add_argument(
        "lon", type=float, help="Longitude of the location to fetch Street View image."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output directory to save the downloaded files.",
    )
    parser.add_argument(
        "--show-image", action="store_true", help="Display the downloaded image."
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to extract, the heading will be divided into equal parts starting at 0 and ending at 360",
    )
    args = parser.parse_args()
    return args


def get_street_view_details(
    lat: float,
    lon: float,
    heading: int = 0,
    pitch: int = 0,
    fov: int = 120,
    size: str = "600x300",
    verbose=False,
):
    """Fetches the capture date and image URL for the closest Street View image to the given coordinates."""
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    image_url_base = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "location": f"{lat},{lon}",
        "key": creds.GOOGLE_API_KEY,
        "size": size,
        "fov": fov,
        "heading": heading,
        "pitch": pitch,
        "radius": "1000",
        "source": "outdoor",
    }
    if verbose:
        print(f"Retrieving Street View image metadata for location: {lat},{lon}")
    # Fetch metadata
    metadata_response = requests.get(metadata_url, params=params)
    if metadata_response.status_code == 200:
        metadata = metadata_response.json()
        # Construct image URL
        image_url = requests.Request("GET", image_url_base, params=params).prepare().url
        return image_url, metadata
    else:
        return "Failed to retrieve metadata", None


def get_street_view_image(image_url: str, verbose=False):
    """Downloads the Street View image from the given URL."""
    # Add API key as query parameter to image_url
    image_url += f"&key={creds.GOOGLE_API_KEY}"
    # Get local filename from image_url
    if verbose:
        print(f"Retrieving Street View image from URL: {image_url}")
    response = requests.get(image_url)
    if response.status_code == 200:
        return response.content
    print(
        f"Failed to download image from URL: {image_url}, status code: {response.status_code}, response: {response.text}"
    )
    return None


@simple_cache.cache_it("img_cache.cache")
def get_image(
    lat: float,
    lon: float,
    heading: int = 0,
    pitch: int = -30,
    fov: int = 120,
    size: str = "600x300",
    verbose=False,
):
    # Retrieves metadata then retrieves the image from streetview api
    if lat < -90 or lat > 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if lon < -180 or lon > 180:
        raise ValueError("Longitude must be between -180 and 180.")
    if heading < 0 or heading > 360:
        raise ValueError("Heading must be between 0 and 360.")
    if fov < 10 or fov > 120:
        raise ValueError("Field of view must be between 10 and 120.")
    if size not in ["600x300", "400x400", "800x400"]:
        raise ValueError("Size must be one of '600x300', '400x400', or '800x400'.")
    image_url, metadata = get_street_view_details(
        lat, lon, pitch=pitch, heading=heading, fov=fov, size=size
    )
    if metadata["status"] == "ZERO_RESULTS" or metadata["status"] == "NOT_FOUND":
        print(f"No Street View image found for the given location: {lat}, {lon}")
    elif image_url is not None:
        if verbose:
            print(f"Image URL: {image_url}")
        # Download image
        image = get_street_view_image(image_url)
        return image, metadata
    else:
        return None, None


def center_crop_image(image: bytes, percent: int = 90) -> bytes:
    """Crops the image to the given size."""
    if not (0 < percent <= 100):
        raise ValueError("Percent must be between 1 and 100.")

    # Open the image from bytes
    img = Image.open(io.BytesIO(image))

    # Calculate the cropping dimensions
    width, height = img.size
    new_width = int(width * percent / 100)
    new_height = int(height * percent / 100)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # Save the cropped image back to bytes
    img_byte_arr = io.BytesIO()
    cropped_img.save(img_byte_arr, format=img.format)
    cropped_img_bytes = img_byte_arr.getvalue()

    return cropped_img_bytes


def get_images(
    lat: float,
    lon: float,
    num_images: int = 1,
    show_image: bool = False,
    record_location: bool = False,
    verbose=False,
) -> list[pathlib.Path]:
    image_paths = []
    if num_images < 1:
        raise ValueError("Number of images must be greater than 0.")
    if num_images > 5:
        raise ValueError("Number of images must be less than or equal to 5.")
    headings = [0]
    if num_images > 1:
        headings = [heading for heading in range(0, 360, 360 // num_images)]
    if verbose:
        print(f"Generating images for headings: {headings}")
    for heading in headings:
        output_dir = pathlib.Path(IMAGES_DIR)
        image, metadata = get_image(lat, lon, heading=heading)
        if image is None:
            raise ValueError("Failed to download image.")

        image = center_crop_image(image)

        # Remove dots from lat and lon
        lat_str = str(lat).replace(".", "dot")
        lon_str = str(lon).replace(".", "dot")
        date_captured = metadata.get("date", "no-date")
        output_file = (
            output_dir / f"streetview_{date_captured}_{lat_str}_{lon_str}_{heading}.jpg"
        )
        if record_location:
            with open(r"location-data.csv", "a") as f:
                location_data = [
                    metadata["location"]["lat"],
                    metadata["location"]["lng"],
                    f"streetview_{metadata['date']}_{lat_str}_{lon_str}_{heading}.jpg",
                ]
                writer = csv.writer(f)
                writer.writerow(location_data)
        with open(output_file, "wb") as file:
            file.write(image)
        if verbose:
            print(f"Image saved to {output_file}")

        if show_image:
            show_img.show_image(output_file)
        image_paths.append(output_file)
    return image_paths


if __name__ == "__main__":
    args = parse_args()
    lat = args.lat
    lon = args.lon
    show_image = args.show_image
    num_images = args.num_images
    print(f"Fetching {num_images} image(s) for location: {lat}, {lon}")
    get_images(lat, lon, num_images, show_image)
